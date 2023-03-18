use std::{convert::Infallible, io::Write};

use cli_args::CLI_ARGS;
use llama_rs::{InferenceError, InferenceParameters, InferenceSession, InferenceSnapshot};
use rand::thread_rng;
use rustyline::error::ReadlineError;

mod cli_args;

fn repl_mode(
    prompt: &str,
    model: &llama_rs::Model,
    vocab: &llama_rs::Vocabulary,
    params: &InferenceParameters,
    mut session: InferenceSession,
) {
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(line) => {
                session = model.start_session(CLI_ARGS.repeat_last_n);
                let prompt = prompt.replace("$PROMPT", &line);
                let mut rng = thread_rng();

                let mut sp = spinners::Spinner::new(spinners::Spinners::Dots2, "".to_string());
                if let Err(InferenceError::ContextFull) =
                    session.feed_prompt::<Infallible>(model, vocab, params, &prompt, |_| Ok(()))
                {
                    log::error!("Prompt exceeds context window length.")
                };
                sp.stop();

                let res = session.inference_with_prompt::<Infallible>(
                    model,
                    vocab,
                    params,
                    "",
                    CLI_ARGS.num_predict,
                    &mut rng,
                    |tk| {
                        print!("{tk}");
                        std::io::stdout().flush().unwrap();
                        Ok(())
                    },
                );
                println!();

                if let Err(InferenceError::ContextFull) = res {
                    log::error!("Reply exceeds context window length");
                }
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(err) => {
                log::error!("{err}");
            }
        }
    }
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = &*CLI_ARGS;

    let inference_params = InferenceParameters {
        n_threads: args.num_threads as i32,
        n_batch: args.batch_size,
        top_k: args.top_k,
        top_p: args.top_p,
        repeat_penalty: args.repeat_penalty,
        temp: args.temp,
    };

    let prompt = if let Some(path) = &args.prompt_file {
        match std::fs::read_to_string(path) {
            Ok(prompt) => prompt,
            Err(err) => {
                log::error!("Could not read prompt file at {path}. Error {err}");
                std::process::exit(1);
            }
        }
    } else if let Some(prompt) = &args.prompt {
        prompt.clone()
    } else {
        log::error!("No prompt or prompt file was provided. See --help");
        std::process::exit(1);
    };

    let (mut model, vocab) =
        llama_rs::Model::load(&args.model_path, args.num_ctx_tokens as i32, |progress| {
            use llama_rs::LoadProgress;
            match progress {
                LoadProgress::HyperparametersLoaded(hparams) => {
                    log::debug!("Loaded HyperParams {hparams:#?}")
                }
                LoadProgress::BadToken { index } => {
                    log::info!("Warning: Bad token in vocab at index {index}")
                }
                LoadProgress::ContextSize { bytes } => log::info!(
                    "ggml ctx size = {:.2} MB\n",
                    bytes as f64 / (1024.0 * 1024.0)
                ),
                LoadProgress::MemorySize { bytes, n_mem } => log::info!(
                    "Memory size: {} MB {}",
                    bytes as f32 / 1024.0 / 1024.0,
                    n_mem
                ),
                LoadProgress::PartLoading {
                    file,
                    current_part,
                    total_parts,
                } => log::info!(
                    "Loading model part {}/{} from '{}'\n",
                    current_part,
                    total_parts,
                    file.to_string_lossy(),
                ),
                LoadProgress::PartTensorLoaded {
                    current_tensor,
                    tensor_count,
                    ..
                } => {
                    if current_tensor % 8 == 0 {
                        log::info!("Loaded tensor {current_tensor}/{tensor_count}");
                    }
                }
                LoadProgress::PartLoaded {
                    file,
                    byte_size,
                    tensor_count,
                } => {
                    log::info!("Loading of '{}' complete", file.to_string_lossy());
                    log::info!(
                        "Model size = {:.2} MB / num tensors = {}",
                        byte_size as f64 / 1024.0 / 1024.0,
                        tensor_count
                    );
                }
            }
        })
        .expect("Could not load model");

    log::info!("Model fully loaded!");

    let mut rng = thread_rng();
    let mut session = if let Some(restore_path) = &args.restore_prompt {
        let snapshot = InferenceSnapshot::load_from_disk(restore_path);
        match snapshot.and_then(|snapshot| model.session_from_snapshot(snapshot)) {
            Ok(session) => {
                log::info!("Restored cached memory from {restore_path}");
                session
            }
            Err(err) => {
                log::error!("{err}");
                std::process::exit(1);
            }
        }
    } else {
        model.start_session(args.repeat_last_n)
    };

    if args.repl {
        repl_mode(&prompt, &model, &vocab, &inference_params, session);
    } else if let Some(cache_path) = &args.cache_prompt {
        let res =
            session.feed_prompt::<Infallible>(&model, &vocab, &inference_params, &prompt, |t| {
                print!("{t}");
                std::io::stdout().flush().unwrap();

                Ok(())
            });

        println!();

        match res {
            Ok(_) => (),
            Err(InferenceError::ContextFull) => {
                log::warn!(
                    "Context is not large enough to fit the prompt. Saving intermediate state."
                );
            }
            Err(InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
        }

        // Write the memory to the cache file
        // SAFETY: no other model functions used inside the block
        unsafe {
            let memory = session.get_snapshot();
            match memory.write_to_disk(cache_path) {
                Ok(_) => {
                    log::info!("Successfully written prompt cache to {cache_path}");
                }
                Err(err) => {
                    eprintln!("Could not restore prompt. Error: {err}");
                    std::process::exit(1);
                }
            }
        }
    } else {
        let res = session.inference_with_prompt::<Infallible>(
            &model,
            &vocab,
            &inference_params,
            &prompt,
            args.num_predict,
            &mut rng,
            |t| {
                print!("{t}");
                std::io::stdout().flush().unwrap();

                Ok(())
            },
        );
        println!();

        match res {
            Ok(_) => (),
            Err(InferenceError::ContextFull) => {
                log::warn!("Context window full, stopping inference.")
            }
            Err(InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
        }
    }
}
