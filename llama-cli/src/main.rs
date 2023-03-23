use std::{convert::Infallible, io::Write};

use cli_args::CLI_ARGS;
use llama_rs::{
    InferenceError, InferenceParameters, InferenceSessionParameters, InferenceSnapshot,
    ModelKVMemoryType, TokenBias, Vocabulary, EOD_TOKEN_ID,
};
use rand::thread_rng;
use rand::SeedableRng;
use rustyline::error::ReadlineError;

mod cli_args;

fn repl_mode(
    prompt: &str,
    model: &llama_rs::Model,
    vocab: &llama_rs::Vocabulary,
    params: &InferenceParameters,
    session_params: &InferenceSessionParameters,
) {
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(line) => {
                let mut session = model.start_session(*session_params);
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

fn dump_tokens(text: &str, vocab: &Vocabulary) -> Result<(), InferenceError> {
    let toks = match vocab.tokenize(text, false) {
        Ok(toks) => toks,
        Err(e) => {
            log::error!("Could not tokenize prompt: {e}");
            return Err(e);
        }
    };
    log::info!("=== Dumping prompt tokens:");
    log::info!(
        "{}",
        toks.iter()
            .map(|(_, tid)| tid.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    log::info!(
        "{}",
        toks.iter()
            .map(|(s, tid)| format!("{s:?}:{tid}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
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
        bias_tokens: args.token_bias.clone().unwrap_or_else(|| {
            if args.ignore_eos {
                TokenBias::new(vec![(EOD_TOKEN_ID, -1.0)])
            } else {
                TokenBias::default()
            }
        }),
    };
    let inference_session_params = {
        let mem_typ = if args.float16 {
            ModelKVMemoryType::Float16
        } else {
            ModelKVMemoryType::Float32
        };
        InferenceSessionParameters {
            memory_k_type: mem_typ,
            memory_v_type: mem_typ,
            last_n_size: args.repeat_last_n,
        }
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

    if args.dump_prompt_tokens {
        dump_tokens(&prompt, &vocab).ok();
        return;
    }

    let mut rng = if let Some(seed) = CLI_ARGS.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

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
        model.start_session(inference_session_params)
    };

    if args.repl {
        repl_mode(
            &prompt,
            &model,
            &vocab,
            &inference_params,
            &inference_session_params,
        );
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
            Err(llama_rs::InferenceError::TokenizationFailed) => {
                log::error!("Failed to tokenize initial prompt. Exiting.");
                return;
            }
            Err(llama_rs::InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
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
            Ok(stats) => {
                println!("{}", stats);
            }
            Err(llama_rs::InferenceError::ContextFull) => {
                log::warn!("Context window full, stopping inference.")
            }
            Err(llama_rs::InferenceError::TokenizationFailed) => {
                log::error!("Failed to tokenize initial prompt.");
            }
            Err(llama_rs::InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
        }
    }
}
