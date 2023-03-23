use llama_rs::{InferenceError, InferenceParameters, InferenceSnapshot, Model, Vocabulary};
use rand::SeedableRng;
use std::{convert::Infallible, io::Write};
use clap::{Parser, Subcommand};
use once_cell::sync::Lazy;


use Commands::LlamaCmd;
mod Commands;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    pub cmds: LlamaCmd,
}

/// CLI args are stored in a lazy static variable so they're accessible from
/// everywhere. Arguments are parsed on first access.
pub static CLI_ARGS: Lazy<Args> = Lazy::new(Args::parse);

fn read_prompt_from_file(path: &str) -> Result<String, String> {
    match std::fs::read_to_string(path) {
        Ok(prompt) => Ok(prompt),
        Err(err) => {
            log::error!("Could not read prompt file at {}. Error: {}", path, err);
            return Err(format!(
                "Could not read prompt file at {}. Error: {}",
                path, err
            ));
        }
    }
}

fn create_prompt(prompt: &str) -> Result<&str, String> {
    Ok(prompt)
}

fn select_model_and_vocab(args: &Args) -> Result<(Model, Vocabulary), String> {
    let (model, vocab) =
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
    Ok((model, vocab))
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

    let prompt = create_prompt(prompt).unwrap();
    let (mut model, vocab) = select_model_and_vocab(args).unwrap();

    fn create_seed(seed: u64) -> Result<String, String> {}

    // seed flag
    let mut rng = if let Some(seed) = CLI_ARGS.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    // restore_prompt flag
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

    if args.interactive {
        interactive_mode(&model, &vocab)
    } else if args.repl {
        repl_mode(&prompt, &model, &vocab, &inference_params);
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
            Ok(stats) => {
                println!("{}", stats);
            }
            Err(llama_rs::InferenceError::ContextFull) => {
                log::warn!("Context window full, stopping inference.")
            }
            Err(InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
        }
    }
}
