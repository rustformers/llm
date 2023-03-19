use std::{convert::Infallible, io::Write, path::Path};

use clap::Parser;
use llama_rs::{InferenceParameters, InferenceSession, InferenceSnapshot, Model};
use rand::SeedableRng;

mod cli_args;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = cli_args::Args::parse();

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

    let mut rng = if let Some(seed) = args.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    let mut session = {
        fn load_snapshot_from_disk(model: &Model, path: &Path) -> InferenceSession {
            let snapshot = InferenceSnapshot::load_from_disk(path);
            match snapshot.and_then(|snapshot| model.session_from_snapshot(snapshot)) {
                Ok(session) => {
                    log::info!("Loaded inference session from {path:?}");
                    session
                }
                Err(err) => {
                    eprintln!("Could not load inference session. Error: {err}");
                    std::process::exit(1);
                }
            }
        }

        match (&args.persist_session, &args.load_session) {
            (Some(path), _) if path.exists() => load_snapshot_from_disk(&model, path),
            (_, Some(path)) => load_snapshot_from_disk(&model, path),
            _ => model.start_session(args.repeat_last_n),
        }
    };

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
        Err(llama_rs::InferenceError::ContextFull) => {
            log::warn!("Context window full, stopping inference.")
        }
        Err(llama_rs::InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
    }

    if let Some(session_path) = args.save_session.or(args.persist_session) {
        // Write the memory to the cache file
        // SAFETY: no other model functions used inside the block
        unsafe {
            let memory = session.get_snapshot();
            match memory.write_to_disk(&session_path) {
                Ok(_) => {
                    log::info!("Successfully wrote session to {session_path:?}");
                }
                Err(err) => {
                    log::error!("Could not write session at {session_path:?}: {err}");
                    std::process::exit(1);
                }
            }
        }
    }
}
