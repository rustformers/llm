use std::io::Write;

use cli_args::CLI_ARGS;
use llama_rs::{InferenceParameters, ModelMemory};
use rand::thread_rng;

mod cli_args;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = &*CLI_ARGS;

    let inference_params = InferenceParameters {
        n_threads: args.num_threads as i32,
        n_predict: args.num_predict,
        n_batch: args.batch_size,
        top_k: args.top_k,
        top_p: args.top_p,
        repeat_last_n: args.repeat_last_n,
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

    let (mut model, vocab) = llama_rs::Model::load(
        &args.model_path,
        args.num_ctx_tokens as i32,
        args.repeat_last_n,
        |progress| {
            use llama_rs::LoadProgress;
            match progress {
                LoadProgress::HyperParamsLoaded(hparams) => {
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
        },
    )
    .expect("Could not load model");

    log::info!("Model fully loaded!");

    if let Some(restore_path) = &args.restore_prompt {
        let memory = ModelMemory::load_from_disk(restore_path);
        match memory.and_then(|memory| model.set_memory(memory)) {
            Ok(_) => {
                log::info!("Restored cached memory from {restore_path}");
            }
            Err(err) => {
                eprintln!("Could not restore prompt. Error: {err}");
                std::process::exit(1);
            }
        }
    }

    let mut rng = thread_rng();
    model.inference_with_prompt(&vocab, &inference_params, &prompt, &mut rng, |t| {
        print!("{t}");
        std::io::stdout().flush().unwrap();
    });
    println!();

    if let Some(cache_path) = &args.cache_prompt {
        // SAFETY: no other model functions used inside the block
        unsafe {
            let memory = model.get_memory();
            match memory.write_to_disk(cache_path) {
                Ok(_) => {
                    log::info!("Successfully written prompt cache to {cache_path}");
                }
                Err(err) => {
                    log::error!("Could not write prompt cache at {cache_path}: {err}");
                    std::process::exit(1);
                }
            }
        }
    }
}
