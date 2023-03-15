use std::io::Write;

use cli_args::CLI_ARGS;
use llama_rs::{InferenceParams, ModelMemory};
use rand::thread_rng;

mod cli_args;

fn main() {
    env_logger::init();

    let args = &*CLI_ARGS;

    let inference_params = InferenceParams {
        n_threads: args.num_threads as i32,
        n_predict: args.num_predict,
        n_batch: args.batch_size,
        top_k: args.top_k as i32,
        top_p: args.top_p,
        repeat_last_n: args.repeat_last_n,
        repeat_penalty: args.repeat_penalty,
        temp: args.temp,
    };

    let prompt = if let Some(path) = &args.prompt_file {
        match std::fs::read_to_string(path) {
            Ok(prompt) => prompt,
            Err(err) => {
                eprintln!("Could not read prompt file at {path}. Error {err}");
                std::process::exit(1);
            }
        }
    } else if let Some(prompt) = &args.prompt {
        prompt.clone()
    } else {
        eprintln!("No prompt or prompt file was provided. See --help");
        std::process::exit(1);
    };

    let (mut model, vocab) = llama_rs::Model::load(&args.model_path, args.num_ctx_tokens as i32)
        .expect("Could not load model");

    if let Some(restore_path) = &args.restore_prompt {
        let memory = ModelMemory::load_compressed(restore_path);
        match memory.and_then(|memory| model.set_memory(memory)) {
            Ok(_) => (),
            Err(err) => {
                eprintln!("Could not restore prompt. Error: {err}");
                std::process::exit(1);
            }
        }
    }

    let mut rng = thread_rng();
    let stop_after_prompt = args.cache_prompt.is_some();
    model.inference_with_prompt(
        &vocab,
        &inference_params,
        &prompt,
        &mut rng,
        stop_after_prompt,
        |t| {
            print!("{t}");
            std::io::stdout().flush().unwrap();
        },
    );

    if let Some(cache_path) = &args.cache_prompt {
        // SAFETY: no other model functions used inside the block
        unsafe {
            let memory = model.get_memory();
            match memory.write_compressed(cache_path) {
                Ok(_) => (),
                Err(err) => {
                    eprintln!("Could not write prompt cache: {err}");
                    std::process::exit(1);
                }
            }
        }
    }

    println!();
}
