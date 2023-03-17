use std::io::Write;

use cli_args::CLI_ARGS;
use llama_rs::InferenceParameters;
use rand::thread_rng;
use rustyline::DefaultEditor;

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
        top_k: args.top_k as i32,
        top_p: args.top_p,
        repeat_last_n: args.repeat_last_n,
        repeat_penalty: args.repeat_penalty,
        temp: args.temp,
    };

    let repl_mode = args.repl.unwrap_or(false);

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
    } else if repl_mode {
        // Hack just to make things work for now, REPL ignores prompt CLI args
        "".to_string()
    } else {
        eprintln!("No prompt or prompt file was provided. See --help");
        std::process::exit(1);
    };

    let (model, vocab) =
        llama_rs::Model::load(&args.model_path, args.num_ctx_tokens as i32, |progress| {
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
        })
        .expect("Could not load model");

    log::info!("Model fully loaded!");

    let mut rng = thread_rng();

    if repl_mode {
        let mut rl = DefaultEditor::new().unwrap();
        loop {
            let readline = rl.readline(">> ");
            match readline {
                Ok(line) => {
                    let prompt = format!("
prompt: Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

{}

### Response:
", line);

                    model.inference_with_prompt(
                        &vocab,
                        &inference_params,
                        &prompt,
                        &mut rng,
                        |t| {
                            print!("{t}");
                            std::io::stdout().flush().unwrap();
                        },
                    );
                    println!();
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }
    } else {
        model.inference_with_prompt(&vocab, &inference_params, &prompt, &mut rng, |t| {
            print!("{t}");
            std::io::stdout().flush().unwrap();
        });
        println!();
    }
}
