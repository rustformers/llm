// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{cell::RefCell, convert::Infallible, io::Write};

use llama_rs::{InferenceParameters, Model, Vocabulary};
use rand::SeedableRng;
use tauri::Window;

fn load_model(path: &str) -> (Model, Vocabulary) {
    let (model, vocab) = llama_rs::Model::load(path, 512, |progress| {
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
    (model, vocab)
}

#[derive(Clone, serde::Serialize)]
struct Payload {
    message: String,
}

#[tauri::command(async)]
fn start(window: Window, path: &str, prompt: &str) {
    let (model, vocab) = load_model(path);
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut session = model.start_session(64);
    let inference_params = InferenceParameters {
        n_threads: num_cpus::get_physical() as i32,
        n_batch: 8,
        top_k: 40,
        top_p: 0.95,
        repeat_penalty: 1.3,
        temp: 0.8,
    };
    let message = RefCell::new(String::new());

    let res = session.inference_with_prompt::<Infallible>(
        &model,
        &vocab,
        &inference_params,
        &prompt,
        Some(512),
        &mut rng,
        |t| {
            message.borrow_mut().push_str(&t.to_string());

            window
                .emit(
                    "message",
                    Payload {
                        message: message.borrow_mut().to_string(),
                    },
                )
                .unwrap();
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
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![start])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
