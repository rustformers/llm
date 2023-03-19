use std::{cell::RefCell, convert::Infallible};

use llama_rs::{InferenceParameters, Model, Vocabulary};
use rand::SeedableRng;
use tauri::Window;

pub fn load_model(path: &str) -> (Model, Vocabulary) {
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

#[derive(serde::Deserialize)]
pub struct Params {
    path: String,
    prompt: String,
    repeat_last_n: Option<usize>,
    n_batch: Option<usize>,
    n_threads: Option<usize>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repeat_penalty: Option<f32>,
    temp: Option<f32>,
    num_predict: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            path: "".to_string(),
            prompt: "".to_string(),
            repeat_last_n: Some(64),
            n_batch: Some(8),
            n_threads: Some(num_cpus::get_physical()),
            top_k: Some(40),
            top_p: Some(0.95),
            repeat_penalty: Some(1.3),
            temp: Some(0.8),
            num_predict: Some(512),
        }
    }
}

#[tauri::command(async)]
pub fn complete(window: Window, input: Params) -> String {
    let (model, vocab) = load_model(&input.path);
    let mut rng = rand::rngs::StdRng::from_entropy();
    let mut session = model.start_session(input.repeat_last_n.unwrap_or_default());

    let inference_params = InferenceParameters {
        n_threads: input.n_threads.unwrap_or_default() as i32,
        n_batch: input.n_batch.unwrap(),
        top_k: input.top_k.unwrap_or_default(),
        top_p: input.top_p.unwrap_or_default(),
        repeat_penalty: input.repeat_penalty.unwrap_or_default(),
        temp: input.temp.unwrap_or_default(),
    };
    let message = RefCell::new(String::new());
    let res = session.inference_with_prompt::<Infallible>(
        &model,
        &vocab,
        &inference_params,
        &input.prompt,
        Some(input.num_predict.unwrap_or_default()),
        &mut rng,
        |t| {
            message.borrow_mut().push_str(&t.to_string());
            println!("{}", t.to_string());
            window
                .emit(
                    "message",
                    Payload {
                        message: message.borrow_mut().to_string(),
                    },
                )
                .unwrap();
            Ok(())
        },
    );

    match res {
        Ok(_) => (),
        Err(llama_rs::InferenceError::ContextFull) => {
            log::warn!("Context window full, stopping inference.")
        }
        Err(llama_rs::InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
    }
    return message.borrow_mut().to_string();
}
