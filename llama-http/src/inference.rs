use llama_rs::{
    InferenceParameters, InferenceSession, InferenceSessionParameters, InferenceSnapshot,
    LoadProgress, ModelKVMemoryType, TokenBias,
};
use rand::thread_rng;
use std::convert::Infallible;

use crate::cli_args::CLI_ARGS;
use flume::{unbounded, Receiver, Sender};

#[derive(Debug)]
pub struct InferenceRequest {
    /// The channel to send the tokens to.
    pub tx_tokens: Sender<Result<String, hyper::Error>>,

    pub num_predict: Option<usize>,
    pub prompt: String,
    pub n_batch: Option<usize>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub temp: Option<f32>,
}

pub fn initialize_model_and_handle_inferences() -> Sender<InferenceRequest> {
    // Create a channel for InferenceRequests and spawn a thread to handle them

    let (tx, rx) = unbounded();

    std::thread::spawn(move || {
        let args = &*CLI_ARGS;

        let mut inference_session_manager = InferenceSessionManager::new();

        let rx: Receiver<InferenceRequest> = rx.clone();
        loop {
            if let Ok(inference_request) = rx.try_recv() {
                let mut session = inference_session_manager.get_session();
                let inference_params = InferenceParameters {
                    n_threads: args.num_threads as i32,
                    n_batch: inference_request.n_batch.unwrap_or(args.batch_size),
                    top_k: inference_request.top_k.unwrap_or(args.top_k),
                    top_p: inference_request.top_p.unwrap_or(args.top_p),
                    repeat_penalty: inference_request
                        .repeat_penalty
                        .unwrap_or(args.repeat_penalty),
                    temp: inference_request.temp.unwrap_or(args.temp),
                    bias_tokens: TokenBias::default(),
                };
                let mut rng = thread_rng();
                // Run inference
                let model = &(inference_session_manager.model);
                let vocabulary = &(inference_session_manager.vocabulary);
                session
                    .inference_with_prompt::<Infallible>(
                        model,
                        vocabulary,
                        &inference_params,
                        &inference_request.prompt,
                        inference_request.num_predict,
                        &mut rng,
                        {
                            let tx_tokens = inference_request.tx_tokens.clone();
                            move |t| {
                                let text = t.to_string();
                                match tx_tokens.send(Ok(text)) {
                                    Ok(_) => {
                                        log::debug!("Sent token {} to receiver.", t);
                                    }
                                    Err(_) => {
                                        // The receiver has been dropped.
                                        log::warn!("Could not send token to receiver.");
                                    }
                                }

                                Ok(())
                            }
                        },
                    )
                    .expect("Could not run inference");
            }

            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });

    tx
}

/// `InferenceSessionManager` is a way to create new sessions for a model and vocabulary.
/// In the future, it can also manage how many sessions are created and manage creating sessions
/// between threads.
struct InferenceSessionManager {
    model: llama_rs::Model,
    vocabulary: llama_rs::Vocabulary,
}

impl InferenceSessionManager {
    fn new() -> Self {
        // TODO It's not a great pattern to inject these arguments from CLI_ARGS.
        // If we ever wanted to support this struct in multiple places, please move the `args`
        // variable into properties of this struct.
        let args = &*CLI_ARGS;

        // Load model
        let (model, vocabulary) =
            llama_rs::Model::load(&args.model_path, args.num_ctx_tokens as i32, |progress| {
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

        Self { model, vocabulary }
    }

    fn get_session(&mut self) -> InferenceSession {
        // TODO It's not a great pattern to inject these arguments from CLI_ARGS.
        // If we ever wanted to support this struct in multiple places, please move the `args`
        // variable into properties of this struct.
        let args = &*CLI_ARGS;

        if let Some(restore_path) = &args.restore_prompt {
            let snapshot = InferenceSnapshot::load_from_disk(restore_path);
            match snapshot.and_then(|snapshot| self.model.session_from_snapshot(snapshot)) {
                Ok(session) => {
                    log::info!("Restored cached memory from {restore_path}");
                    session
                }
                Err(err) => {
                    panic!("Could not restore from snapshot. Error: {err}");
                }
            }
        } else {
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
            self.model.start_session(inference_session_params)
        }
    }
}
