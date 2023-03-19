use llama_rs::{InferenceParameters, InferenceSnapshot};
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

        // TODO Preload prompt

        // Load model
        let (mut model, vocabulary) =
            llama_rs::Model::load(&args.model_path, args.num_ctx_tokens as i32, |_progress| {
                println!("Loading model...");
            })
            .expect("Could not load model");

        let mut session = if let Some(restore_path) = &args.restore_prompt {
            let snapshot = InferenceSnapshot::load_from_disk(restore_path);
            match snapshot.and_then(|snapshot| model.session_from_snapshot(snapshot)) {
                Ok(session) => {
                    println!("Restored cached memory from {restore_path}");
                    session
                }
                Err(err) => {
                    eprintln!("Could not restore prompt. Error: {err}");
                    std::process::exit(1);
                }
            }
        } else {
            model.start_session(args.repeat_last_n)
        };

        let mut rng = thread_rng();
        let rx: Receiver<InferenceRequest> = rx.clone();
        loop {
            if let Ok(inference_request) = rx.try_recv() {
                let inference_params = InferenceParameters {
                    n_threads: args.num_threads as i32,
                    n_batch: inference_request.n_batch.unwrap_or(args.batch_size),
                    top_k: inference_request.top_k.unwrap_or(args.top_k),
                    top_p: inference_request.top_p.unwrap_or(args.top_p),
                    repeat_penalty: inference_request
                        .repeat_penalty
                        .unwrap_or(args.repeat_penalty),
                    temp: inference_request.temp.unwrap_or(args.temp),
                };

                // Run inference
                session
                    .inference_with_prompt::<Infallible>(
                        &model,
                        &vocabulary,
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
                                        println!("Sent token {} to receiver.", t);
                                    }
                                    Err(_) => {
                                        // The receiver has been dropped.
                                        println!("Could not send token to receiver.");
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
