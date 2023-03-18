use std::convert::Infallible;
use hyper::{Body, Method, Request, Response, Server};
use hyper::service::{make_service_fn, service_fn};
use llama_rs::{InferenceParameters, InferenceSession};
use std::net::SocketAddr;
use futures::{SinkExt, channel::mpsc};
use flume::{Sender, unbounded};

use serde::Deserialize;

use rand::thread_rng;
mod cli_args;

use cli_args::CLI_ARGS;

#[derive(Debug, Deserialize)]
struct PredictionRequest {
    num_predict: Option<usize>,
    prompt: String,
}

async fn handle_request(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/stream") => {
            // Parse POST request body as a PredictionRequest
            let body = hyper::body::to_bytes(req.into_body()).await?;
            let prediction_request = match serde_json::from_slice::<PredictionRequest>(&body) {
                Ok(prediction_request) => prediction_request,
                Err(_) => {
                    // Return 400 bad request if the body could not be parsed
                    let response = Response::builder()
                        .status(400)
                        .body(Body::empty())
                        .unwrap();
                    return Ok(response);
                }
            };

            // Create a channel for the stream
            let (tx, rx) = unbounded();
            let response_stream = rx.into_stream();
            inference_with_prediction_request(prediction_request, tx);

            // Create a response with a streaming body
            let body = Body::wrap_stream(response_stream);
            // Create a response with a streaming body
            let response = Response::builder()
                .header("Content-Type", "text/plain")
                .body(body)
                .unwrap();
            Ok(response)
        },
        _ => {
            // Return 404 not found for any other request
            let response = Response::builder()
                .status(404)
                .body(Body::empty())
                .unwrap();
            Ok(response)
        }
    }
}


fn inference_with_prediction_request(prediction_request: PredictionRequest, tx: Sender<Result<String, hyper::Error>>) {
    let args = &*CLI_ARGS;

    let inference_params = InferenceParameters {
        n_threads: args.num_threads as i32,
        n_batch: args.batch_size,
        top_k: args.top_k,
        top_p: args.top_p,
        repeat_penalty: args.repeat_penalty,
        temp: args.temp,
    };

    // TODO Preload prompt
    

    // Load model
    let (model, vocabulary) = llama_rs::Model::load(&args.model_path, args.num_ctx_tokens as i32, |_progress| {
        println!("Loading model...");
    }).expect("Could not load model");

    let mut rng = thread_rng();

    let mut session = model.start_session(args.repeat_last_n);

    // print prompt
    println!("{}", prediction_request.prompt);

    session.inference_with_prompt::<Infallible>(
        &model,
        &vocabulary,
        &inference_params,
        &prediction_request.prompt,
        prediction_request.num_predict,
        &mut rng,
        {
            let tx = tx.clone();
            move |t| {
                // Send the generated text to the channel
                let text = t.to_string();
                println!("{}", text);
                tx.send(Ok(text)).unwrap();

                Ok(())
            }
        },
    ).expect("Could not run inference");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = &*CLI_ARGS;

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let server = Server::bind(&addr).serve(make_service_fn(|_| async {
        Ok::<_, hyper::Error>(service_fn(handle_request))
    }));

    println!("Listening on http://{}", addr);

    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }

    Ok(())
}
