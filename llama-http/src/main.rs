use flume::{unbounded, Sender};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server};
use std::net::SocketAddr;

use serde::Deserialize;

mod cli_args;
mod inference;

use cli_args::CLI_ARGS;

#[derive(Clone)]
struct HttpContext {
    tx_inference_request: Sender<inference::InferenceRequest>,
}

/// The JSON POST request body for the /stream endpoint.
#[derive(Debug, Deserialize)]
struct InferenceHttpRequest {
    num_predict: Option<usize>,
    prompt: String,
    n_batch: Option<usize>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repeat_penalty: Option<f32>,
    temp: Option<f32>,
}

impl InferenceHttpRequest {
    /// This function is used to convert the HTTP request into an `inference::InferenceRequest`.
    /// This is is passed to the inference thread via a stream in the HTTP request.
    ///
    /// We cannot use the same `InferenceRequest` struct for parsing the HTTP response and
    /// requesting an inference because the
    /// inference thread needs to be able to send tokens back to the HTTP thread
    /// via a channel, and this cannot be serialized.
    fn to_inference_request(
        &self,
        tx_tokens: Sender<Result<String, hyper::Error>>,
    ) -> inference::InferenceRequest {
        inference::InferenceRequest {
            tx_tokens,
            num_predict: self.num_predict,
            prompt: self.prompt.clone(),
            n_batch: self.n_batch,
            top_k: self.top_k,
            top_p: self.top_p,
            repeat_penalty: self.repeat_penalty,
            temp: self.temp,
        }
    }
}

async fn handle_request(
    context: HttpContext,
    req: Request<Body>,
) -> Result<Response<Body>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/stream") => {
            // Parse POST request body as an InferenceHttpRequest
            let body = hyper::body::to_bytes(req.into_body()).await?;
            let inference_http_request = match serde_json::from_slice::<InferenceHttpRequest>(&body)
            {
                Ok(inference_http_request) => inference_http_request,
                Err(_) => {
                    // Return 400 bad request if the body could not be parsed
                    let response = Response::builder().status(400).body(Body::empty()).unwrap();
                    return Ok(response);
                }
            };

            // Create a channel for the stream
            let (tx_tokens, rx_tokens) = unbounded();

            // Send the prediction request to the inference thread
            let inference_request = inference_http_request.to_inference_request(tx_tokens);
            context.tx_inference_request.send(inference_request).expect(
                "Could not send request to inference thread - did the inference thread die?",
            );

            // Create a response channel.
            let (mut tx_http, rx_http) = Body::channel();
            tokio::spawn(async move {
                // Read tokens from the channel and send them to the response channel
                while let Ok(token) = rx_tokens.recv() {
                    let token = token.unwrap();

                    // Add a newline to the token to get around Hyper's buffering.
                    let token = format!("{}\n", token);

                    if let Err(error) = tx_http.send_data(token.into()).await {
                        eprintln!("Error sending data to client: {}", error);
                        break;
                    }
                }
            });

            // Create a response with a streaming body
            let response = Response::builder()
                .header("Content-Type", "text/plain")
                .body(rx_http)
                .unwrap();

            Ok(response)
        }
        _ => {
            // Return 404 not found for any other request
            let response = Response::builder().status(404).body(Body::empty()).unwrap();
            Ok(response)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = &*CLI_ARGS;

    let request_tx = inference::initialize_model_and_handle_inferences();

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    // Make HttpContext available to all requests
    let http_context = HttpContext {
        tx_inference_request: request_tx,
    };

    let server = Server::bind(&addr).serve(make_service_fn(move |_| {
        let http_context = http_context.clone();
        let service = service_fn(move |req| handle_request(http_context.clone(), req));
        async move { Ok::<_, hyper::Error>(service) }
    }));

    println!("Listening on http://{}", addr);

    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }

    Ok(())
}
