use std::{convert::Infallible, io::Write, path::Path};

fn main() {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.len() < 2 {
        println!("Usage: cargo run --release --example inference <model_architecture> <model_path> [prompt] [overrides, json]");
        std::process::exit(1);
    }

    let model_architecture: llm::ModelArchitecture = raw_args[0].parse().unwrap();
    let model_path = Path::new(&raw_args[1]);
    let prompt = raw_args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Rust is a cool programming language because");
    let overrides = raw_args.get(3).map(|s| serde_json::from_str(s).unwrap());

    let now = std::time::Instant::now();

    let model = llm::load_dynamic(
        model_architecture,
        model_path,
        Default::default(),
        overrides,
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    let mut session = model.start_session(Default::default());

    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: prompt.into(),
            parameters: &llm::InferenceParameters {
                n_threads: 8,
                n_batch: 8,
                sampler: &llm::samplers::TopPTopK::default(),
            },
            play_back_previous_tokens: false,
            maximum_token_count: None,
        },
        // OutputRequest
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                print!("{t}");
                std::io::stdout().flush().unwrap();

                Ok(llm::InferenceFeedback::Continue)
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    );

    match res {
        Ok(result) => println!("\n\nInference stats:\n{result}"),
        Err(err) => println!("\n{err}"),
    }
}
