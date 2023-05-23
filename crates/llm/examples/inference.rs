use llm::{
    load_progress_callback_stdout as load_callback, InferenceFeedback, InferenceRequest,
    InferenceResponse, ModelArchitecture, VocabularySource,
};
use std::{convert::Infallible, io::Write, path::Path};

fn main() {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.len() < 2 {
        println!("Usage: cargo run --release --example inference <model_architecture> <model_path> [prompt] [overrides, json]");
        std::process::exit(1);
    }

    let model_architecture: ModelArchitecture = raw_args[0].parse().unwrap();
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
        VocabularySource::ModelEmbedded,
        Default::default(),
        overrides,
        load_callback,
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
        &InferenceRequest {
            prompt: prompt.into(),
            ..Default::default()
        },
        // OutputRequest
        &mut Default::default(),
        |r| match r {
            InferenceResponse::PromptToken(t) | InferenceResponse::InferredToken(t) => {
                print!("{t}");
                std::io::stdout().flush().unwrap();

                Ok(InferenceFeedback::Continue)
            }
            _ => Ok(InferenceFeedback::Continue),
        },
    );

    match res {
        Ok(result) => println!("\n\nInference stats:\n{result}"),
        Err(err) => println!("\n{err}"),
    }
}
