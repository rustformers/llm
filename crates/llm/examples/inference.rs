use clap::Parser;
use llm::{
    load_progress_callback_stdout as load_callback, InferenceFeedback, InferenceRequest,
    InferenceResponse, ModelArchitecture,
};
use std::{convert::Infallible, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    model_architecture: ModelArchitecture,
    model_path: PathBuf,
    prompt: Option<String>,

    #[arg(short, long)]
    overrides: Option<String>,

    #[arg(short, long)]
    vocabulary_path: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    let model_architecture: ModelArchitecture = args.model_architecture;
    let model_path = args.model_path;
    let prompt = args
        .prompt
        .as_deref()
        .unwrap_or("Rust is a cool programming language because");
    let overrides = args.overrides.map(|s| serde_json::from_str(&s).unwrap());

    let now = std::time::Instant::now();

    let model = llm::load_dynamic(
        model_architecture,
        &model_path,
        args.vocabulary_path.as_deref(),
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
            prompt,
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
