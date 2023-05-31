use clap::Parser;
use std::{convert::Infallible, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    #[arg(long, short = 'p')]
    prompt: Option<String>,
    #[arg(long, short = 'v')]
    vocabulary_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    vocabulary_repository: Option<String>,
}
impl Args {
    pub fn to_vocabulary_source(&self) -> llm::VocabularySource {
        match (&self.vocabulary_path, &self.vocabulary_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --vocabulary-path and --vocabulary-repository");
            }
            (Some(path), None) => llm::VocabularySource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::VocabularySource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::VocabularySource::Model,
        }
    }
}

fn main() {
    let args = Args::parse();

    let vocabulary_source = args.to_vocabulary_source();
    let model_architecture = args.model_architecture;
    let model_path = args.model_path;
    let prompt = args
        .prompt
        .as_deref()
        .unwrap_or("Rust is a cool programming language because");

    let now = std::time::Instant::now();

    let model = llm::load_dynamic(
        model_architecture,
        &model_path,
        vocabulary_source,
        Default::default(),
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
