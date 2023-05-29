use clap::Parser;
use std::{convert::Infallible, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    architecture: String,
    path: PathBuf,
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
    let architecture = args.architecture.parse().unwrap();
    let path = args.path;
    let prompt = args
        .prompt
        .as_deref()
        .unwrap_or("Rust is a cool programming language because");

    let now = std::time::Instant::now();

    let model = llm::load_dynamic(
        architecture,
        &path,
        vocabulary_source,
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| panic!("Failed to load {architecture} model from {path:?}: {err}"));

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
            parameters: &llm::InferenceParameters::default(),
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
