use clap::Parser;
use llm_base::conversation_inference_callback;
use rustyline::error::ReadlineError;
use std::{convert::Infallible, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,
}
impl Args {
    pub fn to_tokenizer_source(&self) -> llm::TokenizerSource {
        match (&self.tokenizer_path, &self.tokenizer_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --tokenizer-path and --tokenizer-repository");
            }
            (Some(path), None) => llm::TokenizerSource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::TokenizerSource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::TokenizerSource::Embedded,
        }
    }
}

fn main() {
    let args = Args::parse();

    let tokenizer_source = args.to_tokenizer_source();
    let model_architecture = args.model_architecture;
    let model_path = args.model_path;
    let model = llm::load_dynamic(
        Some(model_architecture),
        &model_path,
        tokenizer_source,
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });

    let mut session = model.start_session(Default::default());

    let character_name = "### Assistant";
    let user_name = "### Human";
    let persona = "A chat between a human and an assistant.";
    let history = format!(
        "{character_name}: Hello - How may I help you today?\n\
         {user_name}: What is the capital of France?\n\
         {character_name}:  Paris is the capital of France."
    );

    let inference_parameters = llm::InferenceParameters::default();

    session
        .feed_prompt(
            model.as_ref(),
            format!("{persona}\n{history}").as_str(),
            &mut Default::default(),
            llm::feed_prompt_callback(|resp| match resp {
                llm::InferenceResponse::PromptToken(t)
                | llm::InferenceResponse::InferredToken(t) => {
                    print_token(t);

                    Ok::<llm::InferenceFeedback, Infallible>(llm::InferenceFeedback::Continue)
                }
                _ => Ok(llm::InferenceFeedback::Continue),
            }),
        )
        .expect("Failed to ingest initial prompt.");

    let mut rl = rustyline::DefaultEditor::new().expect("Failed to create input reader");

    let mut rng = rand::thread_rng();
    let mut res = llm::InferenceStats::default();

    loop {
        println!();
        let readline = rl.readline(format!("{user_name}: ").as_str());
        print!("{character_name}:");
        match readline {
            Ok(line) => {
                let stats = session
                    .infer::<Infallible>(
                        model.as_ref(),
                        &mut rng,
                        &llm::InferenceRequest {
                            prompt: format!("{user_name}: {line}\n{character_name}:")
                                .as_str()
                                .into(),
                            parameters: &inference_parameters,
                            play_back_previous_tokens: false,
                            maximum_token_count: None,
                        },
                        &mut Default::default(),
                        conversation_inference_callback(&format!("{character_name}:"), print_token),
                    )
                    .unwrap_or_else(|e| panic!("{e}"));

                res.feed_prompt_duration = res
                    .feed_prompt_duration
                    .saturating_add(stats.feed_prompt_duration);
                res.prompt_tokens += stats.prompt_tokens;
                res.predict_duration = res.predict_duration.saturating_add(stats.predict_duration);
                res.predict_tokens += stats.predict_tokens;
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(err) => {
                println!("{err}");
            }
        }
    }

    println!("\n\nInference stats:\n{res}");
}

fn print_token(t: String) {
    print!("{t}");
    std::io::stdout().flush().unwrap();
}
