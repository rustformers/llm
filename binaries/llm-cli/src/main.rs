use std::{
    convert::Infallible,
    fs::File,
    io::{BufReader, BufWriter, Write},
};

use clap::Parser;
use cli_args::{Args, BaseArgs};
use color_eyre::eyre::{Context, Result};
use llm::InferenceError;
use rustyline::error::ReadlineError;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{history::DefaultHistory, Cmd, Event, EventHandler, KeyCode, KeyEvent, Modifiers};
use rustyline::{Completer, Helper, Highlighter, Hinter};

mod cli_args;
mod snapshot;

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();
    color_eyre::install()?;

    let cli_args = Args::parse();
    match &cli_args {
        Args::Llama { args } => handle_args::<llm::models::Llama>(args),
        Args::Bloom { args } => handle_args::<llm::models::Bloom>(args),
        Args::Gpt2 { args } => handle_args::<llm::models::Gpt2>(args),
        Args::GptJ { args } => handle_args::<llm::models::GptJ>(args),
        Args::NeoX { args } => handle_args::<llm::models::NeoX>(args),
    }
}

fn handle_args<M: llm::KnownModel + 'static>(args: &cli_args::BaseArgs) -> Result<()> {
    match args {
        BaseArgs::Infer(args) => infer::<M>(args),
        BaseArgs::Info(args) => info::<M>(args),
        BaseArgs::PromptTokens(args) => prompt_tokens::<M>(args),
        BaseArgs::Repl(args) => interactive::<M>(args, false),
        BaseArgs::Chat(args) => interactive::<M>(args, true),
        BaseArgs::Quantize(args) => quantize::<M>(args),
    }
}

fn infer<M: llm::KnownModel + 'static>(args: &cli_args::Infer) -> Result<()> {
    let prompt = load_prompt_file_with_prompt(&args.prompt_file, args.prompt.as_deref());
    let inference_session_config = args.generate.inference_session_config();
    let model = args.model_load.load::<M>()?;
    let (mut session, session_loaded) = snapshot::read_or_create_session(
        model.as_ref(),
        args.persist_session.as_deref(),
        args.generate.load_session.as_deref(),
        inference_session_config,
    );
    let inference_params = args.generate.inference_parameters(model.eot_token_id());

    let mut rng = args.generate.rng();
    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rng,
        &llm::InferenceRequest {
            prompt: &prompt,
            parameters: Some(&inference_params),
            play_back_previous_tokens: session_loaded,
            maximum_token_count: args.generate.num_predict,
        },
        // OutputRequest
        &mut Default::default(),
        |t| {
            print!("{t}");
            std::io::stdout().flush().unwrap();

            Ok(())
        },
    );
    println!();

    match res {
        Ok(_) => (),
        Err(InferenceError::ContextFull) => {
            log::warn!("Context window full, stopping inference.")
        }
        Err(InferenceError::TokenizationFailed) => {
            log::error!("Failed to tokenize initial prompt.");
        }
        Err(InferenceError::UserCallback(_)) | Err(InferenceError::EndOfText) => {
            unreachable!("cannot fail")
        }
    }

    if let Some(session_path) = args.save_session.as_ref().or(args.persist_session.as_ref()) {
        // Write the memory to the cache file
        snapshot::write_session(session, session_path);
    }

    Ok(())
}

fn info<M: llm::KnownModel + 'static>(args: &cli_args::Info) -> Result<()> {
    let file = File::open(&args.model_path)?;
    let mut reader = BufReader::new(&file);
    let mut loader: llm::Loader<M::Hyperparameters, _> = llm::Loader::new(|_| {
        // We purposely do not print progress here, as we are only interested in the metadata
    });

    llm::ggml_format::load(&mut reader, &mut loader)?;

    log::info!("Container type: {:?}", loader.container_type);
    log::info!("Hyperparameters: {:?}", loader.hyperparameters);
    log::info!(
        "Tensors: {:?}",
        loader
            .tensors
            .iter()
            .map(|(name, tensor)| format!("{} ({:?})", name, tensor.element_type))
            .collect::<Vec<_>>()
    );
    log::info!("Vocabulary size: {}", loader.vocabulary.id_to_token.len());

    if args.dump_vocabulary {
        log::info!("Dumping vocabulary:");
        for (tid, token) in loader.vocabulary.id_to_token.iter().enumerate() {
            log::info!("{}: {}", tid, utf8_or_array(token));
        }
    }

    fn utf8_or_array(token: &[u8]) -> String {
        std::str::from_utf8(token)
            .map(|s| s.to_owned())
            .unwrap_or(format!("{:?}", token))
    }

    Ok(())
}

fn prompt_tokens<M: llm::KnownModel + 'static>(args: &cli_args::PromptTokens) -> Result<()> {
    let prompt = load_prompt_file_with_prompt(&args.prompt_file, args.prompt.as_deref());
    let model = args.model_load.load::<M>()?;
    let toks = match model.vocabulary().tokenize(&prompt, false) {
        Ok(toks) => toks,
        Err(e) => {
            log::error!("Could not tokenize prompt: {e}");
            std::process::exit(1);
        }
    };
    log::info!("=== Dumping prompt tokens:");
    log::info!(
        "{}",
        toks.iter()
            .map(|(_, tid)| tid.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    log::info!(
        "{}",
        toks.iter()
            .map(|(s, tid)| format!("{s:?}:{tid}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    Ok(())
}

fn interactive<M: llm::KnownModel + 'static>(
    args: &cli_args::Repl,
    // If set to false, the session will be cloned after each inference
    // to ensure that previous state is not carried over.
    chat_mode: bool,
) -> Result<()> {
    let prompt_file = args.prompt_file.contents();
    let inference_session_config = args.generate.inference_session_config();
    let model = args.model_load.load::<M>()?;
    let (mut session, session_loaded) = snapshot::read_or_create_session(
        model.as_ref(),
        None,
        args.generate.load_session.as_deref(),
        inference_session_config,
    );
    let inference_params = args.generate.inference_parameters(model.eot_token_id());

    let mut rng = args.generate.rng();
    let mut rl = rustyline::Editor::<LineContinuationValidator, DefaultHistory>::new()?;
    rl.set_helper(Some(LineContinuationValidator));
    rl.bind_sequence(
        Event::KeySeq(vec![KeyEvent(KeyCode::Enter, Modifiers::SHIFT)]),
        EventHandler::Simple(Cmd::Newline),
    );

    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(raw_line) => {
                let session_backup = if chat_mode {
                    None
                } else {
                    Some(session.clone())
                };
                let line = raw_line.replace("\\\n", "\n");

                let prompt = prompt_file
                    .as_deref()
                    .map(|pf| process_prompt(pf, &line))
                    .unwrap_or(line);

                let sp = spinoff::Spinner::new(spinoff::spinners::Dots2, "".to_string(), None);
                if let Err(InferenceError::ContextFull) = session.feed_prompt::<Infallible>(
                    model.as_ref(),
                    &inference_params,
                    &prompt,
                    // OutputRequest
                    &mut Default::default(),
                    |_| Ok(()),
                ) {
                    log::error!("Prompt exceeds context window length.")
                };
                sp.clear();

                let res = session.infer::<Infallible>(
                    model.as_ref(),
                    &mut rng,
                    &llm::InferenceRequest {
                        prompt: "",
                        parameters: Some(&inference_params),
                        play_back_previous_tokens: session_loaded,
                        maximum_token_count: args.generate.num_predict,
                    },
                    // EvaluateOuputRequest
                    &mut Default::default(),
                    |tk| {
                        print!("{tk}");
                        std::io::stdout().flush().unwrap();
                        Ok(())
                    },
                );
                println!();

                if let Err(InferenceError::ContextFull) = res {
                    log::error!("Reply exceeds context window length");
                }

                if let Some(session_backup) = session_backup {
                    session = session_backup;
                }
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(err) => {
                log::error!("{err}");
            }
        }
    }

    Ok(())
}

fn quantize<M: llm::KnownModel + 'static>(args: &cli_args::Quantize) -> Result<()> {
    use llm::QuantizeProgress;

    let mut source = BufReader::new(std::fs::File::open(&args.source)?);
    let mut destination = BufWriter::new(std::fs::File::create(&args.destination)?);

    llm::quantize::<M, _, _>(
        &mut source,
        &mut destination,
        args.target.into(),
        |progress| match progress {
            QuantizeProgress::HyperparametersLoaded => log::info!("Loaded hyperparameters"),
            QuantizeProgress::TensorLoading {
                name,
                dims,
                element_type,
                n_elements,
            } => log::info!(
                "Loading tensor `{name}` ({n_elements} ({dims:?}) {element_type} elements)"
            ),
            QuantizeProgress::TensorQuantizing { name } => log::info!("Quantizing tensor `{name}`"),
            QuantizeProgress::TensorQuantized {
                name,
                original_size,
                reduced_size,
                history,
            } => log::info!(
            "Quantized tensor `{name}` from {original_size} to {reduced_size} bytes ({history:?})"
        ),
            QuantizeProgress::TensorSkipped { name, size } => {
                log::info!("Skipped tensor `{name}` ({size} bytes)")
            }
            QuantizeProgress::Finished {
                original_size,
                reduced_size,
                history,
            } => log::info!(
                "Finished quantization from {original_size} to {reduced_size} bytes ({history:?})"
            ),
        },
    )
    .wrap_err("failed to quantize model")
}

fn load_prompt_file_with_prompt(
    prompt_file: &cli_args::PromptFile,
    prompt: Option<&str>,
) -> String {
    if let Some(prompt_file) = prompt_file.contents() {
        if let Some(prompt) = prompt {
            process_prompt(&prompt_file, prompt)
        } else {
            prompt_file
        }
    } else if let Some(prompt) = prompt {
        prompt.to_owned()
    } else {
        log::error!("No prompt or prompt file was provided. See --help");
        std::process::exit(1);
    }
}

#[derive(Completer, Helper, Highlighter, Hinter, Debug, Clone, Copy)]
struct LineContinuationValidator;

impl Validator for LineContinuationValidator {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        if ctx.input().ends_with('\\') {
            Ok(ValidationResult::Incomplete)
        } else {
            Ok(ValidationResult::Valid(None))
        }
    }
}

fn process_prompt(raw_prompt: &str, prompt: &str) -> String {
    raw_prompt.replace("{{PROMPT}}", prompt)
}
