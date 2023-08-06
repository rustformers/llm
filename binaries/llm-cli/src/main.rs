use std::{
    convert::Infallible,
    fs::File,
    io::{BufReader, BufWriter},
};

use clap::Parser;
use cli_args::Args;
use color_eyre::eyre::{self, Context, ContextCompat};
use is_terminal::IsTerminal;

mod cli_args;
mod interactive;
mod snapshot;
mod util;

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_ansi(std::io::stderr().is_terminal())
        .init();

    color_eyre::install()?;

    let args = Args::parse();
    match args {
        Args::Infer(args) => infer(&args),
        Args::Perplexity(args) => perplexity(&args),
        Args::Info(args) => info(&args),
        Args::PromptTokens(args) => prompt_tokens(&args),
        Args::Repl(args) => interactive::repl(&args),
        Args::Chat(args) => interactive::chat(&args),
        Args::Quantize(args) => quantize(&args),
    }
}

#[tracing::instrument(skip_all)]
fn infer(args: &cli_args::Infer) -> eyre::Result<()> {
    let prompt = load_prompt_file_with_prompt(&args.prompt_file, args.prompt.as_deref())?;
    let inference_session_config = args.generate.inference_session_config();
    let model = args.model_load.load(args.generate.use_gpu)?;

    let (mut session, session_loaded) = snapshot::read_or_create_session(
        model.as_ref(),
        args.persist_session.as_deref(),
        args.load_session.as_deref(),
        inference_session_config,
    );
    let parameters = args
        .generate
        .inference_parameters(model.eot_token_id(), model.tokenizer().len())?;

    let mut rng = args.generate.rng();

    let span = tracing::trace_span!("infer");

    span.in_scope(|| {
        // do work inside the span...
        let res = session.infer::<Infallible>(
            model.as_ref(),
            &mut rng,
            &llm::InferenceRequest {
                prompt: prompt.as_str().into(),
                parameters: &parameters,
                play_back_previous_tokens: session_loaded,
                maximum_token_count: args.generate.num_predict,
            },
            // OutputRequest
            &mut Default::default(),
            |r| {
                match r {
                    llm::InferenceResponse::PromptToken(t) if !args.hide_prompt => {
                        util::print_token(t)
                    }
                    llm::InferenceResponse::InferredToken(t) => util::print_token(t),
                    _ => {}
                }
                Ok(llm::InferenceFeedback::Continue)
            },
        );

        println!();

        match res {
            Ok(stats) => {
                if args.stats {
                    println!();
                    println!("{}", stats);
                    println!();
                }
            }
            Err(llm::InferenceError::ContextFull) => {
                log::warn!("Context window full, stopping inference.")
            }
            Err(llm::InferenceError::TokenizationFailed(err)) => {
                log::error!("A tokenization-related failure occurred: {}", err);
            }
            Err(llm::InferenceError::SamplerFailure(err)) => {
                log::error!("A sampling-related failure occurred: {}", err);
            }
            Err(llm::InferenceError::UserCallback(_)) | Err(llm::InferenceError::EndOfText) => {
                unreachable!("cannot fail")
            }
        }
    });

    if let Some(session_path) = args.save_session.as_ref().or(args.persist_session.as_ref()) {
        // Write the memory to the cache file
        snapshot::write_session(session, session_path);
    }

    Ok(())
}

fn perplexity(args: &cli_args::Perplexity) -> eyre::Result<()> {
    let prompt = load_prompt_file_with_prompt(&args.prompt_file, args.prompt.as_deref())?;
    let inference_session_config = args.generate.inference_session_config();
    let model = args.model_load.load(args.generate.use_gpu)?;
    let (mut session, _) =
        snapshot::read_or_create_session(model.as_ref(), None, None, inference_session_config);

    session.perplexity(model.as_ref(), prompt.as_str(), |chunk, perplexity| {
        println!("Perplexity[{chunk}]: {perplexity}");
    })?;

    Ok(())
}

fn info(args: &cli_args::Info) -> eyre::Result<()> {
    struct InfoVisitor<'a>(&'a cli_args::Info);
    impl llm::ModelArchitectureVisitor<eyre::Result<()>> for InfoVisitor<'_> {
        fn visit<M: llm::KnownModel + 'static>(&mut self) -> eyre::Result<()> {
            let args = self.0;

            let model_path = &args.model_and_tokenizer.model_path;
            let tokenizer = args.model_and_tokenizer.to_source()?.retrieve(model_path)?;

            let file = File::open(model_path)?;
            let mut reader = BufReader::new(&file);
            let mut loader: llm::Loader<M::Hyperparameters, _> =
                llm::Loader::new(tokenizer, |_| {
                    // We purposely do not print progress here, as we are only interested in the metadata
                });

            llm::ggml_format::load(&mut reader, &mut loader)?;

            log::info!("Container type: {:?}", loader.container_type);
            log::info!("Hyperparameters: {:?}", loader.hyperparameters);
            log::info!("Tokenizer vocabulary size: {}", loader.tokenizer.len());

            if args.tokenizer {
                log::info!("Tokens:");
                for i in 0..loader.tokenizer.len() {
                    log::info!("- {}: {}", i, utf8_or_array(&loader.tokenizer.token(i)));
                }
            }

            if args.tensors {
                log::info!("Tensors:");
                for (name, tensor) in &loader.tensors {
                    log::info!("- {} ({:?} {:?})", name, tensor.element_type, tensor.dims());
                }
            }

            fn utf8_or_array(token: &[u8]) -> String {
                std::str::from_utf8(token)
                    .map(|s| s.to_owned())
                    .unwrap_or(format!("{:?}", token))
            }

            Ok(())
        }
    }

    args.model_and_tokenizer
        .architecture
        .model_architecture
        .wrap_err("a model architecture is required at present")?
        .visit(&mut InfoVisitor(args))
}

fn prompt_tokens(args: &cli_args::PromptTokens) -> eyre::Result<()> {
    let prompt = load_prompt_file_with_prompt(&args.prompt_file, args.prompt.as_deref())?;
    let model = args.model_load.load(false)?;
    let toks = match model.tokenizer().tokenize(&prompt, false) {
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

fn quantize(args: &cli_args::Quantize) -> eyre::Result<()> {
    use llm::QuantizeProgress;

    struct QuantizeVisitor<'a>(&'a cli_args::Quantize);
    impl llm::ModelArchitectureVisitor<eyre::Result<()>> for QuantizeVisitor<'_> {
        fn visit<M: llm::KnownModel>(&mut self) -> eyre::Result<()> {
            let args = self.0;

            let mut source: BufReader<File> = BufReader::new(std::fs::File::open(&args.source)?);
            let mut destination: BufWriter<File> =
                BufWriter::new(std::fs::File::create(&args.destination)?);
            let tokenizer: llm::Tokenizer = args.tokenizer.to_source()?.retrieve(&args.source)?;

            llm::quantize::<M, _, _>(
                &mut source,
                &mut destination,
                tokenizer,
                args.container_type.into(),
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
    }

    args.architecture
        .model_architecture
        .wrap_err("the architecture must be known for quantization")?
        .visit(&mut QuantizeVisitor(args))
}

fn load_prompt_file_with_prompt(
    prompt_file: &cli_args::PromptFile,
    prompt: Option<&str>,
) -> eyre::Result<String> {
    Ok(match (prompt_file.contents()?, prompt) {
        (Some(prompt_file), None) => prompt_file,
        (None, Some(prompt)) => prompt.to_owned(),
        (Some(prompt_file), Some(prompt)) => util::process_prompt(&prompt_file, prompt),
        (None, None) => eyre::bail!("No prompt or prompt file was provided. See --help"),
    })
}
