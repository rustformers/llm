use std::{
    convert::Infallible,
    fmt,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek},
    path::Path,
};

use clap::Parser;
use cli_args::Args;
use color_eyre::eyre;
use is_terminal::IsTerminal;
use llm::ggml_format::gguf::{self, MetadataValue};

mod cli_args;
mod interactive;
mod snapshot;
mod util;

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing_subscriber::filter::LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .with_ansi(std::io::stderr().is_terminal())
        .init();

    color_eyre::install()?;

    let args = Args::parse();
    match args {
        Args::Infer(args) => infer(&args),
        Args::Perplexity(args) => perplexity(&args),
        Args::Gguf { gguf: args } => gguf(&args),
        Args::PromptTokens(args) => prompt_tokens(&args),
        Args::Repl(args) => interactive::repl(&args),
        Args::Chat(args) => interactive::chat(&args),
        // Args::Quantize(args) => quantize(&args),
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

fn gguf(args: &cli_args::Gguf) -> eyre::Result<()> {
    match args {
        cli_args::Gguf::Info(args) => info(args),
        cli_args::Gguf::Rebuild(args) => rebuild(args),
        cli_args::Gguf::AddHfTokenizer(args) => add_hf_tokenizer(args),
    }
}

fn info(args: &cli_args::Info) -> eyre::Result<()> {
    let model_path = &args.model_and_tokenizer.model_path;

    let file = File::open(model_path)?;
    let mut reader = BufReader::new(&file);
    let gguf = gguf::Gguf::load(&mut reader)?;

    log::info!("Non-array parameters:");
    for (metadata_key, metadata_value) in gguf.metadata.iter() {
        struct ValueDisplay<'a>(Option<&'a MetadataValue>);
        impl fmt::Debug for ValueDisplay<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if let Some(value) = self.0 {
                    write!(f, "{:?}", value)
                } else {
                    write!(f, "[elided due to size]")
                }
            }
        }

        let elide_due_to_size =
            metadata_value.as_array().is_some() || metadata_key == "tokenizer.huggingface.json";

        log::info!(
            "- {}: {:?}",
            metadata_key,
            ValueDisplay(if elide_due_to_size {
                None
            } else {
                Some(metadata_value)
            })
        );
    }

    if let Ok(tokenizer) = llm::tokenizer::GgufEmbeddedTokenizer::from_metadata(&gguf.metadata) {
        log::info!(
            "Embedded tokenizer vocabulary size: {}",
            tokenizer.tokens.len()
        );

        if args.tokenizer {
            log::info!("Embedded tokenizer vocabulary:");
            for (i, token) in tokenizer.tokens.iter().enumerate() {
                log::info!("- {}: {}", i, token);
            }

            if args.tensors {
                log::info!("Tensors:");
                for (name, tensor) in &gguf.tensor_infos {
                    log::info!(
                        "- {} ({:?} {:?})",
                        name,
                        tensor.element_type,
                        tensor.dimensions
                    );
                }
            }
        }
    }

    if args.tensors {
        log::info!("Tensors:");
        for (name, tensor) in &gguf.tensor_infos {
            log::info!(
                "- {} ({:?} {:?}) @ 0x{:X}",
                name,
                tensor.element_type,
                tensor.dimensions,
                tensor.offset
            );
        }
    }

    Ok(())
}

fn rebuild(args: &cli_args::Rebuild) -> eyre::Result<()> {
    rebuild_with_mutation(&args.input, &args.output, |_| Ok(()))
}

fn add_hf_tokenizer(args: &cli_args::AddHfTokenizer) -> eyre::Result<()> {
    let tokenizer =
        llm::tokenizer::huggingface_tokenizers::Tokenizer::from_pretrained(&args.tokenizer, None)
            .unwrap();

    rebuild_with_mutation(&args.input, &args.output, move |gguf| {
        let tokenizer = tokenizer.to_string(false).unwrap();
        gguf.metadata
            .insert("tokenizer.huggingface.json", tokenizer);

        Ok(())
    })
}

fn rebuild_with_mutation(
    input: &Path,
    output: &Path,
    mut mutator: impl FnMut(&mut gguf::Gguf) -> eyre::Result<()>,
) -> eyre::Result<()> {
    eyre::ensure!(input != output, "input and output must be different files");

    let input = File::open(input)?;
    let mut reader = BufReader::new(&input);
    let mut gguf = gguf::Gguf::load(&mut reader)?;

    let mut output = File::create(output)?;
    let mut writer = BufWriter::new(&mut output);

    mutator(&mut gguf)?;
    gguf.save(&mut writer, |writer, name, _info| {
        let reader = &mut reader;
        let original_info = gguf.tensor_infos.get(name).unwrap();

        reader.seek(std::io::SeekFrom::Start(
            gguf.tensor_data_position + original_info.offset,
        ))?;

        std::io::copy(&mut reader.take(original_info.calc_size() as u64), writer)?;

        Ok(())
    })?;

    Ok(())
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

// fn quantize(args: &cli_args::Quantize) -> eyre::Result<()> {
//     use llm::QuantizeProgress;

//     struct QuantizeVisitor<'a>(&'a cli_args::Quantize);
//     impl llm::ModelArchitectureVisitor<eyre::Result<()>> for QuantizeVisitor<'_> {
//         fn visit<M: llm::Model>(&mut self) -> eyre::Result<()> {
//             let args = self.0;

//             let mut source: BufReader<File> = BufReader::new(std::fs::File::open(&args.source)?);
//             let mut destination: BufWriter<File> =
//                 BufWriter::new(std::fs::File::create(&args.destination)?);
//             let tokenizer: llm::Tokenizer = args.tokenizer.to_source()?.retrieve(&args.source)?;

//             llm::quantize::<M, _, _>(
//                 &mut source,
//                 &mut destination,
//                 tokenizer,
//                 args.container_type.into(),
//                 args.target.into(),
//                 |progress| match progress {
//                     QuantizeProgress::HyperparametersLoaded => log::info!("Loaded hyperparameters"),
//                     QuantizeProgress::TensorLoading {
//                         name,
//                         dims,
//                         element_type,
//                         n_elements,
//                     } => log::info!(
//                         "Loading tensor `{name}` ({n_elements} ({dims:?}) {element_type} elements)"
//                     ),
//                     QuantizeProgress::TensorQuantizing { name } => log::info!("Quantizing tensor `{name}`"),
//                     QuantizeProgress::TensorQuantized {
//                         name,
//                         original_size,
//                         reduced_size,
//                         history,
//                     } => log::info!(
//                     "Quantized tensor `{name}` from {original_size} to {reduced_size} bytes ({history:?})"
//                 ),
//                     QuantizeProgress::TensorSkipped { name, size } => {
//                         log::info!("Skipped tensor `{name}` ({size} bytes)")
//                     }
//                     QuantizeProgress::Finished {
//                         original_size,
//                         reduced_size,
//                         history,
//                     } => log::info!(
//                         "Finished quantization from {original_size} to {reduced_size} bytes ({history:?})"
//                     ),
//                 },
//             )
//             .wrap_err("failed to quantize model")
//         }
//     }

//     args.architecture
//         .model_architecture
//         .wrap_err("the architecture must be known for quantization")?
//         .visit(&mut QuantizeVisitor(args))
// }

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
