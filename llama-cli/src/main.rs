use std::{convert::Infallible, io::Write, path::Path};

use cli_args::CLI_ARGS;
use llama_rs::{
    InferenceError, InferenceParameters, InferenceSession, InferenceSessionParameters, Model,
    ModelKVMemoryType, TokenBias, Vocabulary, EOD_TOKEN_ID,
};
use rand::{thread_rng, SeedableRng};
use rustyline::error::ReadlineError;

mod cli_args;

fn enter_prompt(prompt: &str, session: &mut InferenceSession, model: &llama_rs::Model, vocab: &Vocabulary, params: &InferenceParameters) -> Result<(), InferenceError> {
    let mut sp = spinners::Spinner::new(spinners::Spinners::Dots2, "".to_string());
    session.feed_prompt::<Infallible>(model, vocab, params, &prompt, |_| Ok(()))?;
    sp.stop();
    Ok(())
}

fn recieve_reply(session: &mut InferenceSession, model: &llama_rs::Model, vocab: &Vocabulary, params: &InferenceParameters) -> Result<(), InferenceError> {
    let mut rng = thread_rng();

    session.inference_with_prompt::<Infallible>(
        model,
        vocab,
        params,
        "",
        CLI_ARGS.num_predict,
        &mut rng,
        |tk| {
            print!("{tk}");
            std::io::stdout().flush().unwrap();
            Ok(())
        },
    )?;

    Ok(())
}

fn repl_mode(
    chat_rules: Option<String>,
    user_prompt: &str,
    model: &llama_rs::Model,
    vocab: &llama_rs::Vocabulary,
    params: &InferenceParameters,
    session_params: &InferenceSessionParameters,
) {
    let mut reset_session = true;
    let mut session  = model.start_session(*session_params);
    let mut rl = rustyline::DefaultEditor::new().unwrap();

    if chat_rules.is_some() {
        println!("Entering chat mode using the given rules. State is kept between prompts.")
    } else {
        println!("Entering repl mode. State is not kept between prompts.")
    }

    loop {
        if chat_rules.is_none() || reset_session {
            session = model.start_session(*session_params);

            if let Some(ref chat_rules) = chat_rules{
                enter_prompt(&chat_rules, &mut session, model, vocab, params).expect("Chat rules exceed window length.");
                reset_session = false;
            }
        }

        let readline = rl.readline(">> ");

        match readline {
            Ok(line) => {
                let prompt = user_prompt.replace("$PROMPT", &line);

                if let Err(InferenceError::ContextFull) = enter_prompt(&prompt, &mut session, model, vocab, params) {
                    log::error!("Prompt exceeds context window length");
                    reset_session = true;
                    continue;
                }

                if let Err(InferenceError::ContextFull) = recieve_reply(&mut session, model, vocab, params) {
                    log::error!("Reply exceeds context window length");
                    reset_session = true;
                }

                println!();
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(err) => {
                log::error!("{err}");
            }
        }
    }
}

fn dump_tokens(text: &str, vocab: &Vocabulary) -> Result<(), InferenceError> {
    let toks = match vocab.tokenize(text, false) {
        Ok(toks) => toks,
        Err(e) => {
            log::error!("Could not tokenize prompt: {e}");
            return Err(e);
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

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = &*CLI_ARGS;

    let inference_params = InferenceParameters {
        n_threads: args.num_threads as i32,
        n_batch: args.batch_size,
        top_k: args.top_k,
        top_p: args.top_p,
        repeat_penalty: args.repeat_penalty,
        temp: args.temp,
        bias_tokens: args.token_bias.clone().unwrap_or_else(|| {
            if args.ignore_eos {
                TokenBias::new(vec![(EOD_TOKEN_ID, -1.0)])
            } else {
                TokenBias::default()
            }
        }),
        play_back_previous_tokens: false,
        ..Default::default()
    };
    let inference_session_params = {
        let mem_typ = if args.float16 {
            ModelKVMemoryType::Float16
        } else {
            ModelKVMemoryType::Float32
        };
        InferenceSessionParameters {
            memory_k_type: mem_typ,
            memory_v_type: mem_typ,
            repetition_penalty_last_n: args.repeat_last_n,
        }
    };

    let chat_rules = if let Some(path) = &args.chat_rules_file {
        match std::fs::read_to_string(path) {
            Ok(mut chat_rules) => {
                // Strip off the last character if it's exactly newline. Also strip off a single
                // carriage return if it's there. Since String must be valid UTF-8 it should be
                // guaranteed that looking at the string as bytes here is safe: UTF-8 non-ASCII
                // bytes will always the high bit set.
                if matches!(chat_rules.as_bytes().last(), Some(b'\n')) {
                    chat_rules.pop();
                }
                if matches!(chat_rules.as_bytes().last(), Some(b'\r')) {
                    chat_rules.pop();
                }
                Some(chat_rules)
            }
            Err(err) => {
                log::error!("Could not read chat rules file at {path}. Error {err}");
                std::process::exit(1);
            }
        }
    } else if let Some(chat_rules) = &args.chat_rules {
        Some(chat_rules.clone())
    } else {
        None
    };

    let prompt = if let Some(path) = &args.prompt_file {
        match std::fs::read_to_string(path) {
            Ok(mut prompt) => {
                // Strip off the last character if it's exactly newline. Also strip off a single
                // carriage return if it's there. Since String must be valid UTF-8 it should be
                // guaranteed that looking at the string as bytes here is safe: UTF-8 non-ASCII
                // bytes will always the high bit set.
                if matches!(prompt.as_bytes().last(), Some(b'\n')) {
                    prompt.pop();
                }
                if matches!(prompt.as_bytes().last(), Some(b'\r')) {
                    prompt.pop();
                }
                prompt
            }
            Err(err) => {
                log::error!("Could not read prompt file at {path}. Error {err}");
                std::process::exit(1);
            }
        }
    } else if let Some(prompt) = &args.prompt {
        prompt.clone()
    } else {
        log::error!("No prompt or prompt file was provided. See --help");
        std::process::exit(1);
    };

    let (model, vocab) =
        llama_rs::Model::load(&args.model_path, args.num_ctx_tokens as i32, |progress| {
            use llama_rs::LoadProgress;
            match progress {
                LoadProgress::HyperparametersLoaded(hparams) => {
                    log::debug!("Loaded HyperParams {hparams:#?}")
                }
                LoadProgress::BadToken { index } => {
                    log::info!("Warning: Bad token in vocab at index {index}")
                }
                LoadProgress::ContextSize { bytes } => log::info!(
                    "ggml ctx size = {:.2} MB\n",
                    bytes as f64 / (1024.0 * 1024.0)
                ),
                LoadProgress::MemorySize { bytes, n_mem } => log::info!(
                    "Memory size: {} MB {}",
                    bytes as f32 / 1024.0 / 1024.0,
                    n_mem
                ),
                LoadProgress::PartLoading {
                    file,
                    current_part,
                    total_parts,
                } => log::info!(
                    "Loading model part {}/{} from '{}'\n",
                    current_part,
                    total_parts,
                    file.to_string_lossy(),
                ),
                LoadProgress::PartTensorLoaded {
                    current_tensor,
                    tensor_count,
                    ..
                } => {
                    if current_tensor % 8 == 0 {
                        log::info!("Loaded tensor {current_tensor}/{tensor_count}");
                    }
                }
                LoadProgress::PartLoaded {
                    file,
                    byte_size,
                    tensor_count,
                } => {
                    log::info!("Loading of '{}' complete", file.to_string_lossy());
                    log::info!(
                        "Model size = {:.2} MB / num tensors = {}",
                        byte_size as f64 / 1024.0 / 1024.0,
                        tensor_count
                    );
                }
            }
        })
        .expect("Could not load model");

    log::info!("Model fully loaded!");

    if args.dump_prompt_tokens {
        dump_tokens(&prompt, &vocab).ok();
        return;
    }

    let mut rng = if let Some(seed) = CLI_ARGS.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    let (mut session, session_loaded) = {
        fn load_snapshot_from_disk(model: &Model, path: &Path) -> InferenceSession {
            let snapshot = snapshot::load_from_disk(path);
            match snapshot.and_then(|snapshot| model.session_from_snapshot(snapshot)) {
                Ok(session) => {
                    log::info!("Loaded inference session from {path:?}");
                    session
                }
                Err(err) => {
                    eprintln!("Could not load inference session. Error: {err}");
                    std::process::exit(1);
                }
            }
        }

        match (&args.persist_session, &args.load_session) {
            (Some(path), _) if path.exists() => (load_snapshot_from_disk(&model, path), true),
            (_, Some(path)) => (load_snapshot_from_disk(&model, path), true),
            _ => (model.start_session(inference_session_params), false),
        }
    };

    if args.repl {
        repl_mode(
            chat_rules,
            &prompt,
            &model,
            &vocab,
            &inference_params,
            &inference_session_params,
        );
    } else {
        let inference_params = if session_loaded {
            InferenceParameters {
                play_back_previous_tokens: true,
                ..inference_params
            }
        } else {
            inference_params
        };

        let res = session.inference_with_prompt::<Infallible>(
            &model,
            &vocab,
            &inference_params,
            &prompt,
            args.num_predict,
            &mut rng,
            |t| {
                print!("{t}");
                std::io::stdout().flush().unwrap();

                Ok(())
            },
        );
        println!();

        match res {
            Ok(_) => (),
            Err(llama_rs::InferenceError::ContextFull) => {
                log::warn!("Context window full, stopping inference.")
            }
            Err(llama_rs::InferenceError::TokenizationFailed) => {
                log::error!("Failed to tokenize initial prompt.");
            }
            Err(llama_rs::InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
        }

        if let Some(session_path) = args.save_session.as_ref().or(args.persist_session.as_ref()) {
            // Write the memory to the cache file
            // SAFETY: no other model functions used inside the block
            unsafe {
                match snapshot::write_to_disk(&session.get_snapshot(), session_path) {
                    Ok(_) => {
                        log::info!("Successfully wrote session to {session_path:?}");
                    }
                    Err(err) => {
                        log::error!("Could not write session at {session_path:?}: {err}");
                        std::process::exit(1);
                    }
                }
            }
        }
    }
}

mod snapshot {
    use llama_rs::{InferenceSnapshot, InferenceSnapshotRef, SnapshotError};
    use std::{
        fs::File,
        io::{BufReader, BufWriter},
        path::Path,
    };
    use zstd::zstd_safe::CompressionLevel;

    const SNAPSHOT_COMPRESSION_LEVEL: CompressionLevel = 1;

    pub fn load_from_disk(path: impl AsRef<Path>) -> Result<InferenceSnapshot, SnapshotError> {
        let mut reader =
            zstd::stream::read::Decoder::new(BufReader::new(File::open(path.as_ref())?))?;
        InferenceSnapshot::read(&mut reader)
    }

    pub fn write_to_disk(
        snap: &InferenceSnapshotRef<'_>,
        path: impl AsRef<Path>,
    ) -> Result<(), SnapshotError> {
        let mut writer = zstd::stream::write::Encoder::new(
            BufWriter::new(File::create(path.as_ref())?),
            SNAPSHOT_COMPRESSION_LEVEL,
        )?
        .auto_finish();

        snap.write(&mut writer)
    }
}
