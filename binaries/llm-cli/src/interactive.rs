use std::convert::Infallible;

use color_eyre::eyre;
use rustyline::{
    error::ReadlineError,
    history::DefaultHistory,
    validate::{ValidationContext, ValidationResult, Validator},
    Cmd, Completer, Helper, Highlighter, Hinter, KeyCode, KeyEvent, Modifiers,
};

use crate::{
    cli_args::{Chat, Repl},
    snapshot, util,
};

pub fn repl(
    Repl {
        generate,
        model_load,
        prompt_file,
    }: &Repl,
) -> eyre::Result<()> {
    let (inference_session_config, parameters, model, mut rng) =
        initialize_common_state(generate, model_load)?;

    let template = prompt_file.contents()?;

    let model = model.as_ref();
    let mut session = create_session(model, inference_session_config);
    readline_loop(|raw_line| {
        let line = raw_line.replace("\\\n", "\n");

        let prompt = template
            .as_deref()
            .map(|template| util::process_prompt(template, &line))
            .unwrap_or(line);
        feed_prompt_with_spinner(model, &mut session, prompt)?;

        session.infer::<Infallible>(
            model,
            &mut rng,
            &llm::InferenceRequest {
                prompt: "".into(),
                parameters: &parameters,
                play_back_previous_tokens: false,
                maximum_token_count: generate.num_predict,
            },
            &mut Default::default(),
            |r| {
                if let llm::InferenceResponse::InferredToken(t) = r {
                    util::print_token(t);
                }
                Ok(llm::InferenceFeedback::Continue)
            },
        )?;

        if !session_ends_with_newline(&session) {
            println!();
        }
        session = create_session(model, inference_session_config);

        Ok(())
    })
}

pub fn chat(args: &Chat) -> eyre::Result<()> {
    let Chat {
        model_load,
        prelude_prompt_file,
        generate,
        ..
    } = args;

    let (inference_session_config, parameters, model, mut rng) =
        initialize_common_state(generate, model_load)?;

    let prelude_prompt = std::fs::read_to_string(prelude_prompt_file)?;
    let message_prompt_prefix = args.message_prompt_prefix()?;

    let model = model.as_ref();
    let mut session = create_session(model, inference_session_config);
    feed_prompt_with_spinner(model, &mut session, prelude_prompt)?;

    readline_loop(|raw_line| {
        let prompt = {
            let line = raw_line.replace("\\\n", "\n");
            let mut prompt = format!("{message_prompt_prefix}{line}");
            // Add a newline to the end of the prompt if it doesn't end with one
            if !prompt.ends_with('\n') {
                prompt.push('\n');
            }
            prompt
        };

        session.infer::<Infallible>(
            model,
            &mut rng,
            &llm::InferenceRequest {
                prompt: (&prompt).into(),
                parameters: &parameters,
                play_back_previous_tokens: false,
                maximum_token_count: generate.num_predict,
            },
            &mut Default::default(),
            llm::conversation_inference_callback(&message_prompt_prefix, util::print_token),
        )?;

        if !session_ends_with_newline(&session) {
            println!();
        }

        Ok(())
    })
}

fn initialize_common_state(
    generate: &crate::cli_args::Generate,
    model_load: &crate::cli_args::ModelLoad,
) -> eyre::Result<(
    llm::InferenceSessionConfig,
    llm::InferenceParameters,
    Box<dyn llm::Model>,
    rand::rngs::StdRng,
)> {
    let model = model_load.load(generate.use_gpu)?;
    Ok((
        generate.inference_session_config(),
        generate.inference_parameters(model.eot_token_id(), model.tokenizer().len())?,
        model,
        generate.rng(),
    ))
}

fn feed_prompt_with_spinner(
    model: &dyn llm::Model,
    session: &mut llm::InferenceSession,
    mut prompt: String,
) -> eyre::Result<()> {
    // Add a newline to the beginning of the prompt if the last character in the session is not a newline
    if !session_ends_with_newline(session) {
        prompt.insert(0, '\n');
    }

    let sp = spinoff::Spinner::new(spinoff::spinners::Dots2, "".to_string(), None);
    let result = session.feed_prompt(
        model,
        &prompt,
        // OutputRequest
        &mut Default::default(),
        |_| Ok::<_, Infallible>(llm::InferenceFeedback::Continue),
    );
    sp.clear();

    Ok(result?)
}

fn create_session(
    model: &dyn llm::Model,
    inference_session_config: llm::InferenceSessionConfig,
) -> llm::InferenceSession {
    snapshot::read_or_create_session(model, None, None, inference_session_config).0
}

fn session_ends_with_newline(session: &llm::InferenceSession) -> bool {
    session
        .decoded_tokens()
        .last()
        .map(|t| *t == b'\n')
        .unwrap_or(true)
}

fn readline_loop(mut body: impl FnMut(String) -> eyre::Result<()>) -> eyre::Result<()> {
    let mut rl = rustyline::Editor::<LineContinuationValidator, DefaultHistory>::new()?;
    rl.set_helper(Some(LineContinuationValidator));
    rl.bind_sequence(force_newline_event_seq(), Cmd::Newline);

    loop {
        match rl.readline(">> ") {
            Ok(raw_line) => {
                if let Err(err) = body(raw_line) {
                    log::error!("{err}");
                    break;
                }
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                break;
            }
            Err(err) => {
                log::error!("{err}");
                break;
            }
        }
    }

    Ok(())
}

#[cfg(not(windows))]
fn force_newline_event_seq() -> KeyEvent {
    KeyEvent(KeyCode::Enter, Modifiers::ALT)
}

// On Windows, `SHIFT+ENTER` is the key sequence for forcing a newline. This is
// because `ALT+ENTER` typically maximizes the window.
#[cfg(windows)]
fn force_newline_event_seq() -> KeyEvent {
    KeyEvent(KeyCode::Enter, Modifiers::SHIFT)
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
