use rand::thread_rng;
use rustyline::error::ReadlineError;
use std::{convert::Infallible, io::Write};
use llama_rs::{InferenceError, InferenceParameters};
use crate::cli_args::CLI_ARGS;

pub fn repl_mode(
    prompt: &str,
    model: &llama_rs::Model,
    vocab: &llama_rs::Vocabulary,
    params: &InferenceParameters,
) {
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(line) => {
                let mut session = model.start_session(CLI_ARGS.repeat_last_n);
                let prompt = prompt.replace("$PROMPT", &line);
                let mut rng = thread_rng();

                let mut sp = spinners::Spinner::new(spinners::Spinners::Dots2, "".to_string());
                if let Err(InferenceError::ContextFull) =
                    session.feed_prompt::<Infallible>(model, vocab, params, &prompt, |_| Ok(()))
                {
                    log::error!("Prompt exceeds context window length.")
                };
                sp.stop();

                let res = session.inference_with_prompt::<Infallible>(
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
                );
                println!();

                if let Err(InferenceError::ContextFull) = res {
                    log::error!("Reply exceeds context window length");
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
}
