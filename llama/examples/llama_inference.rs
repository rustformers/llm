use std::{convert::Infallible, env::args, io::Write};

use ggml::loader::{load_progress, LoadError};
use llm_base::snapshot;

extern crate llama;

fn main() -> Result<(), LoadError> {
    let args: Vec<String> = args().collect();
    let bloom = llama::Llama::load(&args[1], 32, load_progress)?;
    let (mut session, _) = snapshot::read_or_create_session(
        &bloom,
        Default::default(),
        Default::default(),
        Default::default(),
    );

    let _ = session.inference_with_prompt::<Infallible>(
        &bloom,
        &Default::default(),
        "The best kind of wine is ",
        Some(32),
        &mut rand::thread_rng(),
        |t| {
            print!("{t}");
            std::io::stdout().flush().unwrap();

            Ok(())
        },
    );

    println!();
    Ok(())
}
