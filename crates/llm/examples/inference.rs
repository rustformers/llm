use llm::{load_progress_callback_stdout, models, LoadError, Model};
use std::{convert::Infallible, env::args, io::Write, path::Path};

fn main() {
    let raw_args: Vec<String> = args().collect();
    let args = match &raw_args.len() {
      3 => (raw_args[1].as_str(), raw_args[2].as_str(), "Rust is a cool programming language because"),
      4 => (raw_args[1].as_str(), raw_args[2].as_str(), raw_args[3].as_str()),
      _ => panic!("Usage: cargo run --release --example inference <model type> <path to model> <optional prompt>")
    };

    let model_type = args.0;
    let model_path = args.1;
    let prompt = args.2;

    let model_load: Result<Box<dyn Model>, LoadError> = match model_type {
        "bloom" => load::<models::Bloom>(model_path),
        "gpt2" => load::<models::Gpt2>(model_path),
        "gptj" => load::<models::GptJ>(model_path),
        "llama" => load::<models::Llama>(model_path),
        "neox" => load::<models::NeoX>(model_path),
        model => panic!("{model} is not a supported model"),
    };

    let model = match model_load {
        Ok(model) => model,
        Err(e) => panic!("Failed to load {model_type} model from {model_path}: {e}"),
    };
    let mut session = model.start_session(Default::default());

    let res = session.infer::<Infallible>(model.as_ref(), prompt, &mut rand::thread_rng(), |t| {
        print!("{t}");
        std::io::stdout().flush().unwrap();

        Ok(())
    });

    match res {
        Ok(result) => println!("\n\nInference stats:\n{result}"),
        Err(err) => println!("\n{err}"),
    }
}

pub fn load<M: llm::KnownModel + 'static>(model_path: &str) -> Result<Box<dyn Model>, LoadError> {
    let now = std::time::Instant::now();

    let model = llm::load::<M>(
        Path::new(model_path),
        true,
        Default::default(),
        load_progress_callback_stdout,
    )?;

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    Ok(Box::new(model))
}
