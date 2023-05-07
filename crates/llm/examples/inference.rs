use llm::load_progress_callback_stdout as load_callback;
use std::{convert::Infallible, env::args, io::Write, path::Path};

fn main() {
    let raw_args: Vec<String> = args().collect();
    let args = match &raw_args.len() {
      3 => (raw_args[1].as_str(), raw_args[2].as_str(), "Rust is a cool programming language because"),
      4 => (raw_args[1].as_str(), raw_args[2].as_str(), raw_args[3].as_str()),
      _ => panic!("Usage: cargo run --release --example inference <model type> <path to model> <optional prompt>")
    };

    let model_type = args.0;
    let model_path = Path::new(args.1);
    let prompt = args.2;

    let now = std::time::Instant::now();

    let architecture = model_type.parse().unwrap_or_else(|e| panic!("{e}"));

    let model = llm::load_dynamic(architecture, model_path, Default::default(), load_callback)
        .unwrap_or_else(|err| {
            panic!("Failed to load {model_type} model from {model_path:?}: {err}")
        });

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    let mut session = model.start_session(Default::default());

    let res = session.infer::<Infallible>(
        model.as_ref(),
        prompt,
        // EvaluateOutputRequest
        &mut Default::default(),
        &mut rand::thread_rng(),
        |t| {
            print!("{t}");
            std::io::stdout().flush().unwrap();

            Ok(())
        },
    );

    match res {
        Ok(result) => println!("\n\nInference stats:\n{result}"),
        Err(err) => println!("\n{err}"),
    }
}
