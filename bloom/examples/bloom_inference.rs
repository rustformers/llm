use std::{convert::Infallible, env::args, io::Write};

use llm_base::{load_progress_callback, model::KnownModel};

extern crate bloom;

fn main() {
    let args: Vec<String> = args().collect();
    let loc = &args[1];
    let prompt = match &args.len() {
        3 => &args[2],
        _ => "Rust is a cool programming language because ",
    };

    println!(" >>> Loading model from {loc}...");
    let now = std::time::Instant::now();

    let bloom = bloom::Bloom::load(loc, true, 512, load_progress_callback)
        .unwrap_or_else(|e| panic!("Error loading model from {loc}: {e}"));

    println!(" >>> Model loaded in {} ms.", now.elapsed().as_millis());

    let mut session = bloom.start_session(Default::default());
    let res = session.inference_with_prompt::<Infallible>(
        &bloom,
        &Default::default(),
        prompt,
        None,
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
