use llm::ModelArchitecture;
use llm_base::{InferenceFeedback, InferenceParameters, ModelParameters};
use std::{convert::Infallible, path::PathBuf};

fn main() {
    let prompt = "What is the meaning of life?";
    let model_path = PathBuf::from(r"C:\Users\lkreu\Downloads\orca-mini-v2_7b.ggmlv3.q5_K_M.bin");
    let now = std::time::Instant::now();

    let model = llm::load_dynamic(
        Some(ModelArchitecture::Llama),
        &model_path,
        llm_base::TokenizerSource::Embedded,
        ModelParameters {
            use_gpu: true,
            ..Default::default()
        },
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| panic!("Failed to load llama model from {model_path:?}: {err}"));

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    for i in 0..10 {
        println!("Starting session {i}");
        let mut session = model.start_session(Default::default());
        session
            .feed_prompt(model.as_ref(), prompt, &mut Default::default(), |_| {
                Ok::<InferenceFeedback, Infallible>(llm::InferenceFeedback::Continue)
            })
            .unwrap();
        drop(session);
        println!("Dropped session {i}");
    }

    drop(model);

    println!("Model dropped! Elapsed: {}ms", now.elapsed().as_millis());

    for _ in 0..5 {
        let model = llm::load_dynamic(
            Some(ModelArchitecture::Llama),
            &model_path,
            llm_base::TokenizerSource::Embedded,
            ModelParameters {
                use_gpu: true,
                ..Default::default()
            },
            llm::load_progress_callback_stdout,
        )
        .unwrap_or_else(|err| panic!("Failed to load llama model from {model_path:?}: {err}"));
        drop(model);
    }
}
