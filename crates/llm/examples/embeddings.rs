use std::{path::Path, time::Instant};
use llm::{
    ModelArchitecture, LoadProgress, ModelParameters, OutputRequest, Model,
};
use spinoff::{spinners::Dots2, Spinner};


fn main() {
    let model_path = Path::new(r"./models/ggml-vicuna-7B-1.1-q5_0.bin");
    let model_architecture: ModelArchitecture = ModelArchitecture::Llama;
    let overrides = serde_json::from_str("{}").unwrap();
    let sp = Some(Spinner::new(Dots2, "Loading model...", None));

    let now = Instant::now();
    let prev_load_time = now;

    let model_params = ModelParameters {
        prefer_mmap: true,
        context_size: 4096,
        inference_parameters: Default::default(),
        lora_adapters: None,
    };
    let model = llm::load_dynamic(
        model_architecture,
        model_path,
        model_params,
        overrides,
        load_progress_callback(sp, now, prev_load_time),
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });

    let query = "My favourite animal is the dog";
    //let query = "What is your favourite animal";


    let sentences = vec![
        //"My favourite animal is the dog",
        "I have just adopted a cute dog",
        "My favourite animal is the cat",
    ];
    fn get_embeddings(model: &Box<dyn Model>, query: &str) -> Option<Vec<f32>> {
        let mut session = model.start_session(Default::default());
        let mut output_request = OutputRequest {
            all_logits: None,
            embeddings: Some(Vec::new()),
        };
        let vocab = model.vocabulary();
        let beginning_of_sentence = true;
        let query_token_ids = vocab
            .tokenize(&format!(" {}", query), beginning_of_sentence).unwrap()
            .iter().map(|(_, tok)| *tok).collect::<Vec<_>>();
        let _ = model.evaluate(
            &mut session,
            //&inference_params,
            &Default::default(),
            &query_token_ids,
            &mut output_request,
        );
        output_request.embeddings
    }
    
    let query_embeddings = get_embeddings(&model, query).unwrap();
    println!("Text: {:?} Embeddings length: {:?}\nThe first 10 elements of embeddings: \n{:?}\n", query, query_embeddings.len(), query_embeddings.get(0..10));
    
    let mut sentences_similarity = sentences
        .iter()
        .enumerate()
        .map(|(_idx, &s)| {
            let embeddings = get_embeddings(&model, s).unwrap();
            println!("Text: {:?} Embeddings length: {:?}\nThe first 10 elements of embeddings: \n{:?}\n", s, embeddings.len(), embeddings.get(0..10));
            let value = cosine_similarity(&query_embeddings, &embeddings);
            TextSimilrity {
                text: s.to_owned(),
                similarity: value,
            }
        })
        .collect::<Vec<_>>();
    sentences_similarity.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap()
    });
    sentences_similarity.iter().for_each(|i| println!("Cosine Similrity: {:#?}\n", i));
}

#[derive(Debug)]
pub struct TextSimilrity {
    pub text: String,
    pub similarity: f32,
}



fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product = dot(&v1, &v2);
    let magnitude1 = magnitude(&v1);
    let magnitude2 = magnitude(&v2);

    dot_product / (magnitude1 * magnitude2)
}

fn dot(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn load_progress_callback(
    mut sp: Option<Spinner>,
    now: Instant,
    mut prev_load_time: Instant,
) -> impl FnMut(LoadProgress) {
    move |progress| match progress {
        LoadProgress::HyperparametersLoaded => {
            if let Some(sp) = sp.as_mut() {
                sp.update_text("Loaded hyperparameters")
            };
        }
        LoadProgress::ContextSize { bytes } => log::debug!(
            "ggml ctx size = {}",
            bytesize::to_string(bytes as u64, false)
        ),
        LoadProgress::TensorLoaded {
            current_tensor,
            tensor_count,
            ..
        } => {
            if prev_load_time.elapsed().as_millis() > 500 {
                // We don't want to re-render this on every message, as that causes the
                // spinner to constantly reset and not look like it's spinning (and
                // it's obviously wasteful).
                if let Some(sp) = sp.as_mut() {
                    sp.update_text(format!(
                        "Loaded tensor {}/{}",
                        current_tensor + 1,
                        tensor_count
                    ));
                };
                prev_load_time = std::time::Instant::now();
            }
        }
        LoadProgress::LoraApplied { name, source } => {
            if let Some(sp) = sp.as_mut() {
                sp.update_text(format!("Applied LoRA: {} Source: {}", name, source.to_string_lossy()));
            };
        }
        LoadProgress::Loaded {
            file_size,
            tensor_count,
        } => {
            if let Some(sp) = sp.take() {
                sp.success(&format!(
                    "Loaded {tensor_count} tensors ({}) after {}ms",
                    bytesize::to_string(file_size, false),
                    now.elapsed().as_millis()
                ));
            };
        }
    }
}
