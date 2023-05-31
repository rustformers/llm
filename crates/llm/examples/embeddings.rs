use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
struct Args {
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    #[arg(long, short = 'v')]
    pub vocabulary_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    pub vocabulary_repository: Option<String>,
    #[arg(long, short = 'q')]
    pub query: Option<String>,
    #[arg(long, short = 'c')]
    pub comparands: Vec<String>,
}
impl Args {
    pub fn to_vocabulary_source(&self) -> llm::VocabularySource {
        match (&self.vocabulary_path, &self.vocabulary_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --vocabulary-path and --vocabulary-repository");
            }
            (Some(path), None) => llm::VocabularySource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::VocabularySource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::VocabularySource::Model,
        }
    }
}

fn main() {
    let args = Args::parse();

    let vocabulary_source = args.to_vocabulary_source();
    let model_architecture = args.model_architecture;
    let model_path = args.model_path;
    let query = args
        .query
        .as_deref()
        .unwrap_or("My favourite animal is the dog");
    let comparands = if !args.comparands.is_empty() {
        args.comparands
    } else {
        vec![
            "My favourite animal is the dog".to_string(),
            "I have just adopted a cute dog".to_string(),
            "My favourite animal is the cat".to_string(),
        ]
    };

    // Load model
    let model_params = llm::ModelParameters {
        prefer_mmap: true,
        context_size: 4096,
        lora_adapters: None,
    };
    let model = llm::load_dynamic(
        model_architecture,
        &model_path,
        vocabulary_source,
        model_params,
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });
    let inference_parameters = llm::InferenceParameters {
        n_threads: 8,
        n_batch: 8,
        sampler: &llm::samplers::TopPTopK::default(),
    };

    // Generate embeddings for query and comparands
    let query_embeddings = get_embeddings(model.as_ref(), &inference_parameters, query);
    let comparand_embeddings: Vec<(String, Vec<f32>)> = comparands
        .iter()
        .map(|text| {
            (
                text.clone(),
                get_embeddings(model.as_ref(), &inference_parameters, text),
            )
        })
        .collect();

    // Print embeddings
    fn print_embeddings(text: &str, embeddings: &[f32]) {
        println!("{text}");
        println!("  Embeddings length: {}", embeddings.len());
        println!("  Embeddings first 10: {:.02?}", embeddings.get(0..10));
    }

    print_embeddings(query, &query_embeddings);
    println!("---");
    for (text, embeddings) in &comparand_embeddings {
        print_embeddings(text, embeddings);
    }

    // Calculate the cosine similarity between the query and each comparand, and sort by similarity
    let mut similarities: Vec<(&str, f32)> = comparand_embeddings
        .iter()
        .map(|(text, embeddings)| {
            (
                text.as_str(),
                cosine_similarity(&query_embeddings, &embeddings),
            )
        })
        .collect();
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Print similarities
    println!("---");
    println!("Similarities:");
    for (text, score) in similarities {
        println!("  {text}: {score}");
    }
}

fn get_embeddings(
    model: &dyn llm::Model,
    inference_parameters: &llm::InferenceParameters,
    query: &str,
) -> Vec<f32> {
    let mut session = model.start_session(Default::default());
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    let vocab = model.vocabulary();
    let beginning_of_sentence = true;
    let query_token_ids = vocab
        .tokenize(&format!(" {}", query), beginning_of_sentence)
        .unwrap()
        .iter()
        .map(|(_, tok)| *tok)
        .collect::<Vec<_>>();
    let _ = model.evaluate(
        &mut session,
        inference_parameters,
        &query_token_ids,
        &mut output_request,
    );
    output_request.embeddings.unwrap()
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
