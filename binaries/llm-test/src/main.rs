extern crate indicatif;
extern crate reqwest;
extern crate tokio;

use indicatif::{ProgressBar, ProgressStyle};
use llm::InferenceStats;
use rand::rngs::StdRng;
use rand::SeedableRng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::convert::Infallible;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

async fn download_file(url: &str, local_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(local_path).exists() {
        println!("Model already exists at {}", local_path.to_str().unwrap());
        return Ok(());
    }

    let client = Client::new();

    let mut res = client.get(url).send().await?;
    let total_size = res.content_length().ok_or("Failed to get content length")?;

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
        .progress_chars("#>-"));

    let mut file = File::create(local_path)?;
    let mut downloaded: u64 = 0;

    while let Some(chunk) = res.chunk().await? {
        file.write_all(&chunk)?;
        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message("Download complete");

    Ok(())
}

#[derive(Deserialize, Debug)]
struct TestCase {
    url: String,
    filename: PathBuf,
    architecture: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let specific_model = if args.len() > 1 {
        println!("Testing architecture: {}", args[1].to_lowercase());
        Some(args[1].to_lowercase())
    } else {
        println!("Testing all architectures.");
        None
    };

    // Initialize directories
    let cwd = env::current_dir()?;
    let configs_dir = cwd.join("binaries/llm-test/configs");
    let download_dir = cwd.join(".tests/models");
    fs::create_dir_all(&download_dir)?;
    let results_dir = cwd.join(".tests/results");
    fs::create_dir_all(&results_dir)?;

    // Load configurations
    let mut configs = HashMap::new();
    for entry in fs::read_dir(configs_dir)? {
        let path = entry?.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
            let file_name = path.file_stem().unwrap().to_str().unwrap().to_string();
            let config: TestCase = serde_json::from_str(&fs::read_to_string(&path)?)?;
            configs.insert(file_name, config);
        }
    }

    // Test models
    if let Some(specific_architecture) = specific_model {
        if let Some(config) = configs.get(&specific_architecture) {
            println!("Key: {}, Config: {:?}", specific_architecture, config);
            test_model(config, &download_dir, &results_dir).await?;
        } else {
            println!("No config found for {}", specific_architecture);
        }
    } else {
        for (key, config) in &configs {
            println!("Key: {}, Config: {:?}", key, config);
            test_model(config, &download_dir, &results_dir).await?;
        }
    }

    println!("All tests passed!");
    Ok(())
}

#[derive(Serialize)]
pub struct Report {
    pub could_loaded: bool,
    pub inference_stats: Option<InferenceStats>,
    pub error: Option<String>,
    pub output: String,
}

async fn test_model(
    config: &TestCase,
    download_dir: &Path,
    results_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing architecture: `{}` ...", config.architecture);

    let local_path = if config.filename.is_file() {
        // If this filename points towards a valid file, use it
        config.filename.clone()
    } else {
        // Otherwise, use the download dir
        download_dir.join(&config.filename)
    };

    // Download the model
    download_file(&config.url, &local_path).await?;

    let start_time = Instant::now();

    // Load the model
    let architecture = llm::ModelArchitecture::from_str(&config.architecture)?;
    let model_result = llm::load_dynamic(
        Some(architecture),
        &local_path,
        llm::TokenizerSource::Embedded,
        Default::default(),
        llm::load_progress_callback_stdout,
    );

    let model = match model_result {
        Ok(m) => m,
        Err(err) => {
            // Create a report with could_loaded set to false
            let report = Report {
                could_loaded: false,
                inference_stats: None,
                error: Some(format!("Failed to load model: {}", err)),
                output: String::new(),
            };

            // Serialize the report to a JSON string
            let json_report = serde_json::to_string(&report)?;
            let report_path = results_dir.join(format!("{}.json", config.architecture));

            // Write the JSON report to a file
            fs::write(report_path, json_report)?;

            // Optionally, you can return early or decide how to proceed
            return Err(Box::new(err));
        }
    };

    println!(
        "Model fully loaded! Elapsed: {}ms",
        start_time.elapsed().as_millis()
    );

    // Run the model
    let mut session = model.start_session(Default::default());

    let prompt = "write a story about a lama riding a crab:";
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    let mut output = String::new();

    println!("Running inference...");
    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rng,
        &llm::InferenceRequest {
            prompt: prompt.into(),
            parameters: &llm::InferenceParameters {
                n_threads: 2,
                n_batch: 1,
                ..Default::default()
            },
            play_back_previous_tokens: false,
            maximum_token_count: Some(10),
        },
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                output += &t;
                Ok(llm::InferenceFeedback::Continue)
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    );
    println!("Inference done!");

    // Process the results
    let (inference_results, error) = match res {
        Ok(result) => (Some(result), None),
        Err(err) => (None, Some(err)),
    };

    // Save the results
    let report = Report {
        could_loaded: true,
        inference_stats: inference_results,
        error: error.map(|e| format!("{:?}", e)),
        output,
    };

    // Serialize the report to a JSON string
    let json_report = serde_json::to_string(&report)?;
    let report_path = results_dir.join(format!("{}.json", config.architecture));

    // Write the JSON report to a file
    fs::write(report_path, json_report)?;

    // Optionally, panic if there was an error
    if let Some(err) = &report.error {
        panic!("Error: {}", err);
    }

    println!(
        "Successfully tested architecture `{}`!",
        config.architecture
    );

    Ok(())
}
