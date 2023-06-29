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
    filename: String,
    architecture: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //This shoud be done with clap but I'm lazy
    let args: Vec<String> = env::args().collect();

    let mut specific_model = None;
    if args.len() > 1 {
        println!("Testing architecture: {}", args[1].to_lowercase());
        specific_model = Some(args[1].to_lowercase());
    } else {
        println!("Testing all architectures.");
    }

    let mut configs = HashMap::new();
    let cwd = std::env::current_dir()?;
    let configs_dir = cwd.join("binaries/llm-test/configs");
    let download_dir = cwd.join(".tests/models");
    fs::create_dir_all(&download_dir)?;
    let results_dir = cwd.join(".tests/results");
    fs::create_dir_all(&results_dir)?;

    for entry in fs::read_dir(configs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "json" {
                    let file_name = path.file_stem().unwrap().to_str().unwrap().to_string();
                    let config: TestCase = serde_json::from_str(&fs::read_to_string(path)?)?;
                    configs.insert(file_name, config);
                }
            }
        }
    }

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

    let local_path = download_dir.join(&config.filename);

    //download the model
    download_file(&config.url, &local_path).await?;

    let now = std::time::Instant::now();

    let architecture = llm::ModelArchitecture::from_str(&config.architecture)?;
    //load the model
    let model = llm::load_dynamic(
        architecture,
        &local_path,
        llm::TokenizerSource::Embedded,
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| panic!("Failed to load {architecture} model from {local_path:?}: {err}"));

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    //run the model
    let mut session = model.start_session(Default::default());

    let prompt = "write a story about a lama riding a crab:";
    let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    let mut output: String = String::new();

    println!("Running inference...");
    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rng,
        &llm::InferenceRequest {
            prompt: prompt.into(),
            parameters: &llm::InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: Some(50),
        },
        // OutputRequest
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

    let inference_results: Option<llm::InferenceStats>;
    let error: Option<llm::InferenceError>;

    match res {
        Ok(result) => {
            inference_results = Some(result);
            error = None;
        }
        Err(err) => {
            inference_results = None;
            error = Some(err);
        }
    }

    //save the results
    let report = Report {
        could_loaded: true,
        inference_stats: inference_results,
        error: error.map(|e| format!("{:?}", e)),
        output,
    };

    // Serialize the report to a JSON string
    let json_report = serde_json::to_string(&report).unwrap();
    let report_path = results_dir.join(format!("{}.json", config.architecture));
    match fs::write(report_path, json_report) {
        Ok(_) => println!("Report successfully written to file."),
        Err(e) => println!("Failed to write report to file: {}", e),
    }

    if let Some(err) = &report.error {
        panic!("Error: {}", err);
    }

    println!(
        "Successfully tested architecture `{}`!",
        config.architecture
    );

    Ok(())
}
