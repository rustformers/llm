use anyhow::Context;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use llm::InferenceStats;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    cmp::min,
    collections::HashMap,
    convert::Infallible,
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

#[derive(Parser)]
struct Cli {
    /// The path to the directory containing the model configurations.
    /// If not specified, the default directory will be used.
    #[clap(short, long)]
    configs: Option<PathBuf>,

    /// Whether to use memory mapping when loading the model.
    #[clap(short, long)]
    no_mmap: bool,

    /// The model architecture to test. If not specified, all architectures will be tested.
    architecture: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Set up the logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse command line arguments
    let args = Cli::parse();
    let specific_model = args.architecture.clone();

    // Initialize directories
    let cwd = env::current_dir()?;
    let configs_dir = args
        .configs
        .unwrap_or_else(|| cwd.join("binaries/llm-test/configs"));
    let download_dir = cwd.join(".tests/models");
    fs::create_dir_all(&download_dir)?;
    let results_dir = cwd.join(".tests/results");
    fs::create_dir_all(&results_dir)?;

    // Load configurations
    let test_configs: HashMap<String, TestConfig> = fs::read_dir(configs_dir)?
        .filter_map(Result::ok)
        .map(|de| de.path())
        .filter(|p| p.is_file())
        .filter(|p| p.extension().unwrap_or_default() == "json")
        .map(|path| {
            let file_name = path.file_stem().unwrap().to_string_lossy().to_string();
            let test_config: TestConfig = serde_json::from_str(&fs::read_to_string(&path)?)?;
            anyhow::Ok((file_name, test_config))
        })
        .collect::<Result<_, _>>()?;
    let model_config = ModelConfig {
        mmap: !args.no_mmap,
    };

    // Test models
    let mut test_configs = if let Some(specific_architecture) = specific_model {
        vec![test_configs
            .get(&specific_architecture)
            .with_context(|| {
                format!(
                    "No config found for `{specific_architecture}`. Available configs: {:?}",
                    test_configs.keys()
                )
            })?
            .clone()]
    } else {
        test_configs.values().cloned().collect()
    };
    test_configs.sort_by_key(|tc| tc.architecture.clone());

    let test_configs_len = test_configs.len();
    for test_config in test_configs {
        test_model(&model_config, &test_config, &download_dir, &results_dir).await?;
        if test_configs_len > 1 {
            log::info!("----");
        }
    }

    log::info!("All tests passed!");
    Ok(())
}

struct ModelConfig {
    mmap: bool,
}

#[derive(Deserialize, Debug, Clone)]
struct TestConfig {
    url: String,
    filename: PathBuf,
    architecture: String,
}

#[derive(Serialize)]
pub struct Report {
    pub could_load: bool,
    pub inference_stats: Option<InferenceStats>,
    pub error: Option<String>,
    pub output: String,
}

async fn test_model(
    config: &ModelConfig,
    test_config: &TestConfig,
    download_dir: &Path,
    results_dir: &Path,
) -> anyhow::Result<()> {
    let local_path = if test_config.filename.is_file() {
        // If this filename points towards a valid file, use it
        test_config.filename.clone()
    } else {
        // Otherwise, use the download dir
        download_dir.join(&test_config.filename)
    };

    log::info!(
        "Testing architecture: `{}` ({})",
        test_config.architecture,
        local_path.display()
    );

    // Download the model if necessary
    download_file(&test_config.url, &local_path).await?;

    let start_time = Instant::now();

    // Load the model
    let architecture = llm::ModelArchitecture::from_str(&test_config.architecture)?;
    let model = {
        let model = llm::load_dynamic(
            Some(architecture),
            &local_path,
            llm::TokenizerSource::Embedded,
            llm::ModelParameters {
                prefer_mmap: config.mmap,
                ..Default::default()
            },
            |progress| {
                let print = !matches!(&progress,
                    llm::LoadProgress::TensorLoaded { current_tensor, tensor_count }
                    if current_tensor % (tensor_count / 10) != 0
                );

                if print {
                    log::info!("loading: {:?}", progress);
                }
            },
        );

        match model {
            Ok(m) => m,
            Err(err) => {
                // Create a report with could_load set to false
                let report = Report {
                    could_load: false,
                    inference_stats: None,
                    error: Some(format!("Failed to load model: {}", err)),
                    output: String::new(),
                };

                // Serialize the report to a JSON string
                let json_report = serde_json::to_string(&report)?;
                let report_path = results_dir.join(format!("{}.json", test_config.architecture));

                // Write the JSON report to a file
                fs::write(report_path, json_report)?;

                // Optionally, you can return early or decide how to proceed
                return Err(err.into());
            }
        }
    };

    log::info!(
        "Model fully loaded! Elapsed: {}ms",
        start_time.elapsed().as_millis()
    );

    // Run the model
    let mut session = model.start_session(Default::default());

    let prompt = "When a llama rides a crab, ";
    let mut output: String = String::new();

    log::info!("Running inference...");
    let res = session.infer::<Infallible>(
        model.as_ref(),
        &mut rand::rngs::mock::StepRng::new(0, 1),
        &llm::InferenceRequest {
            prompt: prompt.into(),
            parameters: &llm::InferenceParameters {
                n_threads: 2,
                n_batch: 1,
                sampler: Arc::new(DeterministicSampler),
            },
            play_back_previous_tokens: false,
            maximum_token_count: Some(128),
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
    log::info!("Inference done!");

    // Process the results
    let (inference_results, error) = match res {
        Ok(result) => (Some(result), None),
        Err(err) => (None, Some(err)),
    };

    // Save the results
    let report = Report {
        could_load: true,
        inference_stats: inference_results,
        error: error.map(|e| format!("{:?}", e)),
        output,
    };

    // Serialize the report to a JSON string
    let json_report = serde_json::to_string(&report)?;
    let report_path = results_dir.join(format!("{}.json", test_config.architecture));

    // Write the JSON report to a file
    fs::write(report_path, json_report)?;

    // Optionally, panic if there was an error
    if let Some(err) = &report.error {
        panic!("Error: {}", err);
    }

    log::info!(
        "Successfully tested architecture `{}`!",
        test_config.architecture
    );

    Ok(())
}

#[derive(Debug)]
struct DeterministicSampler;
impl llm::Sampler for DeterministicSampler {
    fn sample(
        &self,
        previous_tokens: &[llm::TokenId],
        logits: &[f32],
        _rng: &mut dyn rand::RngCore,
    ) -> llm::TokenId {
        // Takes the most likely element from the logits, except if they've appeared in `previous_tokens`
        // at all
        let mut logits = logits.to_vec();
        for &token in previous_tokens {
            logits[token as usize] = f32::NEG_INFINITY;
        }

        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as llm::TokenId
    }
}

async fn download_file(url: &str, local_path: &Path) -> anyhow::Result<()> {
    if local_path.exists() {
        return Ok(());
    }

    let client = Client::new();

    let mut res = client.get(url).send().await?;
    let total_size = res
        .content_length()
        .context("Failed to get content length")?;

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
