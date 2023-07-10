//! Test runner for all LLMs.

mod common;
mod delete;
mod inference;
mod tokens;

use anyhow::Context;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use llm::InferenceStats;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    cmp::min,
    collections::HashMap,
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
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

    /// The thread count to use when running inference.
    #[clap(short, long)]
    threads: Option<usize>,

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
        threads: args.threads.unwrap_or(2),
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
    threads: usize,
}

#[derive(Deserialize, Debug, Clone)]
struct TestConfig {
    url: String,
    filename: PathBuf,
    architecture: String,
    test_cases: Vec<TestCase>,
}

#[derive(Deserialize, Debug, Clone)]
enum TestCase {
    Inference {
        input: String,
        output: Option<String>,
        maximum_token_count: usize,
    },
    Tokens {
        input: String,
        output: usize,
    },
    Delete {},
}

#[derive(Serialize)]
enum Report {
    LoadFail { error: String },
    LoadSuccess { test_cases: Vec<TestCaseReport> },
}

#[derive(Serialize)]
struct TestCaseReport {
    meta: TestCaseReportMeta,
    report: TestCaseReportInner,
}

#[derive(Serialize)]
#[serde(untagged)]
enum TestCaseReportMeta {
    Error { error: String },
    Success,
}

#[derive(Serialize)]
pub enum TestCaseReportInner {
    Inference {
        input: String,
        expect_output: Option<String>,
        actual_output: String,
        inference_stats: Option<InferenceStats>,
    },
    Tokens(tokens::TokensReport),
    Delete(delete::DeleteReport),
}

async fn test_model(
    model_config: &ModelConfig,
    test_config: &TestConfig,
    download_dir: &Path,
    results_dir: &Path,
) -> anyhow::Result<()> {
    // Load the model
    let architecture = llm::ModelArchitecture::from_str(&test_config.architecture)?;

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

    struct TestVisitor<'a> {
        model_config: &'a ModelConfig,
        test_config: &'a TestConfig,
        results_dir: &'a Path,
        local_path: &'a Path,
    }
    impl<'a> llm::ModelArchitectureVisitor<anyhow::Result<()>> for TestVisitor<'a> {
        fn visit<M: llm::KnownModel + 'static>(&mut self) -> anyhow::Result<()> {
            let Self {
                model_config,
                test_config,
                results_dir,
                local_path,
            } = *self;

            let start_time = Instant::now();

            let model = {
                let model = llm::load::<M>(
                    local_path,
                    llm::TokenizerSource::Embedded,
                    llm::ModelParameters {
                        prefer_mmap: model_config.mmap,
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
                        write_report(
                            test_config,
                            results_dir,
                            &Report::LoadFail {
                                error: format!("Failed to load model: {}", err),
                            },
                        )?;

                        return Err(err.into());
                    }
                }
            };

            log::info!(
                "Model fully loaded! Elapsed: {}ms",
                start_time.elapsed().as_millis()
            );

            //
            // Non-model-specific tests
            //

            // Confirm that the model can be sent to a thread, then sent back
            let model = common::can_send(model)?;

            // Confirm that the hyperparameters can be roundtripped
            common::can_roundtrip_hyperparameters(&model)?;

            //

            //
            // Model-specific tests
            //

            // Run the test cases
            let mut test_case_reports = vec![];
            for test_case in &test_config.test_cases {
                match test_case {
                    TestCase::Inference {
                        input,
                        output,
                        maximum_token_count,
                    } => test_case_reports.push(inference::can_infer(
                        &model,
                        model_config,
                        input,
                        output.as_deref(),
                        *maximum_token_count,
                    )?),
                    TestCase::Tokens { input, output } => {
                        test_case_reports.push(tokens::can_feed(&model, input, *output));
                    }
                    TestCase::Delete {} => {
                        test_case_reports.push(delete::can_delete(&model));
                    }
                }
            }
            let first_error: Option<String> =
                test_case_reports
                    .iter()
                    .find_map(|report: &TestCaseReport| match &report.meta {
                        TestCaseReportMeta::Error { error } => Some(error.clone()),
                        _ => None,
                    });

            // Save the results
            // Serialize the report to a JSON string
            write_report(
                test_config,
                results_dir,
                &Report::LoadSuccess {
                    test_cases: test_case_reports,
                },
            )?;

            // Optionally, panic if there was an error
            if let Some(err) = first_error {
                panic!("Error: {}", err);
            }

            log::info!(
                "Successfully tested architecture `{}`!",
                test_config.architecture
            );

            Ok(())
        }
    }
    architecture.visit(&mut TestVisitor {
        model_config,
        test_config,
        results_dir,
        local_path: &local_path,
    })?;

    Ok(())
}

fn write_report(
    test_config: &TestConfig,
    results_dir: &Path,
    report: &Report,
) -> anyhow::Result<()> {
    let json_report = serde_json::to_string_pretty(&report)?;
    let report_path = results_dir.join(format!("{}.json", test_config.architecture));
    fs::write(report_path, json_report)?;
    Ok(())
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
