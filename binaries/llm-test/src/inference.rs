//! Tests the model's inference APIs.
//!
//! See [crate::TestCase::Inference].

use std::{convert::Infallible, sync::Arc};

use llm::{InferenceSessionConfig, InferenceStats};

use crate::{ModelConfig, TestCaseReport, TestCaseReportInner, TestCaseReportMeta};

pub(crate) fn can_infer(
    model: &dyn llm::Model,
    model_config: &ModelConfig,
    input: &str,
    expected_output: Option<&str>,
    maximum_token_count: usize,
) -> anyhow::Result<TestCaseReport> {
    let mut session = model.start_session(InferenceSessionConfig {
        n_threads: model_config.threads,
        ..Default::default()
    });
    let (actual_output, res) = run_inference(model, &mut session, input, maximum_token_count);

    // Process the results
    Ok(TestCaseReport {
        meta: match &res {
            Ok(_) => match expected_output {
                Some(expected_output) => {
                    if expected_output == actual_output {
                        log::info!("`can_infer` test passed!");
                        TestCaseReportMeta::Success
                    } else {
                        TestCaseReportMeta::Error {
                            error: "The output did not match the expected output.".to_string(),
                        }
                    }
                }
                None => {
                    log::info!("`can_infer` test passed (no expected output)!");
                    TestCaseReportMeta::Success
                }
            },
            Err(err) => TestCaseReportMeta::Error {
                error: err.to_string(),
            },
        },
        report: TestCaseReportInner::Inference {
            input: input.into(),
            expect_output: expected_output.map(|s| s.to_string()),
            actual_output,
            inference_stats: res.ok(),
        },
    })
}

fn run_inference(
    model: &dyn llm::Model,
    session: &mut llm::InferenceSession,
    input: &str,
    maximum_token_count: usize,
) -> (String, Result<InferenceStats, llm::InferenceError>) {
    let mut actual_output: String = String::new();
    let res = session.infer::<Infallible>(
        model,
        &mut rand::rngs::mock::StepRng::new(0, 1),
        &llm::InferenceRequest {
            prompt: input.into(),
            parameters: &llm::InferenceParameters {
                sampler: Arc::new(DeterministicSampler),
            },
            play_back_previous_tokens: false,
            maximum_token_count: Some(maximum_token_count),
        },
        &mut Default::default(),
        |r| match r {
            llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                actual_output += &t;
                Ok(llm::InferenceFeedback::Continue)
            }
            _ => Ok(llm::InferenceFeedback::Continue),
        },
    );

    (actual_output, res)
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
