//! Tests the model's token manipulation APIs:
//!
//! *   [llm::InferenceSession::feed_prompt()]
//!
//! See [crate::TestCase::Tokens].

use std::convert::Infallible;

use llm::{InferenceFeedback, InferenceSession, Model, OutputRequest};
use serde::Serialize;

use crate::{TestCaseReport, TestCaseReportMeta};

/// Tests that the model performs as expected when feeding tokens
pub(crate) fn can_feed(model: &impl Model, input: &str, expected_output: usize) -> TestCaseReport {
    let mut report = TokensReport::default();
    let mut session = model.start_session(Default::default());
    let mut output = OutputRequest {
        all_logits: Some(vec![]),
        ..Default::default()
    };

    if let Err(err) = feed_prompt(input, &mut session, model, &mut output) {
        return report.failure(&err.to_string());
    };

    let top_token;
    match output.all_logits {
        Some(logits) => {
            let start = logits.len() - model.tokenizer().len();
            let mut iter = logits[start..].iter().enumerate();
            let Some((mut max_idx, mut max)) = iter.next() else {
                return report.failure("Could not find any logits for last token.");
            };
            for (idx, score) in iter {
                if score > max {
                    max = score;
                    max_idx = idx;
                }
            }
            top_token = max_idx;
        }
        None => return report.failure("Model did not output any logits."),
    }

    report.output = top_token;

    if top_token != expected_output {
        let tokenizer = model.tokenizer();
        let top_token_str = String::from_utf8_lossy(&tokenizer.token(top_token)).to_string();
        let expected_str = String::from_utf8_lossy(&tokenizer.token(expected_output)).to_string();
        return report.failure(&format!(
            "Expected top token to be {expected_output} ({expected_str}), \
            but was {top_token} ({top_token_str})"
        ));
    }

    log::info!("`can_feed` test passed!");
    report.success()
}

fn feed_prompt(
    prompt: &str,
    session: &mut InferenceSession,
    model: &impl Model,
    output: &mut OutputRequest,
) -> Result<(), llm::InferenceError> {
    session.feed_prompt(model, prompt, output, always_continue)
}

fn always_continue(_: &[u8]) -> Result<InferenceFeedback, Infallible> {
    Ok(InferenceFeedback::Continue)
}

#[derive(Serialize, Default)]
pub struct TokensReport {
    output: usize,
}

impl TokensReport {
    fn failure(self, msg: &str) -> TestCaseReport {
        TestCaseReport {
            meta: TestCaseReportMeta::Error {
                error: msg.to_owned(),
            },
            report: crate::TestCaseReportInner::Tokens(self),
        }
    }

    fn success(self) -> TestCaseReport {
        TestCaseReport {
            meta: TestCaseReportMeta::Success,
            report: crate::TestCaseReportInner::Tokens(self),
        }
    }
}
