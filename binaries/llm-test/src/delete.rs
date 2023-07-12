//! Tests the model's token manipulation APIs:
//!
//! *   [llm::InferenceSession::feed_prompt()]
//!
//! See [crate::TestCase::Tokens].

use std::convert::Infallible;

use llm::{InferenceFeedback, InferenceSession, Model, OutputRequest};
use serde::Serialize;

use crate::{TestCaseReport, TestCaseReportMeta};

/// Tests that models can delete tokens without changing the model's behavior.
pub(crate) fn can_delete(model: &impl Model) -> TestCaseReport {
    let report = DeleteReport::default();
    let mut session = model.start_session(Default::default());
    let mut output = OutputRequest {
        all_logits: Some(vec![]),
        ..Default::default()
    };

    // Feed some tokens
    if let Err(err) = feed_prompt("The llama lived on the", &mut session, model, &mut output) {
        return report.failure(&err.to_string());
    }

    // Add token and get the logits
    if let Err(err) = feed_prompt(" ", &mut session, model, &mut output) {
        return report.failure(&err.to_string());
    }
    let Some(original_logits) = output.all_logits.clone() else {
        return report.failure("Model did not return logits.");
    };

    // Rewind, then re-add. Verify logits are the same.
    if let Err(err) = session.rewind(model, 1) {
        return report.failure(&err.to_string());
    }
    if let Err(err) = feed_prompt(" ", &mut session, model, &mut output) {
        return report.failure(&err.to_string());
    }
    let Some(redone_logits) = output.all_logits.clone() else {
        return report.failure("Second run of model did not return logits.");
    };

    // Compare the logits
    for (idx, (&original, redone)) in original_logits.iter().zip(redone_logits).enumerate() {
        if original > redone + f32::EPSILON || original < redone - f32::EPSILON {
            return report.failure(&format!(
                "Expected logits to be the same after delete, but differed at {idx}, \
                expected {original}, but was {redone}."
            ));
        }
    }

    log::info!("`can_delete` test passed!");
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
pub struct DeleteReport {
    output: usize,
}

impl DeleteReport {
    fn failure(self, msg: &str) -> TestCaseReport {
        TestCaseReport {
            meta: TestCaseReportMeta::Error {
                error: msg.to_owned(),
            },
            report: crate::TestCaseReportInner::Delete(self),
        }
    }

    fn success(self) -> TestCaseReport {
        TestCaseReport {
            meta: TestCaseReportMeta::Success,
            report: crate::TestCaseReportInner::Delete(self),
        }
    }
}
