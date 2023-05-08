use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use llm::{InferenceSession, InferenceSessionConfig, Model};

use zstd::{
    stream::{read::Decoder, write::Encoder},
    zstd_safe::CompressionLevel,
};

const SNAPSHOT_COMPRESSION_LEVEL: CompressionLevel = 1;

/// Read or create a session
pub fn read_or_create_session(
    model: &dyn Model,
    persist_session: Option<&Path>,
    load_session: Option<&Path>,
    inference_session_config: InferenceSessionConfig,
) -> (InferenceSession, bool) {
    fn load(model: &dyn Model, path: &Path) -> InferenceSession {
        let file = unwrap_or_exit(File::open(path), || format!("Could not open file {path:?}"));
        let decoder = unwrap_or_exit(Decoder::new(BufReader::new(file)), || {
            format!("Could not create decoder for {path:?}")
        });
        let snapshot = unwrap_or_exit(bincode::deserialize_from(decoder), || {
            format!("Could not deserialize inference session from {path:?}")
        });
        let session = unwrap_or_exit(InferenceSession::from_snapshot(snapshot, model), || {
            format!("Could not convert snapshot from {path:?} to session")
        });
        log::info!("Loaded inference session from {path:?}");
        session
    }

    match (persist_session, load_session) {
        (Some(path), _) if path.exists() => (load(model, path), true),
        (_, Some(path)) => (load(model, path), true),
        _ => (model.start_session(inference_session_config), false),
    }
}

/// Write the session
pub fn write_session(mut session: InferenceSession, path: &Path) {
    // SAFETY: the session is consumed here, so nothing else can access it.
    let snapshot = unsafe { session.get_snapshot() };
    let file = unwrap_or_exit(File::create(path), || {
        format!("Could not create file {path:?}")
    });
    let encoder = unwrap_or_exit(
        Encoder::new(BufWriter::new(file), SNAPSHOT_COMPRESSION_LEVEL),
        || format!("Could not create encoder for {path:?}"),
    );
    unwrap_or_exit(
        bincode::serialize_into(encoder.auto_finish(), &snapshot),
        || format!("Could not serialize inference session to {path:?}"),
    );
    log::info!("Successfully wrote session to {path:?}");
}

fn unwrap_or_exit<T, E: Error>(result: Result<T, E>, error_message: impl Fn() -> String) -> T {
    match result {
        Ok(t) => t,
        Err(err) => {
            log::error!("{}. Error: {err}", error_message());
            std::process::exit(1);
        }
    }
}
