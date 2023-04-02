use llama_rs::{InferenceSnapshot, InferenceSnapshotRef, SnapshotError};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};
use zstd::zstd_safe::CompressionLevel;

const SNAPSHOT_COMPRESSION_LEVEL: CompressionLevel = 1;

pub fn load_from_disk(path: impl AsRef<Path>) -> Result<InferenceSnapshot, SnapshotError> {
    let mut reader = zstd::stream::read::Decoder::new(BufReader::new(File::open(path.as_ref())?))?;
    InferenceSnapshot::read(&mut reader)
}

pub fn write_to_disk(
    snap: &InferenceSnapshotRef<'_>,
    path: impl AsRef<Path>,
) -> Result<(), SnapshotError> {
    let mut writer = zstd::stream::write::Encoder::new(
        BufWriter::new(File::create(path.as_ref())?),
        SNAPSHOT_COMPRESSION_LEVEL,
    )?
    .auto_finish();

    snap.write(&mut writer)
}
