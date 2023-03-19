use std::cmp::min;
use std::fs::File;
use std::io::Write;

use futures_util::StreamExt;
use reqwest::Client;
use serde::Serialize;
use tauri::Window;

#[derive(Serialize, Clone)]
pub struct Progress {
    pub downloaded: u64,
    pub total_size: u64,
}

#[tauri::command(async)]
pub async fn download_model(window: Window, url: &str, path: &str) -> Result<(), String> {
    let client = Client::new();
    let res = client
        .get(url)
        .send()
        .await
        .or(Err(format!("Failed to GET from '{}'", &url)))?;
    let total_size = res
        .content_length()
        .ok_or(format!("Failed to get content length from '{}'", &url))?;

    let mut file = File::create(path).or(Err(format!("Failed to create file '{}'", path)))?;
    let mut stream = res.bytes_stream();

    let mut progress = Progress {
        downloaded: 0,
        total_size,
    };
    while let Some(item) = stream.next().await {
        let chunk = item.or(Err(format!("Error while downloading file")))?;
        file.write_all(&chunk)
            .or(Err(format!("Error while writing to file")))?;
        let new = min(progress.downloaded + (chunk.len() as u64), total_size);

        progress.downloaded = new;
        window.emit("progress", progress.clone()).unwrap();
    }

    return Ok(());
}
