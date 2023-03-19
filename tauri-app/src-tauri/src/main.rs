// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use crate::llama::complete;
use crate::download_model::download_model;

mod llama;

mod download_model;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![complete,download_model])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
