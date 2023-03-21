// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use crate::download_model::download_model;
use crate::llama::{complete, is_active, start_model, stop_model};
use llama::State;
use std::sync::Mutex;

mod download_model;
mod llama;
mod toast;

fn main() {
    tauri::Builder::default()
        .manage(State(Mutex::default()))
        .invoke_handler(tauri::generate_handler![
            complete,
            download_model,
            start_model,
            stop_model,
            is_active,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
