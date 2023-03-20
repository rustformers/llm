// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use llama::State;
use crate::download_model::download_model;
use crate::llama::{complete, is_active, start_model, stop_model};

mod llama;

mod download_model;

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
