use tauri::Window;

#[derive(serde::Serialize, Clone)]
struct Toast {
    message: String,
}

#[tauri::command]
pub fn send_toast(message: &str, window: Window) {
    window
        .emit(
            "toast",
            Toast {
                message: message.to_string(),
            },
        )
        .unwrap();
}
