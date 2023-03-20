use std::{cell::RefCell, convert::Infallible, path::Path};

use llama_rs::InferenceParameters;
use rand::SeedableRng;
use std::sync::Mutex;
use tauri::Window;

use crate::toast::send_toast;

#[derive(Clone, serde::Serialize)]
struct Payload {
    message: String,
    id: String,
}

#[derive(serde::Deserialize)]
pub struct Params {
    path: String,
    prompt: String,
    id: String,
    n_batch: Option<usize>,
    n_threads: Option<usize>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repeat_penalty: Option<f32>,
    temp: Option<f32>,
    num_predict: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            path: "".to_string(),
            prompt: "".to_string(),
            id: "".to_string(),
            n_batch: Some(8),
            n_threads: Some(num_cpus::get_physical()),
            top_k: Some(40),
            top_p: Some(0.95),
            repeat_penalty: Some(1.3),
            temp: Some(0.8),
            num_predict: Some(512),
        }
    }
}
pub struct SenderProps {
    params: Option<Params>,
    stop: Option<bool>,
}
pub struct AppState {
    sender: Option<flume::Sender<SenderProps>>,
}
impl Default for AppState {
    fn default() -> Self {
        Self { sender: None }
    }
}
pub struct State(pub Mutex<AppState>);

#[tauri::command]
pub fn is_active(state: tauri::State<State>) -> bool {
    state.0.lock().unwrap().sender.is_some()
}

#[tauri::command(async)]
pub fn start_model(window: Window, path: String, state: tauri::State<State>) {
    let rx = {
        let mut app_state = state.0.lock().unwrap();
        if app_state.sender.is_some() {
            send_toast("Already running!", window);
            return;
        }
        if !Path::new(&path).exists() {
            send_toast("Invalid model path!", window);
            return;
        }
        let (tx, rx) = flume::unbounded::<SenderProps>();
        app_state.sender = Some(tx);
        rx
    };

    std::thread::spawn(move || {
        let (model, vocab) =
            llama_rs::Model::load(path, 512, |_| {}).expect("Could not load model");
        let mut rng = rand::rngs::StdRng::from_entropy();
        let mut session = model.start_session(64);

        loop {
            let props = rx.recv().unwrap();
            if let Some(params) = props.params {
                let inference_params = InferenceParameters {
                    n_threads: params.n_threads.unwrap_or_default() as i32,
                    n_batch: params.n_batch.unwrap_or_default(),
                    top_k: params.top_k.unwrap_or_default(),
                    top_p: params.top_p.unwrap_or_default(),
                    repeat_penalty: params.repeat_penalty.unwrap_or_default(),
                    temp: params.temp.unwrap_or_default(),
                };
                let message = RefCell::new(String::new());
                let res = session.inference_with_prompt::<Infallible>(
                    &model,
                    &vocab,
                    &inference_params,
                    &params.prompt,
                    Some(params.num_predict.unwrap_or_default()),
                    &mut rng,
                    |t| {
                        message.borrow_mut().push_str(&t.to_string());
                        println!("{}", t.to_string());
                        let borrow = message.borrow();
                        let split = borrow.split(&params.prompt);
                        let res = match split.clone().count() {
                            1 => "",
                            _ => split.last().unwrap(),
                        };
                        window
                            .emit(
                                "message",
                                Payload {
                                    id: params.id.clone(),
                                    message: res.to_string(),
                                },
                            )
                            .unwrap();
                        Ok(())
                    },
                );

                match res {
                    Ok(_) => (),
                    Err(llama_rs::InferenceError::ContextFull) => {
                        send_toast("Context full", window.clone());
                    }
                    Err(llama_rs::InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
                }
            }
            if props.stop.unwrap_or_default() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    });
}

#[tauri::command(async)]
pub fn complete(window: Window, params: Params, state: tauri::State<State>) {
    {
        if state.0.lock().unwrap().sender.is_none() {
            start_model(window, params.path.clone(), state.clone())
        }
    }
    state
        .0
        .lock()
        .unwrap()
        .sender
        .as_mut()
        .unwrap()
        .send(SenderProps {
            params: Some(params),
            stop: None,
        })
        .unwrap();
}

#[tauri::command(async)]
pub fn stop_model(state: tauri::State<State>) {
    let mut app_state = state.0.lock().unwrap();
    if app_state.sender.is_some() {
        app_state
            .sender
            .as_mut()
            .unwrap()
            .send(SenderProps {
                params: None,
                stop: Some(true),
            })
            .unwrap();
        app_state.sender = None;
    }
}
