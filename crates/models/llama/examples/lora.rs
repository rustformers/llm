
use llm_base::{load,Loader};
use llm_llama::Llama;

use llm_base::{
    ggml,LoraParameters,
    model::{common, HyperparametersWriteError, },
    util, FileType, InferenceParameters, InferenceSession, InferenceSessionConfig, KnownModel,
    LoadError, LoadProgress, Mmap, ModelParameters, OutputRequest, TensorLoader, TokenId,
    Vocabulary,
};

use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use llm_base::load_progress_callback_stdout as load_callback;
use llm_base::InferenceRequest;
use std::{convert::Infallible, env::args, io::Write};

fn main(){
    let base_model = r"D:\GGML_Models\llama-7B-q4_0.bin";
    let base_model_path = Path::new(base_model);

    let adapter = r"D:\GGML_Models\alpaca-7B-lora-adapter.bin";
    let adapter_path = Path::new(adapter);

    let model_params = ModelParameters {
        prefer_mmap: true,
        n_context_tokens:2048,
        lora_adapter: Some(adapter_path.to_owned()),
        ..Default::default()
    };
    let model:Llama = load(base_model_path,model_params,load_callback).unwrap();

    let mut session = model.start_session(Default::default());
    let input = "The meaning of life is";

    let output = session.infer::<Infallible>(&model,
    &mut rand::thread_rng(),
    &InferenceRequest {
        prompt: input,
        ..Default::default()
    },
    // OutputRequest
    &mut Default::default(),
    |t| {
        print!("{t}");
        std::io::stdout().flush().unwrap();

        Ok(())
    },);
}