
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

fn main(){
    let base_model = r"D:\GGML_Models\llama-7B-q4_0.bin";
    let base_model_path = Path::new(base_model);

    let adapter = r"D:\GGML_Models\alpaca-7B-lora-adapter.bin";
    let adapter_path = Path::new(adapter);

    let mut adapter_loader:Loader<LoraParameters, _> = Loader::new(|x|{});
    

    let file = File::open(adapter_path).map_err(|e| LoadError::OpenFileFailed {
        source: e,
        path: adapter_path.to_owned(),
    }).unwrap();

    let mut reader = BufReader::new(file);
    ggml::format::load(&mut reader, &mut adapter_loader)
        .map_err(|err| LoadError::from_format_error(err, adapter_path.to_owned())).unwrap();

    for key in adapter_loader.tensors.keys(){
        println!("{}",key);
    }

    let model_params = ModelParameters {
        prefer_mmap: true,
        n_context_tokens:2048,
        ..Default::default()
    };
    let model:Llama = load(base_model_path,Default::default(),|x|{}).unwrap();
    println!("Model loaded");
}