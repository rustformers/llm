//! Convert a model from `pth` to `ggml` format.
//!
//! This is *incomplete* and does not convert the weights. It only converts the
//! vocabulary and hyperparameters. It is included as a preliminary step to
//! full conversion.
///
/// For reference, see [the PR](https://github.com/rustformers/llama-rs/pull/83).
use rust_tokenizers::preprocessing::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use serde::Deserialize;
use std::{
    borrow::BorrowMut,
    collections::HashMap,
    fs::{read_to_string, File},
    io::{Read, Write},
    path::Path,
    vec,
};

use crate::{util, Hyperparameters, Vocabulary};

/// Converts a `pth` file to a `ggml` file.
pub fn convert_pth_to_ggml(model_directory: &Path, element_type: ggml::Type) {
    let tokenizer_path = model_directory.parent().unwrap().join("tokenizer.model");
    let vocab = load_vocabulary(tokenizer_path.as_path());

    let hparams = load_hyperparameters(model_directory, element_type, &vocab);

    let model_files = util::find_all_model_files(model_directory).unwrap();

    for (i, _file) in model_files.iter().enumerate() {
        let fname_out = model_directory.join(format!("rust-model-{element_type}.bin"));
        let mut file = File::create(fname_out).expect("Unable to create file");
        write_header(file.borrow_mut(), &hparams).unwrap();
        write_tokens(file.borrow_mut(), &vocab).unwrap();

        let _fname_model = model_directory.join(format!("consolidated.0{i}.pth"));
        // Todo process and write variables
    }
}

fn load_vocabulary(path: &Path) -> Vocabulary {
    let mut f = File::open(path).unwrap();
    let mut contents = Vec::new();
    f.read_to_end(&mut contents).unwrap();

    let proto = protobuf::parse_from_bytes::<ModelProto>(contents.as_slice()).unwrap();
    let mut id_to_token = vec![];
    let mut id_to_token_score = vec![];
    let mut token_to_id = HashMap::new();
    let mut max_token_length = 0;

    // TODO: Does the original model use valid UTF-8 for its tokens? This seems a little suspect to me.
    for (idx, piece) in proto.get_pieces().iter().enumerate() {
        let word = piece.get_piece().as_bytes();
        max_token_length = max_token_length.max(word.len());
        id_to_token.push(word.to_owned());
        token_to_id.insert(word.to_owned(), idx as i32);
        id_to_token_score.push(piece.get_score());
    }
    Vocabulary {
        id_to_token,
        id_to_token_score,
        token_to_id,
        max_token_length,
    }
}

fn load_hyperparameters(
    path: &Path,
    element_type: ggml::Type,
    vocab: &Vocabulary,
) -> Hyperparameters {
    #[derive(Deserialize)]
    struct HyperParametersJson {
        dim: usize,
        multiple_of: usize,
        n_heads: usize,
        n_layers: usize,
        vocab_size: isize,
    }

    let json = read_to_string(path.join("params.json")).expect("Unable to read file");
    let json: HyperParametersJson = serde_json::from_str(&json).expect("Unable to parse json");
    Hyperparameters {
        f16_: match element_type {
            ggml::Type::F32 => 0,
            ggml::Type::F16 => 1,
            ggml::Type::Q4_0 => 2,
            ggml::Type::Q4_1 => 3,
            _ => panic!("unsupported element type"),
        },
        n_ctx: 0,
        n_embd: json.dim,
        n_head: json.n_heads,
        n_layer: json.n_layers,
        n_vocab: match json.vocab_size {
            -1 => vocab.id_to_token.len(),
            _ => json.vocab_size as usize,
        },
        n_mult: json.multiple_of,
        n_rot: json.dim / json.n_heads,
    }
}

fn write_header(fout: &mut File, hparams: &Hyperparameters) -> Result<(), String> {
    let values: Vec<i32> = vec![
        0x67676d66, // magic: ggmf in hex
        1,          // file version
        i32::try_from(hparams.n_vocab).unwrap(),
        i32::try_from(hparams.n_embd).unwrap(),
        i32::try_from(hparams.n_mult).unwrap(),
        i32::try_from(hparams.n_head).unwrap(),
        i32::try_from(hparams.n_layer).unwrap(),
        i32::try_from(hparams.n_embd / hparams.n_head).unwrap(),
        i32::try_from(hparams.f16_).unwrap(),
    ];
    let mut packed_values: Vec<u8> = vec![];

    for value in values {
        packed_values.extend(&value.to_le_bytes());
    }

    fout.write_all(&packed_values)
        .expect("Unable to write headers to the file.");

    Ok(())
}

fn write_tokens(file: &mut File, vocab: &Vocabulary) -> Result<(), String> {
    let mut values: Vec<u8> = vec![];
    for (i, token) in vocab.id_to_token.iter().enumerate() {
        // TODO: Not sure what the behaviour should be if the token is not valid UTF-8.
        //
        // Switching to the HF tokenizer should fix this.
        let text = if let Ok(token) = std::str::from_utf8(token) {
            match token {
                _ if token.contains("<unk>") => " \u{2047} ".as_bytes().to_vec(),
                _ if token.contains("s>") => vec![],
                _ if token.len() == 6 && token.contains("<0x") => {
                    vec![u8::from_str_radix(&token[3..5], 16).unwrap()]
                }
                _ => token.replace('\u{2581}', " ").as_bytes().to_vec(),
            }
        } else {
            token.clone()
        };
        values.extend((text.len() as i32).to_le_bytes());
        values.extend(&text);
        values.extend(vocab.id_to_token_score[i].to_le_bytes());
    }

    file.write_all(&values)
        .expect("Unable to write headers to the file.");

    Ok(())
}
