//! Convert a model from `pth` to `ggml` format.

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

use crate::{Hyperparameters, Vocabulary};

impl From<&Path> for Vocabulary {
    fn from(path: &Path) -> Self {
        let mut f = File::open(path).unwrap();
        let mut contents = Vec::new();
        f.read_to_end(&mut contents).unwrap();

        let proto = protobuf::parse_from_bytes::<ModelProto>(contents.as_slice()).unwrap();
        let mut id_to_token = vec![];
        let mut id_to_token_score = vec![];
        let mut token_to_id = HashMap::new();
        let mut max_token_length = 0;

        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            let word = piece.get_piece().to_string();
            max_token_length = max_token_length.max(word.len());
            id_to_token.push(word.clone());
            token_to_id.insert(word, idx as i32);
            id_to_token_score.push(piece.get_score());
        }
        Vocabulary {
            id_to_token,
            id_to_token_score,
            token_to_id,
            max_token_length,
        }
    }
}

fn get_n_parts(dim: i32) -> usize {
    match dim {
        4096 => 1,
        5120 => 2,
        6656 => 4,
        8192 => 8,
        _ => panic!("Invalid dimension"),
    }
}
fn get_f_type(f32: bool) -> String {
    match f32 {
        true => "f32",
        false => "f16",
    }
    .to_string()
}

fn load_hyperparameters(path: &Path, f32: bool, vocab: &Vocabulary) -> Hyperparameters {
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
        f16_: match f32 {
            true => 0,
            false => 1,
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
        let text = match token {
            _ if token.contains("<unk>") => " \u{2047} ".as_bytes().to_vec(),
            _ if token.contains("s>") => vec![],
            _ if token.len() == 6 && token.contains("<0x") => {
                vec![u8::from_str_radix(&token[3..5], 16).unwrap()]
            }
            _ => token.replace('\u{2581}', " ").as_bytes().to_vec(),
        };
        values.extend((text.len() as i32).to_le_bytes());
        values.extend(&text);
        values.extend(vocab.id_to_token_score[i].to_le_bytes());
    }

    file.write_all(&values)
        .expect("Unable to write headers to the file.");

    Ok(())
}

/// Converts a `pth` file to a `ggml` file.
pub fn convert_pth_to_ggml(dir: &String, f32: bool) {
    let path = Path::new(dir);

    let tokenizer_path = path.parent().unwrap().join("tokenizer.model");
    let vocab = Vocabulary::from(tokenizer_path.as_path());

    let hparams = load_hyperparameters(path, f32, &vocab);
    let n_parts = get_n_parts(hparams.n_embd.try_into().unwrap());

    for i in 0..n_parts {
        let fname_out = path.join(format!("rust-model-{}.bin", get_f_type(f32)));
        let mut file = File::create(fname_out).expect("Unable to create file");
        write_header(file.borrow_mut(), &hparams).unwrap();
        write_tokens(file.borrow_mut(), &vocab).unwrap();

        let _fname_model = path.join(format!("consolidated.0{}.pth", i));
        // Todo process and write variables
    }
}
