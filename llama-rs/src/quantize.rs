use crate::ggml::{
    quantize_q4_0, quantize_q4_1, FILE_MAGIC, FILE_MAGIC_UNVERSIONED, FORMAT_VERSION,
};
use crate::{Hyperparameters, LoadError, Vocabulary};
use half::f16;
use std::path::Path;

const FTYPE_STR: [&str; 4] = ["f32", "f16", "q4_0", "q4_1"];

#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub enum QuantizeLoadProgress<'a> {
    HyperparametersLoaded(&'a Hyperparameters),
    LoadingWeight {
        name: &'a str,
        size: [i32; 2],
        elements: i32,
        ftype: &'a str,
    },
    Quantizing,
    Quantized {
        original_size: f32,
        reduced_size: f32,
        history: Vec<f32>,
    },
    Skipped {
        size: f32,
    },
    Finished {
        original_size: f32,
        reduced_size: f32,
        history: Vec<f32>,
    },
}

pub fn quantize(
    file_name_in: impl AsRef<Path>,
    file_name_out: impl AsRef<Path>,
    itype: u8,
    qk: u8,
    load_progress_callback: impl Fn(QuantizeLoadProgress),
) -> Result<(), LoadError> {
    use crate::file::*;

    if itype != 2 && itype != 3 {
        return Err(LoadError::InvalidItype(itype));
    }

    let file_in = file_name_in.as_ref();
    let mut finp = BufReader::new(File::open(file_in).map_err(|e| LoadError::OpenFileFailed {
        source: e,
        path: file_in.to_owned(),
    })?);

    let file_out = file_name_out.as_ref();
    let mut fout =
        BufWriter::new(
            File::create(file_out).map_err(|e| LoadError::CreateFileFailed {
                source: e,
                path: file_out.to_owned(),
            })?,
        );

    // Verify magic
    {
        let magic = rw_u32(&mut finp, &mut fout)?;
        if magic == FILE_MAGIC_UNVERSIONED {
            return Err(LoadError::UnversionedMagic);
        }
        if magic != FILE_MAGIC {
            return Err(LoadError::InvalidMagic {
                path: file_in.to_owned(),
            });
        }

        let format_version = rw_u32(&mut finp, &mut fout)?;
        if format_version != FORMAT_VERSION {
            return Err(LoadError::InvalidFormatVersion {
                value: format_version,
            });
        }
    }

    let mut hparams = Hyperparameters::default();

    // Load parameters
    {
        hparams.n_vocab = rw_i32(&mut finp, &mut fout)?;
        hparams.n_embd = rw_i32(&mut finp, &mut fout)?;
        hparams.n_mult = rw_i32(&mut finp, &mut fout)?;
        hparams.n_head = rw_i32(&mut finp, &mut fout)?;
        hparams.n_layer = rw_i32(&mut finp, &mut fout)?;
        hparams.n_rot = rw_i32(&mut finp, &mut fout)?;
        hparams.f16_ = rw_i32(&mut finp, &mut fout)?;
    }
    load_progress_callback(QuantizeLoadProgress::HyperparametersLoaded(&hparams));

    // load vocab
    let mut vocab = Vocabulary {
        id_to_token: vec![],
        id_to_token_score: vec![],
        token_to_id: Default::default(),
        max_token_length: 0,
    };

    for i in 0..hparams.n_vocab {
        let len = rw_u32(&mut finp, &mut fout)? as usize;
        let word = rw_string(&mut finp, &mut fout, len)?;
        let score = rw_f32(&mut finp, &mut fout)?;

        vocab.token_to_id.insert(word.clone(), i);
        vocab.id_to_token.push(word);
        vocab.id_to_token_score.push(score);
    }

    // Load weights
    {
        let mut total_size_org: usize = 0;
        let mut total_size_new: usize = 0;

        let mut work: Vec<f32> = vec![];

        let mut data_u8: Vec<u8> = vec![];
        let mut data_f16: Vec<u16> = vec![];
        let mut data_f32: Vec<f32> = vec![];

        let mut hist_all: Vec<i64> = vec![0; 16];

        loop {
            let n_dims: i32;
            if let Ok(r) = read_i32(&mut finp) {
                n_dims = r;
            } else {
                break;
            }

            let length: usize;
            if let Ok(r) = read_i32(&mut finp) {
                length = r as usize;
            } else {
                break;
            }

            let mut ftype: i32;
            if let Ok(r) = read_i32(&mut finp) {
                ftype = r;
            } else {
                break;
            }

            let mut nelements = 1i32;
            let mut ne = [1i32, 1i32];
            for i in 0..n_dims {
                ne[i as usize] = read_i32(&mut finp)?;
                nelements *= ne[i as usize];
            }

            let name = read_string(&mut finp, length)?;

            load_progress_callback(QuantizeLoadProgress::LoadingWeight {
                name: &name,
                size: ne,
                elements: nelements,
                ftype: FTYPE_STR[ftype as usize],
            });

            // Quantize only 2D tensors
            let quantize = name.contains("weight") && n_dims == 2;

            if quantize {
                if ftype != 0 && ftype != 1 {
                    return Err(LoadError::InvalidFtype {
                        ftype,
                        path: file_in.to_owned(),
                    });
                }

                data_f32.resize(nelements as usize, 0.0);
                if ftype == 1 {
                    data_f16.resize(nelements as usize, 0);

                    let mut buffer = vec![0u8; (nelements * 2) as usize];
                    finp.read_exact(&mut buffer)?;
                    // Compute buffer
                    for (index, chunk) in buffer.chunks(2).enumerate() {
                        let i = u16::from_le_bytes([chunk[0], chunk[1]]);
                        data_f16[index] = i;

                        //data_f32[index] = ggml_fp16_to_fp32(i);
                        data_f32[index] = f16::from_bits(i).to_f32();
                    }
                } else {
                    let mut buffer = vec![0u8; (nelements * 4) as usize];
                    finp.read_exact(&mut buffer)?;

                    for (index, chunk) in buffer.chunks(4).enumerate() {
                        data_f32[index] =
                            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                }

                ftype = itype as i32;
            } else {
                // Determines the total bytes were dealing with
                let bpe = (nelements * if ftype == 0 { 4 } else { 2 }) as usize;

                data_u8.resize(bpe, 0);
                finp.read_exact(&mut data_u8)?;
            }

            // Write data
            fout.write_all(&n_dims.to_le_bytes())?;
            fout.write_all(&(length as i32).to_le_bytes())?;
            fout.write_all(&(ftype).to_le_bytes())?;

            for i in 0..n_dims {
                fout.write_all(&ne[i as usize].to_le_bytes())?;
            }
            fout.write_all(name.as_bytes())?;

            if quantize {
                load_progress_callback(QuantizeLoadProgress::Quantizing);
                work.resize(nelements as usize, 0.0);

                let mut hist_cur = vec![0; 16];

                let curr_size = if itype == 2 {
                    quantize_q4_0(
                        &mut data_f32,
                        &mut work,
                        nelements,
                        ne[0],
                        qk as i32,
                        &mut hist_cur,
                    )
                } else {
                    quantize_q4_1(
                        &mut data_f32,
                        &mut work,
                        nelements,
                        ne[0],
                        qk as i32,
                        &mut hist_cur,
                    )
                };

                // We divide curr size by 4
                for i in work.iter().take(curr_size / 4) {
                    fout.write_all(&i.to_le_bytes())?;
                }

                total_size_new += curr_size;

                let mut new_hist = vec![];
                for (i, val) in hist_cur.iter().enumerate() {
                    hist_all[i] += val;
                    new_hist.push(*val as f32 / nelements as f32);
                }

                load_progress_callback(QuantizeLoadProgress::Quantized {
                    original_size: nelements as f32 * 4.0 / 1024.0 / 1024.0,
                    reduced_size: curr_size as f32 / 1024.0 / 1024.0,
                    history: new_hist,
                });
            } else {
                fout.write_all(&data_u8)?;
                load_progress_callback(QuantizeLoadProgress::Skipped {
                    size: data_u8.len() as f32 / 1024.0 / 1024.0,
                });
                total_size_new += data_u8.len();
            }

            total_size_org += (nelements * 4) as usize;
        }

        let sum_all: i64 = hist_all.iter().sum();
        load_progress_callback(QuantizeLoadProgress::Finished {
            original_size: total_size_org as f32 / 1024.0 / 1024.0,
            reduced_size: total_size_new as f32 / 1024.0 / 1024.0,
            history: hist_all
                .iter()
                .map(|hist| *hist as f32 / sum_all as f32)
                .collect(),
        })
    }

    Ok(())
}
