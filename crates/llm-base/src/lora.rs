use crate::{
    Hyperparameters,
    util,
    LoadError,
    model::HyperparametersWriteError,
};

use ggml::format::TensorLoadInfo;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
/// Parameters for a [LoRA](https://arxiv.org/abs/2106.09685) adapter 
pub struct LoraParameters{
    /// r
    pub r: i32,
    /// alpha
    pub alpha: i32,
}

impl LoraParameters{
    /// Returns the scaling factor for the LoRA adapter
    pub fn calculate_scaling(&self) -> f32{
        (self.alpha as f32) / (self.r as f32)
    }
}

impl Hyperparameters for LoraParameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        Ok(LoraParameters {
            r: util::read_i32(reader)?,
            alpha: util::read_i32(reader)?,
        })
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.r)?;
        util::write_i32(writer, self.alpha)?;
        Ok(())
    }

    fn n_vocabulary(&self) -> usize {
        //Lora adapters dont have a vocabulary
        0
    }
}

/// A LoRA adapter for a base model
pub struct LoraAdapter{
    /// Scaling to apply to the LoRA weights
    pub scaling: f32,
    /// Lora Tensors to apply
    pub tensors: HashMap<String, TensorLoadInfo>,
    ///File containing the LoRA weights
    pub file: File,
    ///Path to the LoRA file
    pub path: PathBuf,
}