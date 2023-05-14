use crate::{
    loader::FileContext, model::HyperparametersWriteError, util, Hyperparameters, LoadError,
};

use ggml::format::TensorLoadInfo;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::PathBuf,
};

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
/// Parameters for a [LoRA](https://arxiv.org/abs/2106.09685) adapter.
pub struct LoraParameters {
    /// r
    pub r: i32,
    /// alpha
    pub alpha: i32,
}

impl LoraParameters {
    /// Returns the scaling factor for the LoRA adapter.
    pub fn calculate_scaling(&self) -> f32 {
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
        // LoRA adapters do not have a vocabulary.
        0
    }
}

/// A LoRA adapter for a model.
pub struct LoraAdapter {
    /// Scaling to apply to the LoRA weights.
    pub scaling: f32,
    /// The tensors of the LoRA.
    pub tensors: HashMap<String, TensorLoadInfo>,
    /// Names of the tensors that should be patched.
    pub tensors_to_patch: HashSet<String>,
    /// File containing the LoRA weights.
    pub file: File,
    /// Path to the LoRA file.
    pub path: PathBuf,
}

impl LoraAdapter {
    /// Creates a new LoRA adapter.
    pub fn new(
        scaling: f32,
        tensors: HashMap<String, TensorLoadInfo>,
        file: File,
        path: PathBuf,
    ) -> Self {
        let tensors_to_patch = tensors
            .keys()
            .filter_map(|k| Some(k.rsplit_once('.')?.0.to_owned()))
            .collect();

        LoraAdapter {
            scaling,
            tensors,
            tensors_to_patch,
            file,
            path,
        }
    }

    /// Apply this LoRA adapter to a tensor.
    pub fn patch(&mut self, name: &str, tensor: &mut ggml::Tensor) -> Result<(), LoadError> {
        // Check if we need to patch this tensor
        if !self.tensors_to_patch.contains(name) {
            return Ok(());
        }
        let a_name = format!("{}.loraA", name);
        let b_name = format!("{}.loraB", name);

        // Get the a and b tensor infos
        let a_tensor_info = self.tensors.get(&a_name).ok_or(LoadError::UnknownTensor {
            path: self.path.clone(),
            tensor_name: a_name.to_owned(),
        })?;

        let b_tensor_info = self.tensors.get(&b_name).ok_or(LoadError::UnknownTensor {
            path: self.path.clone(),
            tensor_name: b_name.to_owned(),
        })?;

        // TODO: calculate the size dynmaically
        let patch_context_size = 1024 * 1024 * 128;

        // Create a temporary context for the patching operations
        let patch_context = ggml::Context::init(patch_context_size, true);
        let mut patch_file = FileContext::new(&patch_context, &mut self.file, &self.path, None);

        // Load the A and B tensors
        let a = patch_file.get_tensor(a_tensor_info)?;
        let b = patch_file.get_tensor(b_tensor_info)?;

        // Build a ggml context and apply the patch
        // TODO: maybe pass the model's thread count to this context
        let mut gf = ggml::ComputationGraph::new(8);

        // LoRA formula: w = w + ba*s
        let mut ba = patch_context.op_mul_mat(&a, &b);
        if self.scaling != 1.0 {
            let scaling_tensor = patch_context.new_f32(self.scaling);
            ba = patch_context.op_scale(&ba, &scaling_tensor);
        }
        let mut output = patch_context.op_add(tensor, &ba);

        // Compute the graph
        gf.build_forward_expand(&output);
        patch_context.graph_compute(&mut gf);

        // Overwrite the original tensor.
        // The `output` and the `target_tensor` are not from the same context,
        // so this should be fine.
        unsafe {
            std::ptr::copy_nonoverlapping(output.data(), tensor.data(), tensor.nbytes());
        }

        Ok(())
    }
}
