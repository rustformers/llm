use crate::{
    loader::FileContext, model::HyperparametersWriteError, util, Hyperparameters, LoadError, Loader,
};

use ggml::format::TensorLoadInfo;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
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

/// [LoRA](https://arxiv.org/abs/2106.09685) patches for a model.
pub struct LoraPatches {
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

impl LoraPatches {
    /// Loads LoRA patches from a file.
    pub fn new(path: &PathBuf) -> Result<Self, LoadError> {
        // Read the LoRA file
        let lora_file = File::open(path).map_err(|e| LoadError::OpenFileFailed {
            source: e,
            path: path.to_owned(),
        })?;
        let mut lora_reader = BufReader::new(&lora_file);
        // TODO: Consider updating the progress callback to report the progress of the LoRA file.
        // Most LoRAs are small enough that this is not necessary, but it would be nice to have.
        let mut lora_loader: Loader<LoraParameters, _> = Loader::new(|_| {});
        ggml::format::load(&mut lora_reader, &mut lora_loader)
            .map_err(|err| LoadError::from_format_error(err, path.to_owned()))?;

        // Collect the names of the tensors that should be patched
        let tensors_to_patch = lora_loader
            .tensors
            .keys()
            .filter_map(|k| Some(k.rsplit_once('.')?.0.to_owned()))
            .collect();

        // Return the LoRA patches
        Ok(LoraPatches {
            scaling: lora_loader.hyperparameters.calculate_scaling(),
            tensors: lora_loader.tensors,
            tensors_to_patch,
            file: lora_file,
            path: path.to_owned(),
        })
    }

    /// Patch a tensor via LoRA
    pub fn patch(
        &mut self,
        info: &TensorLoadInfo,
        tensor: &mut ggml::Tensor,
    ) -> Result<(), LoadError> {
        // Check if we need to patch this tensor
        let name = &info.name;
        if !self.tensors_to_patch.contains(name) {
            return Ok(());
        }

        let a_info = self.get_info(&format!("{}.loraA", name))?;
        let b_info = self.get_info(&format!("{}.loraB", name))?;

        // Calculate the size of the patch context via the following steps:
        // 1. Calculate the size of the two `a` and `b` tensors
        // 2. Calculate the size of the original tensor
        // 3. Calculate the  size of the `ba` and `scaling` tensors. These have the same dimensions as the original tensor, but are of the element type of the `a` or `b` tensor e.g. fp16
        // 4. Add 20% as ggml overhead
        let ba_size = ggml::format::tensor_size(a_info.element_type, info.dims().iter().product());
        let patch_context_size = ((a_info.calc_absolute_size(false)
            + b_info.calc_absolute_size(false)
            + info.calc_absolute_size(false)
            + ba_size * 2) as f32
            * 1.2) as usize;
        // Create a temporary context for the patching operations
        let patch_context = ggml::Context::init(patch_context_size, true);
        let mut patch_file = FileContext::new(&patch_context, &mut self.file, &self.path, None);

        // Load the A and B tensors
        let a = patch_file.get_tensor(&a_info)?;
        let b = patch_file.get_tensor(&b_info)?;

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

    fn get_info(&self, name: &str) -> Result<TensorLoadInfo, LoadError> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or(LoadError::UnknownTensor {
                path: self.path.to_owned(),
                tensor_name: name.to_owned(),
            })
    }
}

/// A collection of [LoRA](https://arxiv.org/abs/2106.09685) patches which can be applied to a model.
pub struct LoraAdapter {
    /// The LoRA patches.
    pub patches: Vec<LoraPatches>,
}

impl LoraAdapter {
    /// Loads LoRA patches from the provided paths and returns a new adapter.
    pub fn new(paths: &[PathBuf]) -> Result<Self, LoadError> {
        let patches: Vec<LoraPatches> =
            paths.iter().map(|p| LoraPatches::new(p).unwrap()).collect();
        Ok(LoraAdapter { patches })
    }

    /// Applies this LoRA adapter to the provided tensor.
    pub fn apply(
        &mut self,
        info: &TensorLoadInfo,
        tensor: &mut ggml::Tensor,
    ) -> Result<(), LoadError> {
        for patch in &mut self.patches {
            patch.patch(info, tensor)?;
        }
        Ok(())
    }
}
