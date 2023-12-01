use crate::loader::{Source, TensorLoadError, TensorLoader};

use ggml::{
    format::gguf::{Gguf, TensorInfo},
    GraphExecutionPlan,
};
use indexmap::IndexMap;
use std::{collections::HashSet, path::PathBuf};

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

/// [LoRA](https://arxiv.org/abs/2106.09685) adapter for a model.
pub struct LoraAdapter {
    /// Scaling to apply to the LoRA weights.
    pub scaling: f32,
    /// The tensors of the LoRA.
    pub tensors: IndexMap<String, TensorInfo>,
    /// Names of the tensors that should be patched.
    pub tensors_to_patch: HashSet<String>,
    /// Source containing the LoRA weights.
    pub source: Box<dyn Source>,
    /// Path to the LoRA file.
    pub path: PathBuf,
    /// The loaded GGUF for the LoRA.
    pub gguf: Gguf,
}

impl LoraAdapter {
    /// Patch a tensor via LoRA
    pub fn patch(
        &mut self,
        name: &str,
        info: &TensorInfo,
        tensor: &mut ggml::Tensor,
    ) -> Result<(), TensorLoadError> {
        // Check if we need to patch this tensor
        if !self.tensors_to_patch.contains(name) {
            return Ok(());
        }

        let a_name = format!("{}.loraA", name);
        let a_info = self.get_info(&a_name)?;

        let b_name = format!("{}.loraB", name);
        let b_info = self.get_info(&b_name)?;

        let must_scale = self.scaling != 1.0;
        // Calculate the size of the patch context via the following steps:
        // 1. Calculate the size of the two `a` and `b` tensors
        // 2. Calculate the size of the original tensor
        // 3. Calculate the  size of the `ba` and tensors. It has the same dimensions as the original tensor, but is of the element type of the `a` or `b` tensor e.g. fp16
        let ba_size =
            ggml::format::tensor_size(a_info.element_type, info.dimensions.iter().product());
        let mut patch_context_size = a_info.calc_absolute_size(false)
            + b_info.calc_absolute_size(false)
            + info.calc_absolute_size(false)
            + ba_size;

        // 3b. (Optional) If we need to scale the `ba` tensor, we need to allocate for a second `ba` and the `scaled` tensors which will be crated as an `f32` tensor.
        if must_scale {
            let scaled_size =
                ggml::format::tensor_size(ggml::ElementType::F32, info.dimensions.iter().product());
            patch_context_size += scaled_size + ba_size;
        }

        // 4. Add 5% as ggml overhead (I dont know why this is needed but the calculation is always a few 100-1000 bytes off)
        patch_context_size = patch_context_size + (patch_context_size / 20);

        // Create a temporary context for the patching operations
        // TODO: test if GPU can be enabled (make it configurable)
        let patch_context = ggml::Context::new_with_allocate(patch_context_size);
        let mut loader = TensorLoader {
            source: self.source.as_mut(),
            context: patch_context,
            gguf: &self.gguf,
        };

        // Load the A and B tensors
        let (a, _) = loader.load(&a_name)?;
        let (b, _) = loader.load(&b_name)?;

        // Build a ggml context and apply the patch
        let patch_context = loader.finish();
        let mut gf = patch_context.create_compute_graph();

        // LoRA formula: w = w + ba*s
        let mut ba = patch_context.op_mul_mat(&a, &b);
        if must_scale {
            let scaling_tensor = patch_context.new_f32(self.scaling);
            ba = patch_context.op_scale(&ba, &scaling_tensor);
        }
        let mut output = patch_context.op_add(tensor, &ba);

        // Compute the graph
        gf.build_forward_expand(&output);

        //TODO: maybe pass the model's thread count to this context
        let mut work_buffer = vec![0u8];
        let mut plan = GraphExecutionPlan::new(&mut gf, 8);
        plan.execute(&mut work_buffer);

        // Overwrite the original tensor.
        // The `output` and the `target_tensor` are not from the same context,
        // so this should be fine.
        unsafe {
            std::ptr::copy_nonoverlapping(output.data(), tensor.data(), tensor.nbytes());
        }

        Ok(())
    }

    fn get_info(&self, name: &str) -> Result<TensorInfo, TensorLoadError> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or(TensorLoadError::UnknownTensor {
                tensor_name: name.to_owned(),
            })
    }
}
