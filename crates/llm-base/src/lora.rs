use crate::{
    Hyperparameters,
    util,
    LoadError,
    model::HyperparametersWriteError
};

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
/// Parameters for a [LoRA](https://arxiv.org/abs/2106.09685) adapter 
pub struct LoraParameters{
    /// r
    pub r: i32,
    /// alpha
    pub alpha: i32,
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