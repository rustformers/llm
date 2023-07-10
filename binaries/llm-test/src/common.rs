//! Tests that are run on every model, regardless of config.

pub(super) fn can_send<M: llm::KnownModel + 'static>(model: M) -> anyhow::Result<M> {
    let model = std::thread::spawn(move || model)
        .join()
        .map_err(|e| anyhow::anyhow!("Failed to join thread: {e:?}"));

    log::info!("`can_send` test passed!");

    model
}

pub(super) fn can_roundtrip_hyperparameters<M: llm::KnownModel + 'static>(
    model: &M,
) -> anyhow::Result<()> {
    fn test_hyperparameters<M: llm::Hyperparameters>(hyperparameters: &M) -> anyhow::Result<()> {
        let mut data = vec![];
        hyperparameters.write_ggml(&mut data)?;
        let new_hyperparameters =
            <M as llm::Hyperparameters>::read_ggml(&mut std::io::Cursor::new(data))?;

        assert_eq!(hyperparameters, &new_hyperparameters);

        log::info!("`can_roundtrip_hyperparameters` test passed!");

        Ok(())
    }

    test_hyperparameters(model.hyperparameters())
}
