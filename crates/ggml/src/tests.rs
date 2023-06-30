use std::{
    collections::BTreeMap,
    error::Error,
    io::{BufRead, Write},
};

use crate::*;
use rand::{distributions::Uniform, prelude::*};

#[derive(Debug)]
struct DummyError;
impl std::fmt::Display for DummyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}
impl Error for DummyError {}

#[test]
fn can_roundtrip_loader_and_saver_ggml() {
    let tokenizer = vec![
        ("blazingly".as_bytes().to_vec(), 0.0),
        ("fast".as_bytes().to_vec(), 0.0),
        ("memory".as_bytes().to_vec(), 0.0),
        ("efficient".as_bytes().to_vec(), 0.0),
    ];

    roundtrip_test(format::SaveContainerType::Ggml, tokenizer).unwrap();
}

#[test]
fn will_fail_on_scored_ggml_save() {
    let tokenizer = vec![
        ("blazingly".as_bytes().to_vec(), 0.1),
        ("fast".as_bytes().to_vec(), 0.2),
        ("memory".as_bytes().to_vec(), 0.3),
        ("efficient".as_bytes().to_vec(), 0.4),
    ];

    assert_eq!(
        roundtrip_test(format::SaveContainerType::Ggml, tokenizer)
            .unwrap_err()
            .to_string(),
        format::SaveError::<std::io::Error>::VocabularyScoringNotSupported.to_string()
    );
}

#[test]
fn can_roundtrip_loader_and_saver_ggjt_v3() {
    let tokenizer = vec![
        ("blazingly".as_bytes().to_vec(), 0.1),
        ("fast".as_bytes().to_vec(), 0.2),
        ("memory".as_bytes().to_vec(), 0.3),
        ("efficient".as_bytes().to_vec(), 0.4),
    ];

    roundtrip_test(format::SaveContainerType::GgjtV3, tokenizer).unwrap();
}

fn roundtrip_test(
    save_container_type: format::SaveContainerType,
    tokenizer: Vec<(Vec<u8>, f32)>,
) -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    let element_type = crate::Type::F16;
    let model = Model {
        hyperparameters: Hyperparameters {
            some_hyperparameter: random(),
            some_other_hyperparameter: random(),
            tokenizer_size: tokenizer.len().try_into()?,
        },
        tokenizer,
        tensors: (0..10)
            .map(|i| {
                let n_dims = Uniform::from(1..3).sample(&mut rng);
                let dims = (0..n_dims)
                    .map(|_| Uniform::from(1..10).sample(&mut rng))
                    .chain(std::iter::repeat(1).take(2 - n_dims))
                    .collect::<Vec<_>>();

                let n_elements = dims.iter().product::<usize>();
                let data = (0..format::data_size(element_type, n_elements))
                    .map(|_| random())
                    .collect::<Vec<_>>();

                (
                    format!("tensor_{}", i),
                    format::TensorSaveInfo {
                        n_dims,
                        dims: dims.try_into().unwrap(),
                        element_type,
                        data,
                    },
                )
            })
            .collect(),
    };

    // Save the model.
    let mut buffer = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buffer);
    let mut save_handler = MockSaveHandler { model: &model };
    format::save(
        &mut cursor,
        &mut save_handler,
        save_container_type,
        &model.tokenizer,
        &model.tensors.keys().cloned().collect::<Vec<String>>(),
    )?;

    // Load the model and confirm that it is the same as the original.
    let mut cursor = std::io::Cursor::new(&buffer);
    let mut load_handler = MockLoadHandler {
        data: &buffer,
        loaded_model: Model::default(),
        expected_container_type: save_container_type.into(),
    };
    format::load(&mut cursor, &mut load_handler)?;
    assert_eq!(load_handler.loaded_model, model);

    Ok(())
}

#[derive(Default, PartialEq, Debug)]
struct Hyperparameters {
    some_hyperparameter: u32,
    some_other_hyperparameter: u32,
    tokenizer_size: u32,
}
impl Hyperparameters {
    fn read(reader: &mut dyn BufRead) -> Result<Self, std::io::Error> {
        Ok(Self {
            some_hyperparameter: util::read_u32(reader)?,
            some_other_hyperparameter: util::read_u32(reader)?,
            tokenizer_size: util::read_u32(reader)?,
        })
    }

    fn write(&self, writer: &mut dyn Write) -> Result<(), std::io::Error> {
        util::write_u32(writer, self.some_hyperparameter)?;
        util::write_u32(writer, self.some_other_hyperparameter)?;
        util::write_u32(writer, self.tokenizer_size)?;
        Ok(())
    }
}

#[derive(Default, PartialEq, Debug)]
struct Model {
    hyperparameters: Hyperparameters,
    tokenizer: Vec<(Vec<u8>, f32)>,
    tensors: BTreeMap<String, format::TensorSaveInfo>,
}

struct MockSaveHandler<'a> {
    model: &'a Model,
}
impl format::SaveHandler<DummyError> for MockSaveHandler<'_> {
    fn write_hyperparameters(&mut self, writer: &mut dyn Write) -> Result<(), DummyError> {
        self.model.hyperparameters.write(writer).unwrap();
        Ok(())
    }

    fn tensor_data(&mut self, tensor_name: &str) -> Result<format::TensorSaveInfo, DummyError> {
        self.model
            .tensors
            .get(tensor_name)
            .cloned()
            .ok_or(DummyError)
    }
}

struct MockLoadHandler<'a> {
    data: &'a [u8],
    loaded_model: Model,
    expected_container_type: ContainerType,
}
impl format::LoadHandler<DummyError> for MockLoadHandler<'_> {
    fn container_type(&mut self, container_type: ContainerType) -> Result<(), DummyError> {
        assert_eq!(container_type, self.expected_container_type);
        Ok(())
    }

    fn vocabulary_token(&mut self, i: usize, token: Vec<u8>, score: f32) -> Result<(), DummyError> {
        assert_eq!(i, self.loaded_model.tokenizer.len());
        self.loaded_model.tokenizer.push((token, score));
        Ok(())
    }

    fn read_hyperparameters(
        &mut self,
        reader: &mut dyn BufRead,
    ) -> Result<format::PartialHyperparameters, DummyError> {
        self.loaded_model.hyperparameters = Hyperparameters::read(reader).unwrap();
        Ok(format::PartialHyperparameters {
            n_vocab: self
                .loaded_model
                .hyperparameters
                .tokenizer_size
                .try_into()
                .unwrap(),
        })
    }

    fn tensor_buffer(&mut self, info: format::TensorLoadInfo) -> Result<(), DummyError> {
        let data = format::TensorSaveInfo {
            n_dims: info.n_dims,
            dims: info.dims,
            element_type: info.element_type,
            data: info
                .read_data(&mut std::io::Cursor::new(self.data))
                .unwrap(),
        };
        self.loaded_model.tensors.insert(info.name, data);
        Ok(())
    }
}
