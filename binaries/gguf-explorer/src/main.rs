use std::{fmt::Display, fs::File, io::BufReader};

use egui_extras::{Column, TableBuilder};
use ggml::format::gguf::{self, Gguf};

use eframe::egui::{self, Button, CentralPanel, CollapsingHeader, Label, RichText, TopBottomPanel};

fn main() -> eframe::Result<()> {
    let file_path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("Usage: gguf-explorer <path-to-gguf-file>");
            std::process::exit(1);
        }
    };

    let mut file = File::open(file_path).expect("Failed to open file");
    let gguf = Gguf::load(&mut BufReader::new(&mut file)).expect("Failed to load gguf file");

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "GGUF Explorer",
        native_options,
        Box::new(move |_cc| {
            Box::new(Explorer {
                _file: file,
                gguf,

                selected_tab: Tab::Metadata,
                tensor_sort_order: TensorColumn::Offset,
            })
        }),
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tab {
    Metadata,
    Tensors,
}
impl Display for Tab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tab::Metadata => write!(f, "Metadata"),
            Tab::Tensors => write!(f, "Tensors"),
        }
    }
}
impl Tab {
    const ALL: [Tab; 2] = [Tab::Metadata, Tab::Tensors];
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorColumn {
    Name,
    Dimensions,
    Type,
    Offset,
}
impl Display for TensorColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorColumn::Name => write!(f, "Name"),
            TensorColumn::Dimensions => write!(f, "Dimensions"),
            TensorColumn::Type => write!(f, "Type"),
            TensorColumn::Offset => write!(f, "Offset"),
        }
    }
}
impl TensorColumn {
    const ALL: [Self; 4] = [Self::Name, Self::Dimensions, Self::Type, Self::Offset];
}

struct Explorer {
    _file: File,
    gguf: Gguf,

    selected_tab: Tab,
    tensor_sort_order: TensorColumn,
}
impl eframe::App for Explorer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                for tab in Tab::ALL.iter().copied() {
                    let text = RichText::from(tab.to_string());
                    let text = if tab == self.selected_tab {
                        text.underline()
                    } else {
                        text
                    };

                    if ui.add(Button::new(text)).clicked() {
                        self.selected_tab = tab;
                    }
                }
            });
        });

        CentralPanel::default().show(ctx, |ui| match self.selected_tab {
            Tab::Metadata => {
                self.render_metadata(ui);
            }
            Tab::Tensors => {
                self.render_tensors(ui);
            }
        });
    }
}
impl Explorer {
    fn render_metadata(&mut self, ui: &mut egui::Ui) {
        let metadata = &self.gguf.metadata;
        let mut metadata_keys = metadata.keys().collect::<Vec<_>>();
        metadata_keys.sort_by_key(|k| *k);

        TableBuilder::new(ui)
            .striped(true)
            .auto_shrink([false, true])
            .column(Column::auto().resizable(true))
            .column(Column::remainder().resizable(true))
            .header(20.0, |mut header| {
                header.col(|ui| {
                    ui.label("Key");
                });
                header.col(|ui| {
                    ui.label("Value");
                });
            })
            .body(|mut body| {
                for key in metadata_keys {
                    let value = &metadata[key];

                    body.row(30.0, |mut row| {
                        row.col(|ui| {
                            ui.add(Label::new(monospace(key)).wrap(false));
                        });
                        row.col(|ui| match value {
                            gguf::MetadataValue::Array(value) => {
                                CollapsingHeader::new(format!("array ({} elements)", value.len()))
                                    .id_source(key)
                                    .show(ui, |ui| {
                                        ui.add(
                                            Label::new(monospace(format!("{:?}", value)))
                                                .wrap(false),
                                        );
                                    });
                            }
                            value => {
                                ui.add(Label::new(monospace(format!("{:?}", value))).wrap(false));
                            }
                        });
                    });
                }
            });
    }

    fn render_tensors(&mut self, ui: &mut egui::Ui) {
        let tensors = &self.gguf.tensor_infos;
        let mut tensor_names = tensors.keys().collect::<Vec<_>>();
        match self.tensor_sort_order {
            TensorColumn::Name => tensor_names.sort_by_key(|k| *k),
            TensorColumn::Dimensions => {
                tensor_names.sort_by_key(|k| tensors[*k].dimensions.clone())
            }
            TensorColumn::Type => tensor_names.sort_by_key(|k| tensors[*k].element_type),
            TensorColumn::Offset => tensor_names.sort_by_key(|k| tensors[*k].offset),
        }

        TableBuilder::new(ui)
            .striped(true)
            .auto_shrink([false, true])
            .column(Column::remainder().resizable(true))
            .columns(Column::auto().resizable(true), 3)
            .header(20.0, |mut header| {
                for column in TensorColumn::ALL.iter().copied() {
                    header.col(|ui| {
                        let text = RichText::from(column.to_string());
                        let text = if self.tensor_sort_order == column {
                            text.underline()
                        } else {
                            text
                        };

                        if ui.add(Button::new(text).wrap(false)).clicked() {
                            self.tensor_sort_order = column;
                        }
                    });
                }
            })
            .body(|mut body| {
                for tensor_name in tensor_names {
                    let tensor = &tensors[tensor_name];

                    body.row(30.0, |mut row| {
                        row.col(|ui| {
                            ui.add(Label::new(monospace(tensor_name)).wrap(false));
                        });
                        row.col(|ui| {
                            ui.add(
                                Label::new(monospace(format!("{:?}", tensor.dimensions)))
                                    .wrap(false),
                            );
                        });
                        row.col(|ui| {
                            ui.add(
                                Label::new(monospace(tensor.element_type.to_string())).wrap(false),
                            );
                        });
                        row.col(|ui| {
                            ui.add(Label::new(monospace(tensor.offset.to_string())).wrap(false));
                        });
                    });
                }
            });
    }
}

fn monospace(text: impl Into<String>) -> RichText {
    RichText::new(text).monospace()
}
