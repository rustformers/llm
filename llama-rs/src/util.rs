use std::path::{Path, PathBuf};

use crate::LoadError;

pub fn find_all_model_files(main_path: &Path) -> Result<Vec<PathBuf>, LoadError> {
    Ok(collect_related_paths(
        main_path,
        std::fs::read_dir(main_path.parent().ok_or_else(|| LoadError::NoParentPath {
            path: main_path.to_owned(),
        })?)?
        .filter_map(Result::ok)
        .map(|de| de.path()),
    ))
}

fn collect_related_paths(
    main_path: &Path,
    directory_paths: impl Iterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    let main_filename = main_path.file_name().and_then(|p| p.to_str());

    let mut paths: Vec<PathBuf> = directory_paths
        .filter(|p| {
            p.file_name()
                .and_then(|p| p.to_str())
                .zip(main_filename)
                .map(|(part_filename, main_filename)| {
                    match part_filename.strip_prefix(main_filename) {
                        Some(suffix) => {
                            suffix.is_empty()
                                || (suffix
                                    .strip_prefix('.')
                                    .map(|s| s.parse::<usize>().is_ok())
                                    .unwrap_or(false))
                        }
                        None => false,
                    }
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_related_paths() {
        let main_path = PathBuf::from("/models/llama.bin");
        let directory_paths = [
            "/models/llama.bin",
            "/models/llama.bin.1",
            "/models/llama.bin.2",
            "/models/llama.bin.tmp",
        ]
        .map(PathBuf::from);
        let expected_paths = [
            "/models/llama.bin",
            "/models/llama.bin.1",
            "/models/llama.bin.2",
        ]
        .map(PathBuf::from);

        let output_paths = collect_related_paths(&main_path, directory_paths.into_iter());
        assert_eq!(expected_paths.as_slice(), output_paths);
    }
}
