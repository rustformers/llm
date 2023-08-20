use std::io::BufReader;

use ggml::format::gguf;

fn main() -> anyhow::Result<()> {
    let mut file = BufReader::new(std::fs::File::open(
        std::env::args().nth(1).expect("need a file to read"),
    )?);

    let gguf = gguf::Gguf::load(&mut file)?;
    dbg!(gguf);

    Ok(())
}
