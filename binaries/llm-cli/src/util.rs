use std::io::Write;

pub fn process_prompt(raw_prompt: &str, prompt: &str) -> String {
    raw_prompt.replace("{{PROMPT}}", prompt)
}

pub fn print_token(t: String) {
    print!("{t}");
    std::io::stdout().flush().unwrap();
}
