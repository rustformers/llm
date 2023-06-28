fn main() {
    // Ensure that these match `.github/workflows/rust.yml`.
    cmd("cargo", &["check"]);
    cmd("cargo", &["test", "--all"]);
    cmd("cargo", &["fmt", "--check", "--all"]);
    cmd("cargo", &["doc", "--workspace", "--exclude", "llm-cli"]);
    cmd("cargo", &["clippy", "--", "-Dclippy::all"]);
}

fn cmd(cmd: &str, args: &[&str]) {
    println!("=== Running command: {cmd} {args:?}");
    let mut child = std::process::Command::new(cmd).args(args).spawn().unwrap();
    if !child.wait().unwrap().success() {
        panic!("Failed to run command: {} {:?}", cmd, args);
    }
}
