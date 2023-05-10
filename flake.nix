{
  description = "Rust-based tool for inference of LLMs.";
  inputs = {
    nixpkgs.url = github:nixos/nixpkgs/nixpkgs-unstable;
    naersk.url = github:nix-community/naersk;
    flake-utils.url = github:numtide/flake-utils;
  };

  outputs = { self, nixpkgs, naersk, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        naersk' = pkgs.callPackage naersk { };
        llm = naersk'.buildPackage {
          src = ./.;
        };
      in
      {
        formatter = pkgs.nixpkgs-fmt;
        packages.default = llm;
        apps.default = {
          type = "app";
          program = "${llm}/bin/llm";
        };
        devShells.default = with pkgs; mkShell {
          packages = [ cargo rustc rust-analyzer rustfmt cmake ];
          RUST_SRC_PATH = rustPlatform.rustLibSrc;
        };
      }
    );
}
