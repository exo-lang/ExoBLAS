{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.ninja
    pkgs.gbenchmark
    pkgs.cmake
    pkgs.pkg-config

    #feel free to comment this out depending on which line you use...
    pkgs.openblas
    pkgs.darwin.apple_sdk.frameworks.Accelerate

    # keep this line if you use bash
    pkgs.bashInteractive
  ];

  # edit this with the package of your venv
  shellHook = ''
    source .venv/bin/activate
  '';
}
