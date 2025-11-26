{
  description = "Diabetes Data Mining Pipeline Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python environment with poetry
        pythonEnv = pkgs.python311;

        # The diabetes dataset URL
        datasetUrl = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv";
        datasetPath = "./data/raw/diabetes.csv";
      in
      {
        # nix run . -- <command> [args]
        apps.default = {
          type = "app";
          program = toString (
            pkgs.writeShellScript "diabetes-pipeline" ''
              export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
              ${pkgs.poetry}/bin/poetry run diabetes-pipeline "$@"
            ''
          );
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python311
            poetry
            curl
            stdenv.cc.cc.lib
            zlib
          ];

          shellHook = ''
            echo "==============================================="
            echo "Diabetes Pipeline Dev Environment aktiviert"
            echo "   Python Version: $(python --version)"
            echo "   Poetry Version: $(poetry --version)"
            echo "==============================================="

            # Fix für Bibliotheken, die .so Files (C-Extensions) laden müssen
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

            # Damit Poetry die venv im Projektordner erstellt
            export POETRY_VIRTUALENVS_IN_PROJECT=true

            # Check and fetch dataset if not present
            if [ ! -f "${datasetPath}" ]; then
              echo "Dataset not found. Downloading from GitHub..."
              mkdir -p data/raw
              curl -sL "${datasetUrl}" -o "${datasetPath}"
              echo "Dataset downloaded to ${datasetPath}"
            else
              echo "Dataset already present at ${datasetPath}"
            fi
          '';
        };
      }
    );
}
