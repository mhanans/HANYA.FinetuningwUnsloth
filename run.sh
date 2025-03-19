#!/bin/bash

# Setup script to generate JSONL, fine-tune a model, and test with Gradio

# Directories
INSTALL_DIR="$(pwd)/install_dir"
CONDA_ROOT="$INSTALL_DIR/conda"
ENV_DIR="$INSTALL_DIR/env"
PYTHON_VERSION="3.10"

# Utility Functions

function check_path_for_spaces() {
    if [[ $PWD =~ \  ]]; then
        echo "The current workdir has whitespace which can lead to unintended behaviour. Please modify your path and continue later."
        exit 1
    fi
}

function check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: NVIDIA GPU not detected. This script requires a CUDA-compatible GPU to works normally."
    fi
    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)
    if [ "$gpu_count" -eq 0 ]; then
        echo "Warning: No NVIDIA GPU detected. Please ensure your GPU drivers are installed."
    fi
    echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
}

function install_miniconda() {
    local sys_arch=$(uname -m)
    case "${sys_arch}" in
    x86_64*) sys_arch="x86_64" ;;
    arm64*) sys_arch="aarch64" ;;
    aarch64*) sys_arch="aarch64" ;;
    *) {
        echo "Unknown system architecture: ${sys_arch}! This script runs only on x86_64 or arm64"
        exit 1
    } ;;
    esac

    if ! "${CONDA_ROOT}/bin/conda" --version &>/dev/null; then
        local miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${sys_arch}.sh"
        mkdir -p "$INSTALL_DIR"
        curl -Lk "$miniconda_url" >"$INSTALL_DIR/miniconda_installer.sh"
        chmod u+x "$INSTALL_DIR/miniconda_installer.sh"
        bash "$INSTALL_DIR/miniconda_installer.sh" -b -p "$CONDA_ROOT"
        rm -rf "$INSTALL_DIR/miniconda_installer.sh"
    fi
    echo "Miniconda is installed at $CONDA_ROOT"
}

function create_conda_env() {
    if [ ! -d "$ENV_DIR" ]; then
        echo "Creating conda environment with python=$PYTHON_VERSION in $ENV_DIR"
        "${CONDA_ROOT}/bin/conda" create -y -k --prefix "$ENV_DIR" python="$PYTHON_VERSION" || {
            echo "Failed to create conda environment."
            rm -rf "$ENV_DIR"
            exit 1
        }
    else
        echo "Conda environment exists at $ENV_DIR"
    fi
}

function activate_conda_env() {
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$ENV_DIR" || {
        echo "Failed to activate environment. Please remove $ENV_DIR and run the script again."
        exit 1
    }
    echo "Activated conda environment at $CONDA_PREFIX"
}

function deactivate_conda_env() {
    if [ "$CONDA_PREFIX" == "$ENV_DIR" ]; then
        conda deactivate
        echo "Deactivated conda environment at $ENV_DIR"
    fi
}

function install_dependencies() {
    echo "Installing PyTorch with CUDA support..."
    conda install -y pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia || {
        echo "Failed to install PyTorch."
        exit 1
    }

    echo "Installing Unsloth from GitHub..."
    pip install "unsloth[local] @ git+https://github.com/unslothai/unsloth.git" || {
        echo "Failed to install Unsloth."
        exit 1
    }

    echo "Installing additional dependencies..."
    pip install transformers datasets gradio trl peft accelerate bitsandbytes triton docling PyPDF2 || {
        echo "Failed to install additional dependencies."
        exit 1
    }
}

function print_highlight() {
    local message="${1}"
    echo "" && echo "******************************************************"
    echo "$message"
    echo "******************************************************" && echo ""
}

# Main Execution

SCRIPT_DIR=$(dirname "$0")

# Check for too many arguments
if [ $# -gt 1 ]; then
    echo "Usage: $0 [documents_folder]"
    echo "If no folder is provided, it will use the default: $SCRIPT_DIR/dataset"
    exit 1
fi

# Set DOCUMENTS_FOLDER: default if no argument, otherwise use provided path
if [ $# -eq 0 ]; then
    DOCUMENTS_FOLDER="$SCRIPT_DIR/dataset"
else
    DOCUMENTS_FOLDER="$1"
fi

# Verify the folder exists
if [ ! -d "$DOCUMENTS_FOLDER" ]; then
    echo "Documents folder not found: $DOCUMENTS_FOLDER"
    echo "Please provide a valid path or create the default folder."
    exit 1
fi

echo "Using documents folder: $DOCUMENTS_FOLDER"

# Existing variables (adjust as needed)
JSONL_FILE="training_data.jsonl"
FINETUNED_MODEL_DIR="my-finetuned-model"

print_highlight "Checking prerequisites"
check_path_for_spaces
check_gpu

print_highlight "Setting up Miniconda"
install_miniconda

print_highlight "Creating Conda environment"
create_conda_env
activate_conda_env

print_highlight "Installing dependencies"
install_dependencies

print_highlight "Generating JSONL from documents"
python generate_jsonl.py "$DOCUMENTS_FOLDER" "$JSONL_FILE" || {
    echo "Failed to generate JSONL."
    exit 1
}

print_highlight "Fine-tuning the model"
python finetune_model.py "$JSONL_FILE" "$FINETUNED_MODEL_DIR" || {
    echo "Fine-tuning failed."
    exit 1
}

print_highlight "Launching Grado interface"
python launch_gradio.py "$FINETUNED_MODEL_DIR" || {
    echo "Failed to launch Gradio interface."
    exit 1
}

print_highlight "Cleaning up"
deactivate_conda_env

echo "Setup and execution completed. Press enter to exit."
read -p ""