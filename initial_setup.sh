#!/bin/bash

# Setup script to generate JSONL from documents, fine-tune a model, and test with Gradio

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
        echo "NVIDIA GPU not detected. This script requires a CUDA-compatible GPU."
        exit 1
    fi
    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)
    if [ "$gpu_count" -eq 0 ]; then
        echo "No NVIDIA GPU detected. Please ensure your GPU drivers are installed."
        exit 1
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
    pip install transformers datasets gradio trl peft accelerate bitsandbytes triton docling || {
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

if [ $# -ne 1 ]; then
    echo "Usage: $0 <documents_folder>"
    echo "Example: $0 /path/to/documents"
    exit 1
fi

DOCUMENTS_FOLDER="$1"
JSONL_FILE="training_data.jsonl"
FINETUNED_MODEL_DIR="my-finetuned-model"

# Temporary Python scripts
GENERATE_SCRIPT="/tmp/generate_jsonl_$(date +%s).py"
FINETUNE_SCRIPT="/tmp/finetune_model_$(date +%s).py"
GRADI_APP_SCRIPT="/tmp/launch_gradio_$(date +%s).py"

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
cat << EOF > "$GENERATE_SCRIPT"
import os
import json
from docling import parse_document

def extract_paragraphs(json_data):
    paragraphs = []
    for section in json_data.get("content", []):
        if section.get("type") == "section":
            paragraphs.extend(section.get("paragraphs", []))
    return paragraphs

def generate_jsonl(input_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_name in os.listdir(input_folder):
            if file_name.lower().endswith(('.pdf', '.docx')):
                file_path = os.path.join(input_folder, file_name)
                try:
                    json_data = parse_document(file_path)
                    paragraphs = extract_paragraphs(json_data)
                    for para in paragraphs:
                        if para.strip():
                            entry = {"text": para, "source": file_name}
                            f.write(json.dumps(entry) + '\n')
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    generate_jsonl("$DOCUMENTS_FOLDER", "$JSONL_FILE")
EOF
python "$GENERATE_SCRIPT" || {
    echo "Failed to generate JSONL."
    exit 1
}
rm "$GENERATE_SCRIPT"

print_highlight "Fine-tuning the model"
cat << EOF > "$FINETUNE_SCRIPT"
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Load dataset
dataset = load_dataset("json", data_files="$JSONL_FILE", split="train")

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Set up trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=10,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="outputs",
        seed=3407,
    ),
)

# Train
trainer.train()

# Save model
model.save_pretrained("$FINETUNED_MODEL_DIR")
EOF
python "$FINETUNE_SCRIPT" || {
    echo "Fine-tuning failed."
    exit 1
}
rm "$FINETUNE_SCRIPT"

print_highlight "Launching Gradio interface"
cat << EOF > "$GRADI_APP_SCRIPT"
import gradio as gr
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained("$FINETUNED_MODEL_DIR")
FastLanguageModel.for_inference(model)

def generate_text(prefix):
    inputs = tokenizer(prefix, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

interface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(label="Prefix", placeholder="Enter the starting text here"),
    outputs=gr.Textbox(label="Generated Text"),
    title="Test My Fine-Tuned Model",
    description="Enter a prefix, and the model will generate a continuation."
)

interface.launch()
EOF
python "$GRADI_APP_SCRIPT" || {
    echo "Failed to launch Gradio interface."
    exit 1
}
rm "$GRADI_APP_SCRIPT"

print_highlight "Cleaning up"
deactivate_conda_env

echo "Setup and execution completed. Press enter to exit."
read -p ""