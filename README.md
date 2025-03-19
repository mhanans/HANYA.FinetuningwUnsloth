# HANYAFT (HANYA Fine-Tuning)

**HANYAFT** is a modular, automated pipeline for fine-tuning large language models (LLMs) using documents (PDFs and DOCX files). It extracts text from documents, generates a JSONL dataset, fine-tunes a model using [Unsloth](https://github.com/unslothai/unsloth) and [LoRA](https://arxiv.org/abs/2106.09685), and provides a Gradio interface to test the fine-tuned model.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Prepare Documents](#step-1-prepare-documents)
  - [Step 2: Run the Setup Script](#step-2-run-the-setup-script)
  - [Step 3: Interact with the Gradio Interface](#step-3-interact-with-the-gradio-interface)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Overview
HANYAFT simplifies the process of fine-tuning a language model on custom document data. It consists of three main components:
1. **Document Parsing and JSONL Generation**: Extracts text from PDFs and DOCX files and creates a JSONL file.
2. **Model Fine-Tuning**: Uses Unsloth and LoRA to efficiently fine-tune a pre-trained model on the generated dataset.
3. **Testing with Gradio**: Launches a web interface to interact with the fine-tuned model.

The application is designed to be modular, with each component in its own Python file, making it easy to modify and extend.

## Features
- **Automated Setup**: A single shell script handles environment setup, dependency installation, and execution.
- **Efficient Fine-Tuning**: Leverages Unsloth and LoRA for fast, memory-efficient training.
- **Document Support**: Parses PDFs and DOCX files using `PyPDF2` and `python-docx`.
- **Interactive Testing**: Includes a Gradio interface for testing the model with custom inputs.
- **Modular Design**: Separate scripts for document parsing, fine-tuning, and testing.

## Prerequisites
- **Operating System**: Linux (tested on Ubuntu)
- **Hardware**: CUDA-compatible GPU (e.g., NVIDIA RTX series)
- **Software**:
  - Bash shell
  - Python 3.10
  - Miniconda (installed automatically by the script)
- **Documents**: PDFs and/or DOCX files in a folder (default: `./dataset-doc`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mhanans/HANYA.FinetuningwUnsloth.git
   cd HANYA.FinetuningwUnsloth

2. **Just Run**:
   ```bash
   bash run.sh

## Usage

1. **Step 1: Prepare Documents**
Place your PDF and DOCX files in the ./documents folder (relative to the script’s location).
If you want to use a different folder, specify it when running the script (see below).
2. **Step 2: Run the Setup Script**
To use the default ./documents folder:
    ```bash
    ./setup.sh
To specify a custom folder:
    ```bash
    ./setup.sh /path/to/your/documents

The script will:
- Create a Conda environment with Python 3.10.
- Install dependencies (e.g., PyTorch, Unsloth, Gradio).
- Generate a JSONL file (training_data.jsonl) from your documents.
- Fine-tune the model and save it to my-finetuned-model.
- Launch a Gradio interface to test the model.

3. **Step 3: Interact with the Gradio Interface**
After the script runs, a URL (e.g., http://127.0.0.1:7860) will appear in the terminal.
Open this URL in your browser.
Enter a prefix (e.g., "The quick brown fox"), and the model will generate a continuation.
To stop the interface, press Ctrl+C in the terminal.
Project Structure:
- setup.sh: Main script that sets up the environment, installs dependencies, and runs the pipeline.
- generate_jsonl.py: Extracts text from documents and creates a JSONL file.
- finetune_model.py: Fine-tunes the model using the JSONL data.
- launch_gradio.py: Launches a Gradio interface to test the fine-tuned model.

documents/: Default folder for input documents (create this and add your files).
training_data.jsonl: Generated JSONL file (output of generate_jsonl.py).
my-finetuned-model/: Directory where the fine-tuned model is saved.

## Customization

Change the Model: Edit finetune_model.py to use a different pre-trained model (e.g., unsloth/Mixtral-7B).
Adjust Training Parameters: Modify max_steps, learning_rate, etc., in finetune_model.py.
Document Parsing: Update generate_jsonl.py to support other formats or extraction methods.
Gradio Interface: Customize input/output fields in launch_gradio.py.

## Troubleshooting
Import Errors: If you see ImportError, ensure all dependencies are installed. Check setup.sh for missing packages.
GPU Issues: Confirm your GPU drivers are updated and nvidia-smi works.
Document Parsing Failures: If files fail to process, verify their format or try libraries like pdfminer.six for PDFs.
Training Memory Errors: Lower per_device_train_batch_size or max_seq_length in finetune_model.py.
License
This project is licensed under the MIT License. See the LICENSE file for details.