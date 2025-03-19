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
- [License](#license)

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
   cd HANYAFT