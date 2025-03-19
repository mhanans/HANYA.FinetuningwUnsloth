import sys
import os
import json
import PyPDF2
from docx import Document

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyPDF2."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file using python-docx."""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def generate_jsonl(input_folder, output_file):
    """Generate a JSONL file from PDFs and DOCX files in the input folder."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_name in os.listdir(input_folder):
            file_path = os.path.join(input_folder, file_name)
            try:
                if file_name.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                elif file_name.lower().endswith('.docx'):
                    text = extract_text_from_docx(file_path)
                else:
                    continue
                # Split text into paragraphs based on double newlines
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if para.strip():  # Skip empty paragraphs
                        entry = {"text": para, "source": file_name}
                        f.write(json.dumps(entry) + '\n')
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_jsonl.py <input_folder> <output_file>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    generate_jsonl(input_folder, output_file)