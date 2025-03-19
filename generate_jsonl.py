import sys
import os
import json
from docling import parse_document

def extract_paragraphs(json_data):
    """
    Extract paragraphs from the JSON structure returned by Docling.
    Adjust based on actual Docling output structure if needed.
    """
    paragraphs = []
    for section in json_data.get("content", []):
        if section.get("type") == "section":
            paragraphs.extend(section.get("paragraphs", []))
    return paragraphs

def generate_jsonl(input_folder, output_file):
    """Generate a JSONL file from PDFs and DOCX files in the input folder."""
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
    if len(sys.argv) != 3:
        print("Usage: python generate_jsonl.py <input_folder> <output_file>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    generate_jsonl(input_folder, output_file)