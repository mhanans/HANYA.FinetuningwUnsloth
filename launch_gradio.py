import sys
import gradio as gr
from unsloth import FastLanguageModel

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python launch_gradio.py <model_dir>")
        sys.exit(1)
    model_dir = sys.argv[1]

    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(model_dir)
    FastLanguageModel.for_inference(model)

    def generate_text(prefix):
        """Generate text based on the input prefix."""
        inputs = tokenizer(prefix, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Set up Gradio interface
    interface = gr.Interface(
        fn=generate_text,
        inputs=gr.Textbox(label="Prefix", placeholder="Enter the starting text here"),
        outputs=gr.Textbox(label="Generated Text"),
        title="Test My Fine-Tuned Model",
        description="Enter a prefix, and the model will generate a continuation."
    )

    # Launch the interface
    interface.launch()