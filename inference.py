import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import argparse

from config import MODEL_NAME

def parse_opt():
    parser = argparse.ArgumentParser(description="Generate a text using finetuned transformer.")

    parser.add_argument("-mp", "--model_path", type=str, required=True,
                    help="Path to the finetuned huggingface transformer model.")
    parser.add_argument("-pr", "--prompt", type=str, required=True,
                    help="Input prompt.")

    return parser.parse_args()

if __name__ == '__main__':
    # args
    args = parse_opt()

    prompt_text = args.prompt
    model_path = args.model_path

    # Load the finetuned model and tokenizer
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # Place the model in evaluation mode
    model.eval()

    # Prepare the prompt text
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

    # Generate text predictions
    # Adjust the parameters like max_length, num_return_sequences as needed
    outputs = model.generate(
        input_ids, 
        max_length = 100,
        num_return_sequences = 1,
        temperature = 0.7,
        top_k = 50,
        top_p = 0.95,
        repetition_penalty = 1.2,
        do_sample = True,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens = True)

    print("Generated text:-")
    print(generated_text)
