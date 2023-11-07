import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from config import MODEL_NAME, DATA_PATH
from dataset import MyDataset


def runner(model_name, sentences):
    # Load the tokenizer and the model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the text
    inputs = tokenizer(sentences, padding = "max_length", truncation = True, max_length = 256, return_tensors = "pt")

    # Need to clone the input_ids to use as labels for language modeling
    inputs['labels'] = inputs.input_ids.detach().clone()

    dataset = MyDataset(inputs)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./gpt-neo-finetuned",   # specify where checkpoints will be saved
        overwrite_output_dir = True,           # overwrite the content of the output directory
        num_train_epochs = 10,                  # number of training epochs
        per_device_train_batch_size = 2,       # batch size for training
        per_device_eval_batch_size = 2,        # batch size for evaluation
        eval_steps = 100,                      # perform evaluation every X steps
        save_steps = 200,                      # save checkpoint every X steps
        warmup_steps = 500,                    # number of warmup steps for learning rate scheduler
    )

    # Initialize the Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        eval_dataset = dataset,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        texts = file.read()
    texts = texts.split("\t")

    runner(MODEL_NAME)

    
