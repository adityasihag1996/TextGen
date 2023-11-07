# Project: Fine-Tuning GPT-Neo 125M on Toy Dataset

## Overview
This project aims to fine-tune the GPT-Neo 125M model, a smaller variant of the GPT-Neo family designed for causal language modeling tasks.\
I used it to fine-tune on a toy dataset provided in the data folder, which consists of Eminem Lyrics data.\
The primary purpose of this project is experimental learning and to familiarize users with the fine-tuning process using the Hugging Face Transformers library.\
The project utilizes the GPT-Neo 125M model from the EleutherAI/gpt-neo-125M configuration. This model is a transformer-based neural network designed for natural language understanding and generation tasks.\
**_It is not intended to surpass state-of-the-art models or achieve top accuracy metrics._**

## Dataset
The toy dataset, comprised of Eminem Lyrics, can be found within the `/data` folder. This dataset is a minimal set used for demonstration purposes and to enable quick experimentation.\
**_Dataset Credits :-_** https://www.kaggle.com/datasets/aditya2803/eminem-lyrics/data

## Setup
To set up the project for training and inference, follow the steps outlined below:

### Environment Setup:

Ensure Python 3.6+ is installed on your system.
Install required Python packages using pip:
```
pip install -r requirements.txt
```

### Data Preparation:

Place your training data inside the data directory.
Ensure the data is formatted properly for consumption by the model.

#### Training:

The script `train.py` is provided for fine-tuning the model.
Adjust hyperparameters within the script as needed for your specific training setup.
To initiate training, run the script:
```
python train.py
```

#### Inference:

The script inference.py can be used to generate text using the fine-tuned model.
Execute the script to test the outputs of your model:
```
python inference.py --model_path "/path/to/your/model" --prompt "My name is"
```

Here is a sample inference using the prompt "my name is"
```
my name is marsh and this world im out of it  
cause withallthis abc shitim starting to sound like alfa bit 
ha i kill me this medicines counterfeit 
i was mislead by the sound of it how am
```

## Disclaimer
This project is for educational and experimental purposes only. The fine-tuning and inference scripts are provided to demonstrate the usage of the Hugging Face Transformers library and should be adapted for any rigorous training or research endeavors.

## Support
For any queries or issues related to this project, please open an issue on the project's GitHub repository.
