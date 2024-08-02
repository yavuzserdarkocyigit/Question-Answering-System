# Question Answering System

Developed a question answering system using the BERT base model and the SQuAD dataset. The system is capable of understanding and providing accurate answers to questions based on a given context.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)

## Installation

1. Clone the repo:
    ```sh
    git clone https://github.com/yavuzserdarkocyigit/Question-Answering-System.git
    ```
2. Install required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To use the question answering system, follow these steps:

1. Open the Jupyter Notebook:
    ```sh
    jupyter notebook question_answering_last.ipynb
    ```
2. Follow the instructions in the notebook to preprocess the data, train the model, and evaluate its performance.

## Model Training and Evaluation

The system uses the BERT base model and the SQuAD dataset for training and evaluation. The main steps involved are:

1. **Data Preprocessing**: Tokenizing the input data and preparing it for training.
2. **Model Training**: Fine-tuning the BERT model on the SQuAD dataset.
3. **Evaluation**: Assessing the model's performance on the validation set using accuracy metrics.

Example code snippets:

```python
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, SequentialSampler

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Tokenize input
inputs = tokenizer(question, context, return_tensors='pt')

# Get model predictions
outputs = model(**inputs)
