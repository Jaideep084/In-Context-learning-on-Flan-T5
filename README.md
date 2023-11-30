# In-Context-learning-on-Flan-T5
## Overview
This repository contains code for in-context learning using the Flan-T5 model. The code demonstrates how to use Flan-T5 for various summarization tasks, including zero-shot, one-shot, and few-shot inference on a conversational dataset.
## Prerequisites
Make sure you have the necessary Python packages installed. You can install them using the following command:
```bash
pip install transformers datasets
```
## Code Structure
#### Import necessary libraries:
```python
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BertGenerationConfig
```
#### Load the conversational dataset:
```python
dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(dataset_name)
```
#### Define example indices for demonstration:
```python
example_indices = [40, 200]
```
#### Initialize the Flan-T5 model and tokenizer:
```python
model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```
