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
#### Display input dialogues and baseline human summaries:
```python
for i, index in enumerate(example_indices):
    # Display dialogue and summary information
```
#### Perform model generation without prompt:
```python
for i, index in enumerate(example_indices):
    # Generate summaries without using a prompt
```
#### Perform zero-shot inference:
```python
for i, index in enumerate(example_indices):
    # Generate summaries using zero-shot inference
```
#### Define functions for one-shot and few-shot inference:
```python
def make_prompt(example_indices_full, example_index_to_summarize):
    # Generate a prompt for one-shot and few-shot inference

example_indices_full = [40]
example_index_to_summarize = 200
one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)
```
#### Perform one-shot inference
#### Perform few-shot inference
#### Display baseline human summary and model-generated summary

## Acknowledgments
<br>This code utilizes the Flan-T5 model developed by Google.
<br>The conversational dataset used in this example is sourced from the "knkarthick/dialogsum" dataset.
