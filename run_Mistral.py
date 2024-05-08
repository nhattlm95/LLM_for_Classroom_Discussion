import fire
import os
import sys
import time
import gradio as gr

import torch
from transformers import AutoTokenizer
from IQA import *


def extract_text_after_inst(text):
    # Use regular expression to find text after [/INST]
    match = re.search(r'\[/INST\](.*)', text, re.DOTALL)
    
    if match:
        # Extract and return the text after [/INST]
        result = match.group(1).strip()
        return result
    else:
        # Return None if [/INST] is not found
        return None

def inference(model, prompt):
    messages = [{"role": "user", "content": prompt}]
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)

    decoded = tokenizer.batch_decode(generated_ids)
    output =  extract_text_after_inst(decoded[0]).replace('</s>', '')
    return output

print(decoded[0])

def main():
    
    #Load model
    local_path = 'models/Mistral-7B-Instruct-v0.1'

    device = 'cuda'
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(local_path, local_files_only = True
                                                  , device_map=device
                                                  , torch_dtype=torch.float16)

    dialogues = load_data("data/")
    dialogues_turns = extract_turns(dialogues)
    #DS
    for s in range(0, 5):
        for conv in dialogues:
            prompt = "Based on the given dialogue between a teacher and students in a classroom, rate how well {IQA_description} on a scale of 1-4 (low-high) as follow:\n{IQA_ins}\n#Dialogue\n{Dial_content}\nRating (only specify a number between 1-4):".format(IQA_description=IQA_DES[s], IQA_ins=IQA_SCORE, Dial_content=conv.text)
            inference(model, prompt)
            
    #DC
    for s in range(0, 5):
        for conv in dialogues:
            prompt = "How many times {IQA_description} in the given dialogue between a teacher and students in a classroom?\n#Dialogue\n{Dial_content}\nAnswer:".format(IQA_description=IQA_DES[s], Dial_content=conv.text)
            inference(model, prompt)

    #EC
    for s in range(0, 5):
        for conv in dialogues:
            prompt = "Based on the given dialogue between a teacher and students in a classroom, provide up to 3 examples in which {IQA_description}. Answer \"Not Found\" if no example is found\n#Dialogue\n{Dial_content}\nAnswer:".format(IQA_description=IQA_DES[s], Dial_content=conv.text)
            inference(model, prompt)

    #BC
    for s in range(0, 5):
        for conv in dialogues_turns:
            prompt = "Given a dialogue between a teacher and students in a classroom, in the last turn {IQA_description}?\n#Dialogue\n{Dial_content}\nAnswer (yes or no):".format(IQA_description=IQA_DES[s], Dial_content=conv.text)
            inference(model, prompt)

if __name__ == "__main__":
    main()

