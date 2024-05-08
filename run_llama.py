# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import gradio as gr

import torch
from transformers import AutoTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model

from accelerate.utils import is_xpu_available
from IQA import *

def main(
    model_name='model/Llama-2-7b-hf',
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =300, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=7.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool=False,
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    
    #Load model
    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    def inference(prompt):
        safety_checker = get_safety_checker(enable_azure_content_safety,
                                            enable_sensitive_topics,
                                            enable_salesforce_content_safety,
                                            enable_llamaguard_content_safety
                                            )

        # Safety check of the user prompt
        safety_results = [check(user_prompt) for check in safety_checker]
        are_safe = all([r[1] for r in safety_results])
        if are_safe:
            print("User prompt deemed safe.")
            print(f"User prompt:\n{user_prompt}")
        else:
            print("User prompt deemed unsafe.")
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
            print("Skipping the inference as the prompt is not safe.")
            sys.exit(1)  # Exit the program with an error status

        # Set the seeds for reproducibility
        if is_xpu_available():
            torch.xpu.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
        # e2e_inference_time = (time.perf_counter()-start)*1000
        # print(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Safety check of the model output
        # safety_results = [check(output_text, agent_type=AgentType.AGENT, user_prompt=user_prompt) for check in safety_checker]
        # are_safe = all([r[1] for r in safety_results])
        # if are_safe:
        #     print("User input and model output deemed safe.")
        #     print(f"Model output:\n{output_text}")
        # else:
        #     print("Model output deemed unsafe.")
        #     for method, is_safe, report in safety_results:
        #         if not is_safe:
        #             print(method)
        #             print(report)
        return output_text

    dialogues = load_data("data/")
    dialogues_turns = extract_turns(dialogues)
    #DS
    for s in range(0, 5):
        for conv in dialogues:
            prompt = "Based on the given dialogue between a teacher and students in a classroom, rate how well {IQA_description} on a scale of 1-4 (low-high) as follow:\n{IQA_ins}\n#Dialogue\n{Dial_content}\nRating (only specify a number between 1-4):".format(IQA_description=IQA_DES[s], IQA_ins=IQA_SCORE, Dial_content=conv.text)
            inference(prompt)
            
    #DC
    for s in range(0, 5):
        for conv in dialogues:
            prompt = "How many times {IQA_description} in the given dialogue between a teacher and students in a classroom?\n#Dialogue\n{Dial_content}\nAnswer:".format(IQA_description=IQA_DES[s], Dial_content=conv.text)
            inference(prompt)

    #EC
    for s in range(0, 5):
        for conv in dialogues:
            prompt = "Based on the given dialogue between a teacher and students in a classroom, provide up to 3 examples in which {IQA_description}. Answer \"Not Found\" if no example is found\n#Dialogue\n{Dial_content}\nAnswer:".format(IQA_description=IQA_DES[s], Dial_content=conv.text)
            inference(prompt)

    #BC
    for s in range(0, 5):
        for conv in dialogues_turns:
            prompt = "Given a dialogue between a teacher and students in a classroom, in the last turn {IQA_description}?\n#Dialogue\n{Dial_content}\nAnswer (yes or no):".format(IQA_description=IQA_DES[s], Dial_content=conv.text)
            inference(prompt)



if __name__ == "__main__":
    main()

