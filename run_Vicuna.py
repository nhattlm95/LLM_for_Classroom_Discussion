"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model models/vicuna-7b-v1.5-16k --device gpu

"""
import argparse

import torch

from fastchat.model import load_model, get_conversation_template, add_model_args
from IQA import *

def run_prompt(args, model, tokenizer, msg):
    # Build the prompt with a conversation template
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
    output_ids = model.generate(
        **inputs,
        do_sample=True if args.temperature > 1e-5 else False,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    return outputs

def main(args):
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
    )
    dialogues = load_data("data/")
    dialogues_turns = extract_turns(dialogues)
    #DS
    for s in range(0, 5):
        for conv in dialogues:
            prompt = "Based on the given dialogue between a teacher and students in a classroom, rate how well {IQA_description} on a scale of 1-4 (low-high) as follow:\n{IQA_ins}\n#Dialogue\n{Dial_content}\nRating (only specify a number between 1-4):".format(IQA_description=IQA_DES[s], IQA_ins=IQA_SCORE, Dial_content=conv.text)
            run_prompt(prompt)
            
    #DC
    for s in range(0, 5):
        for conv in dialogues:
            prompt = "How many times {IQA_description} in the given dialogue between a teacher and students in a classroom?\n#Dialogue\n{Dial_content}\nAnswer:".format(IQA_description=IQA_DES[s], Dial_content=conv.text)
            run_prompt(prompt)

    #EC
    for s in range(0, 5):
        for conv in dialogues:
            prompt = "Based on the given dialogue between a teacher and students in a classroom, provide up to 3 examples in which {IQA_description}. Answer \"Not Found\" if no example is found\n#Dialogue\n{Dial_content}\nAnswer:".format(IQA_description=IQA_DES[s], Dial_content=conv.text)
            run_prompt(prompt)

    #BC
    for s in range(0, 5):
        for conv in dialogues_turns:
            prompt = "Given a dialogue between a teacher and students in a classroom, in the last turn {IQA_description}?\n#Dialogue\n{Dial_content}\nAnswer (yes or no):".format(IQA_description=IQA_DES[s], Dial_content=conv.text)
            run_prompt(prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)