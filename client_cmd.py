import torch
import peft
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.dataset import template

# Model setup
model_type = 'phi'
device = 'cuda'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_type = 'phi'
if model_type == 'phi':
    model_path = 'xxond/disco-limbic-dialogue'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        flash_attn=True,
        flash_rotary=True,
        fused_dense=True,
        trust_remote_code=True,
        device_map={'': 0},
        quantization_config=bnb_config
    )
elif model_type == 'minichat':
    model_path = 'model/disco-limbic-dialogue-512'
    model = AutoModelForCausalLM.from_pretrained(
        model_path, use_cache=True,
        device_map="auto",
        quantization_config=bnb_config,
    )
else:
    raise NotImplementedError

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Moving setting eos and pad to generation to remove UserWarning
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.eos_token_id = tokenizer.eos_token_id

model = model.eval()

dialog = [
]

while True:
    inp = input('[You]: ')
    if inp == 'q':
        break
    if inp:
        dialog.append(inp)

    query = template(dialog, model_type) + ' [|Assistant|] '
    
    model_inputs = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_len = len(model_inputs[0])
    generated_ids = model.generate(input_ids=model_inputs, max_new_tokens=128,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                temperature=0.7,
                                repetition_penalty=1.15)
    output = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    dialog.append(output)
    print(output, end='\n\n')
    time.sleep(0.5)
