import torch
import logging
import os

from tinydb import TinyDB, Query
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.dataset import template
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# DB setup
db = TinyDB('db.json')
Req = Query()

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

with open('data/actors.txt', 'r') as f:
    ACTORS = [i.replace('\n', '') for i in f.readlines()]

HELP_MESSAGE = '''Model will generate answer in respose to messages

/start -- restart/clean dialogue
/actors -- giving list of actors available for /set command
/set <search request> -- setting model to answer with give actor
/unset -- removing actor restriction of model'''


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    username = user.name
    if len(db.search(Req.user == username)) == 0:
        db.insert({'user': username, 'dialog': [], 'prefix': ''})
        await update.message.reply_html(
            rf"Hi {user.mention_html()}! Type /help for additional information",
        )
    else:
        db.update({'dialog': []}, Req.user == username)
        await update.message.reply_text("Dialogue was cleaned")


async def generate_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    new_text = update.message.text
    username = update.message.from_user.name
    record = db.search(Req.user == username)
    if len(record) > 0:
        dialog = record[0]['dialog']
        actor = record[0]['prefix']
        if len(actor) > 0:
            answer_start = f"[{actor}]: "
        else:
            answer_start = ''

        if new_text:
            dialog.append(new_text)
        db.update({'dialog': dialog}, Req.user == username)
    else:
        answer_start = ''
        dialog = [new_text] if new_text else []
        db.insert({'user': username, 'dialog': dialog, 'prefix': ''})

    query = template(dialog, model_type) + ' [|Assistant|] ' + answer_start
    model_inputs = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_len = len(model_inputs[0])
    generated_ids = model.generate(
        input_ids=model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,                        
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.15)

    output = answer_start + tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    dialog.append(output)
    db.update({'dialog': dialog}, Req.user == username)
    await update.message.reply_text(output)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_MESSAGE)


async def set_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    username = update.message.from_user.name
    message_parts = update.message.text.split(' ')
    if len(message_parts) <= 1:
        await update.message.reply_text("After set command you should specify name of actor")
        return
    else:
        search_str = ' '.join(message_parts[1:]).lower()
        for i in ACTORS:
            if search_str in i.lower():
                db.update({'prefix': i}, Req.user == username)
                await update.message.reply_text(f"Setting actor: [{i}]")
                return
        await update.message.reply_text(f"Nothing found by request: '{search_str}'")


async def unset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    username = update.message.from_user.name
    db.update({'prefix': ''}, Req.user == username)
    await update.message.reply_text("Unsetting actor")


async def set_list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('\n'.join([f'{i}' for i in ACTORS]))


def main() -> None:
    token = os.environ.get('TG_SECRET')
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("set", set_command))
    application.add_handler(CommandHandler("unset", unset_command))
    application.add_handler(CommandHandler("actors", set_list_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_answer))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()