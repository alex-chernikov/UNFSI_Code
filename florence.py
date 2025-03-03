from transformers import AutoModelForCausalLM
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoProcessor, get_scheduler)
from torch.utils.data import Dataset

from PIL import Image

import const

SEQLEN=128
train_loader,val_loader=None,None
device,model,processor = None,None,None

model_id="microsoft/Florence-2-base-ft"
FOOD_SAFETY_MODEL_PATH = const.MODELS_PATH + '/florence2'


def init_model(model_path=None):
    global device, model, processor

    torch.cuda.empty_cache()

    revision=None
    if model_path is None:
        model_path=model_id
        revision='refs/pr/6'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                 revision=revision).to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, revision='refs/pr/6')
    torch.cuda.empty_cache()


def run_example(task_prompt):
    print(task_prompt)
    task_prompt, text_input, image=task_prompt[0], task_prompt[1], task_prompt[2]
    prompt = task_prompt + text_input

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=SEQLEN,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt,
                                                      image_size=(image.width, image.height))
    return parsed_answer


class torch_dataset_from_dataframe(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data.iloc[idx,:]
        question = example['question']
        dcode = example['dcode']
        fclass = example['class']
        image_path = example['file_name']
        answer = example['gpt_reply']
        image=Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, image, answer, f'{fclass}:{dcode}:{image_path}'


def create_food_safety_dataset(train_df=None,val_df=None,batch_size=1):
    global train_loader,val_loader

    if train_df is not None:
        train_dataset = torch_dataset_from_dataframe(train_df.iloc[:,:])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=0)

    if val_df is not None:
        val_dataset = torch_dataset_from_dataframe(val_df)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=0)

    val_loader=val_loader if val_loader is not None else train_loader


def collate_fn(batch):
    questions, images, answers, aux_info = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers, aux_info


def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        i = -1
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            inputs, answers, _ = batch

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True,
                                         return_token_type_ids=False).input_ids.to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                inputs, answers,_ = batch

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True,
                                             return_token_type_ids=False).input_ids.to(device)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch + 1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)


def train_florence(bf16=False):
    init_model()
    for param in model.vision_tower.parameters():
        param.is_trainable = True

    if bf16:
        model.config.text_config.use_bfloat16 = True
        model.config.vision_config.use_bfloat16 = True

    train_model(train_loader, val_loader, model, processor, epochs=30)
    model.save_pretrained(FOOD_SAFETY_MODEL_PATH)
    processor.save_pretrained(FOOD_SAFETY_MODEL_PATH)


def predict_dataframe(df,output_path=None):
    import pandas as pd
    from os.path import basename
    from time import time

    init_model(FOOD_SAFETY_MODEL_PATH)

    aux_infos,llm_replies=[],[]

    create_food_safety_dataset(val_df=df,batch_size=6)

    t=time()

    for inputs in val_loader:

        input_ids = inputs[0]["input_ids"]
        pixel_values = inputs[0]["pixel_values"]
        aux_info = [tuple(x.split(':')) for x in inputs[2]]

        outputs = model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=SEQLEN)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)

        aux_infos.extend(list(aux_info))
        llm_replies.extend(generated_text)

    print('Florence2 prediction time:', time()-t)
    df=pd.concat([pd.DataFrame(aux_infos), pd.Series(llm_replies)], axis=1)
    df.columns=['class','dcode','file_name','reply']
    df['file_name']=df['file_name'].apply(lambda x: basename(x))

    if output_path:
        df.to_csv(const.RESULTS_PATH+"LLM/florence2_val.csv",index=None)
    return df