import os
from tqdm import tqdm
import torch
import random
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .lm_steer_helper import RunningMean
from .generate_lm_steer_hparam import LmSteerHyperParams

def generate_lm_steer_delta(hparams:LmSteerHyperParams, dataset, model = None):
    from ...models.get_model import get_model
    from ...datasets import load_lm_steer_dataset
    
    
    del_model = True
    device = hparams.device
    if model is None:
        model, tokenizer = get_model(hparams)  # load model
    else:
        model, tokenizer = model, model.tokenizer
        model.hparams = hparams
        del_model = False
        
    train_data=load_lm_steer_dataset(dataset, hparams.subset, tokenizer, hparams.system_prompt, hparams.use_chat_template)
    dataloader = DataLoader(
        train_data, batch_size=hparams.batch_size,
        shuffle=True)  #
    data_iter = iter(dataloader)
    
    model.replace_final_layer(hparams)  # Ensure this is below the data reshuffle
    model.model.to(device)  

    # TRAIN, generate delta
    n_steps = hparams.n_steps
    if n_steps is None or n_steps == -1:
        n_steps = len(train_data)
    print("number of training steps:", n_steps)
    start_step = 0
    
    
    if hparams.save_vectors is True:
        save_dir = os.path.join(hparams.steer_vector_output_dir, 'lm_steer_vector')
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = os.path.join(save_dir, "lm_steer_vector.pt")
        
        if os.path.exists(ckpt_name):
            ckpt = torch.load(ckpt_name,map_location=device)
            model.steer.load_state_dict(ckpt[1])  #load steer
            start_step = ckpt[2]
            print(f"resume training from {start_step}")
            
    if hparams.optimizer == "Adam":
        optimizer = Adam(model.steer.parameters(), lr=hparams.lr)  

    pbar = tqdm(range(start_step, n_steps))
    loss_mean = RunningMean(hparams.gamma_mean)  
    scaler = torch.cuda.amp.GradScaler()

    for step_i in pbar:
        batch = next(data_iter, None)
        if batch is None:
            data_iter = iter(dataloader)
            batch = next(data_iter, None)

      
        cur_batch_size = len(batch["text"])
        batch_stance = torch.zeros(cur_batch_size, hparams.num_steers).to(device)  
        batch_stance[:, hparams.training_steer] = torch.Tensor(batch["label"]).to(device) 
        if hparams.dummy_steer is not None:  
            batch_stance[:, hparams.dummy_steer] = 1

        batch_text = batch["text"]
        tokenized = tokenizer(batch_text, padding=True,
                        max_length=hparams.max_length, truncation=True, add_special_tokens=not hparams.use_chat_template)
        input_ids = torch.LongTensor(tokenized["input_ids"]).to(device)
        attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(device)
        optimizer.zero_grad()

        if hparams.low_resource_mode:  
            with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16):
                model.steer.set_value(steer_values=batch_stance.float())  #set steer value
                loss = model.model(input_ids=input_ids, attention_mask=attention_mask,labels=input_ids).loss 
                regularization_term=model.steer.regularization_term()
            scaler.scale(loss + hparams.regularization * regularization_term).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  
            batch_stance = batch_stance.to(model.torch_dtype)
            model.steer.set_value(steer_values=batch_stance)  #set steer value
            
            loss = model.model(input_ids=input_ids, attention_mask=attention_mask,labels=input_ids).loss
            regularization_term = model.steer.regularization_term()
            (loss + hparams.regularization * regularization_term).backward()
            optimizer.step()

        loss_mean.update(loss)
        pbar.set_description(f"{loss_mean.value}, {regularization_term.item()}")  
        if (step_i + 1) % hparams.log_step == 0:  
            print(pbar.desc, flush=True)


    delta = model.steer
    if hparams.save_vectors is True:
        torch.save([hparams,
                    model.steer.state_dict(),
                    max(n_steps,start_step)], ckpt_name)  
    if del_model:
        model.model.to('cpu')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return delta.state_dict()