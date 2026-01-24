
from .BaseModelTrainer import ModelTrainer
import torch, random
from tqdm.auto import tqdm
import torch.nn.functional as F
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from .utils.model_utils import get_lr
import numpy as np
from transformers import get_scheduler
from .utils.data_utils import make_preference_data_module
import csv
import os


class SFTModelTrainer(ModelTrainer):
    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'SFTModelTrainer'

    def make_preference_dataloader(self, examples, **kwargs):
        data_module = make_preference_data_module(self.model.tokenizer, examples, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.hparams.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, # we shuffle for examples.
            batch_size=self.hparams.batch_size, 
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader

    def train(self, examples, **kwargs):

        # prepare the data
        print(kwargs)
        train_dataloader = self.make_preference_dataloader(
            examples, **kwargs)
    
        torch.cuda.empty_cache()

        # prepare the optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            self.model.steer_vector.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        print(optimizer.param_groups) 

        num_training_steps = self.hparams.n_epochs * (len(train_dataloader) // self.hparams.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            "linear", 
            optimizer=optimizer,
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
        

        # training loop
        progress_bar, curr_step, logging_step = tqdm(range(num_training_steps), leave=True), 0, 0
        
        for epoch in range(self.hparams.n_epochs):
            for step, batch in enumerate(train_dataloader):
                expanded_batch_size = self.hparams.batch_size * len(self.preference_pairs)
                minibatch_size = self.hparams.batch_size
                num_minibatches = (expanded_batch_size + minibatch_size - 1) // minibatch_size
                
                # prepare the batch data
                winning_inputs = {k: [] for k in ["input_ids", "attention_mask", "labels", "intervention_locations", "steering_factors"]}
                losing_inputs = {k: [] for k in ["input_ids", "attention_mask", "labels", "intervention_locations", "steering_factors"]}
                
                for i in range(self.hparams.batch_size):
                    for pair in self.preference_pairs:
                        # fill the winning and losing inputs

                        winning_inputs["input_ids"].append(batch[f"{pair}_winning_input_ids"][i])
                        winning_inputs["attention_mask"].append(batch[f"{pair}_winning_attention_mask"][i])
                        winning_inputs["labels"].append(batch[f"{pair}_winning_labels"][i])
                        winning_inputs["intervention_locations"].append(batch[f"{pair}_winning_intervention_locations"][i])
                        losing_inputs["input_ids"].append(batch[f"{pair}_losing_input_ids"][i])
                        losing_inputs["attention_mask"].append(batch[f"{pair}_losing_attention_mask"][i])
                        losing_inputs["labels"].append(batch[f"{pair}_losing_labels"][i])
                        losing_inputs["intervention_locations"].append(batch[f"{pair}_losing_intervention_locations"][i])
                        
                        # set the steering factors according to the type of the preference pair
                        if "_add" in pair: 
                            winning_inputs["steering_factors"].append(torch.tensor(random.choice(self.hparams.steering_factors)))
                            losing_inputs["steering_factors"].append(torch.tensor(random.choice(self.hparams.steering_factors)))
                        else: 
                            if self.hparams.substraction_type == "null_it_out": 
                                winning_inputs["steering_factors"].append(torch.tensor(0.0))
                                losing_inputs["steering_factors"].append(torch.tensor(0.0))
                            else: 
                                winning_inputs["steering_factors"].append(torch.tensor(-1.0 * random.choice(self.hparams.steering_factors)))
                                losing_inputs["steering_factors"].append(torch.tensor(-1.0 * random.choice(self.hparams.steering_factors)))
                
                # initialize the variables for accumulating the current batch metrics and loss
                loss_sum = 0
                
                # loop through the minibatches and compute the gradient

                for mb in range(num_minibatches):
                    start_idx = mb * minibatch_size
                    end_idx = min((mb + 1) * minibatch_size, expanded_batch_size)
                    
                    if start_idx >= expanded_batch_size:
                        break
                    
                    # minibatch_inputs = {
                    #     k: torch.stack(winning_inputs[k][start_idx:end_idx] + losing_inputs[k][start_idx:end_idx], dim=0).to(self.model.device) 
                    #     for k, _ in winning_inputs.items()
                    # }

                    pos_minibatch_inputs = {
                        k: torch.stack(winning_inputs[k][start_idx:end_idx], dim=0).to(self.model.device) 
                        for k, _ in winning_inputs.items()
                    }

                    # prepare the intervention subspaces
                    subspaces = [{ "steering_factor": pos_minibatch_inputs["steering_factors"]}]
                    subspace_repeat = 1 if not isinstance(self.model.steer_vector, list) else len(self.model.steer_vector)
                    subspaces = subspaces * subspace_repeat
                    self.model.steer_vector.subspaces = subspaces
                    
                    # model forward propagation, severe bug here
                    self.model.steer_vector.intervention_locations = pos_minibatch_inputs["intervention_locations"]
                    pos_outputs_orig = self.model.model(
                        input_ids=pos_minibatch_inputs["input_ids"],
                        attention_mask=pos_minibatch_inputs["attention_mask"],
                        labels=pos_minibatch_inputs["labels"],
                        use_cache=False
                    )

                    neg_minibatch_inputs = {
                        k: torch.stack(losing_inputs[k][start_idx:end_idx], dim=0).to(self.model.device) 
                        for k, _ in losing_inputs.items()
                    }

                    # prepare the intervention subspaces
                    subspaces = [{ "steering_factor": neg_minibatch_inputs["steering_factors"]}]
                    subspace_repeat = 1 if not isinstance(self.model.steer_vector, list) else len(self.model.steer_vector)
                    subspaces = subspaces * subspace_repeat
                    self.model.steer_vector.subspaces = subspaces
                    
                    # model forward propagation, severe bug here
                    self.model.steer_vector.intervention_locations = neg_minibatch_inputs["intervention_locations"]
                    neg_outputs_orig = self.model.model(
                        input_ids=neg_minibatch_inputs["input_ids"],
                        attention_mask=neg_minibatch_inputs["attention_mask"],
                        labels=neg_minibatch_inputs["labels"],
                        use_cache=False
                    )

                    # calculate the reference model output ref_outputs
                    if hasattr(self.model, "steer_vector"):
                        # remove the intervention
                        self.model.reset("sft")
                        
                        # forward propagation without intervention
                        pos_ref_outputs = self.model.model(
                            input_ids=pos_minibatch_inputs["input_ids"],
                            attention_mask=pos_minibatch_inputs["attention_mask"],
                            labels=pos_minibatch_inputs["labels"],
                            use_cache=False
                        )
                        neg_ref_outputs = self.model.model(
                            input_ids=neg_minibatch_inputs["input_ids"],
                            attention_mask=neg_minibatch_inputs["attention_mask"],
                            labels=neg_minibatch_inputs["labels"],
                            use_cache=False
                        )

                        # restore the intervention to all layers
                        for layer in self.layers:
                            self.model.set_intervention(layer, self.model.steer_vector, "sft")
                    else:
                        pos_ref_outputs = pos_outputs_orig
                        neg_ref_outputs = neg_outputs_orig

                    pos_loss_weight = self.hparams.pos_loss_weight
                    neg_loss_weight = self.hparams.neg_loss_weight
                    margin_penalty_weight = self.hparams.margin_penalty_weight
                    ref_loss_weight = self.hparams.ref_loss_weight
                    margin_threshold = self.hparams.margin_threshold

                    pos_loss = pos_outputs_orig.loss
                    neg_loss = neg_outputs_orig.loss
                    # margin constraint
                    margin_loss = margin_penalty_weight * F.relu(margin_threshold - (neg_loss - pos_loss))

                    # ref constraint
                    pos_ref_violation = F.relu(pos_loss - pos_ref_outputs.loss)
                    neg_ref_violation = F.relu(neg_loss - neg_ref_outputs.loss)
                    ref_loss = ref_loss_weight * (pos_ref_violation + neg_ref_violation)

                    # total
                    steer_loss = pos_loss_weight * pos_loss + neg_loss_weight * neg_loss + margin_loss + ref_loss

                    # Save steer_loss and ref_loss to CSV

                    if hasattr(self.hparams, "loss_output_dir") and self.hparams.loss_output_dir is not None:
                        self.loss_log_file = os.path.join(self.hparams.loss_output_dir if hasattr(self.hparams, "loss_output_dir") else ".", f"train_losses.csv")
                        # Write header if file does not exist
                        if not os.path.exists(self.loss_log_file):
                            # Build header dynamically based on weight values
                            header = ["epoch", "step", "pos_steer_loss"]
                            if neg_loss_weight != 0 and neg_loss_weight != 0.0:
                                header.append("neg_steer_loss")
                            if ref_loss_weight != 0 and ref_loss_weight != 0.0:
                                header.extend(["pos_ref_loss", "neg_ref_loss"])
                            if margin_penalty_weight != 0 and margin_penalty_weight != 0.0:
                                header.append("margin_loss")
                            if ref_loss_weight != 0 and ref_loss_weight != 0.0:
                                header.append("ref_loss")
                            header.extend(["pos_steering_factors", "neg_steering_factors"])
                            
                            with open(self.loss_log_file, "w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(header)

                        # Build row dynamically based on weight values
                        row = [epoch, step, pos_outputs_orig.loss.item()]
                        if neg_loss_weight != 0 and neg_loss_weight != 0.0:
                            row.append(neg_outputs_orig.loss.item())
                        if ref_loss_weight != 0 and ref_loss_weight != 0.0:
                            row.extend([pos_ref_outputs.loss.item(), neg_ref_outputs.loss.item()])
                        if margin_penalty_weight != 0 and margin_penalty_weight != 0.0:
                            row.append(margin_loss.item())
                        if ref_loss_weight != 0 and ref_loss_weight != 0.0:
                            row.append(ref_loss.item())
                        row.extend([pos_minibatch_inputs["steering_factors"], neg_minibatch_inputs["steering_factors"]])

                        with open(self.loss_log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(row)

                    # Build print message dynamically
                    print_parts = [f"steer_loss: %.6f" % (steer_loss.item())]
                    if margin_penalty_weight != 0 and margin_penalty_weight != 0.0:
                        print_parts.append(f"margin_loss: %.6f" % (margin_loss.item()))
                    if ref_loss_weight != 0 and ref_loss_weight != 0.0:
                        print_parts.append(f"ref_loss: %.6f" % (ref_loss.item()))
                    print(" ".join(print_parts))
                    minibatch_loss = steer_loss
                    
                    # Normalize loss by total number of minibatches for this step
                    # (instead of dividing by gradient_accumulation_steps)
                    minibatch_loss = minibatch_loss / (num_minibatches * self.hparams.gradient_accumulation_steps)
                    
                    # Backward pass for this minibatch
                    if not self.hparams.inference:
                        minibatch_loss.backward()
                    else:
                        print("inference only, no backward!!!")
                    
                    # Track total loss for logging
                    loss_sum += steer_loss.detach() * (end_idx - start_idx)


                loss = loss_sum / expanded_batch_size
               
                # --- 4.8 optimizer step ---
                if (step + 1) % self.hparams.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.steer_vector.parameters(), 1.0)
                    curr_lr = get_lr(optimizer) 
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    

                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (
                            curr_lr, loss))
                    print(f"Epoch {epoch}, Step {step}")

                    curr_step += 1

        progress_bar.close()
        

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        self.max_activations = {}
        return self.max_activations

