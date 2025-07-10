from .BaseModelTrainer import ModelTrainer
import torch, random
from tqdm.auto import tqdm
import torch.nn.functional as F
from typing import Tuple
from torch.utils.data import DataLoader
from .utils.model_utils import get_lr
import numpy as np
from transformers import get_scheduler
from .utils.data_utils import make_preference_data_module


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Ref of Eric's repo: 
    https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    gemma: float,
                    simpo_scaler: float,
                    winning_lens: torch.LongTensor,
                    losing_lens: torch.LongTensor,
                    label_smoothing: float = 0.0,
                    loss_type: str = "dpo",
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Ref of Eric's repo: 
    https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L45

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        loss_type: different preference loss functions.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    ref_logratios_reverse = reference_rejected_logps - reference_chosen_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if loss_type == "ipo":
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    elif loss_type == "dpo":
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    elif loss_type == "simpo":
        losses = -F.logsigmoid((beta / winning_lens) * policy_chosen_logps - (beta / losing_lens) * policy_rejected_logps - gemma)
    elif loss_type == "scaled_simpo":
        scaled_policy_chosen_logps = (
            torch.max(ref_logratios_reverse * simpo_scaler, torch.ones_like(ref_logratios_reverse)) / winning_lens) * policy_chosen_logps
        scaled_policy_rejected_logps = (1.0 / losing_lens) * policy_rejected_logps
        losses = -F.logsigmoid(scaled_policy_chosen_logps - scaled_policy_rejected_logps)
        """
        negative steering:

        input: steering prefix + original instruction
        winning output: original response
        losing output: steered response

        scaler = p_ref(losing output) - p_ref(winning output)
        losses = -F.logsigmoid(
            (torch.max(scaler, 1) / winning_lens) * policy_chosen_logps - (1.0 / losing_lens) * policy_rejected_logps)
        """
    elif loss_type == "apo_zero":
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        losses = -F.logsigmoid(beta * chosen_logratios) + F.logsigmoid(beta * rejected_logratios) 
    else:
        raise ValueError(f"Loss type {loss_type} not supported")

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


class PreferenceModelTrainer(ModelTrainer):
    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'PreferenceModelTrainer'

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

    def _compute_metrics(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, 
                         chosen_rewards, rejected_rewards, losses):
        """Helper method to compute metrics for a minibatch"""
        metrics = {}
        
        # Compute reward accuracies
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        metrics[f'rewards_train/steer_chosen_rewards'] = chosen_rewards.mean().cpu().float().numpy().tolist()
        metrics[f'rewards_train/steer_rejected_rewards'] = rejected_rewards.mean().cpu().float().numpy().tolist()
        metrics[f'rewards_train/steer_margins'] = (chosen_rewards - rejected_rewards).mean().cpu().float().numpy().tolist()
        metrics[f'rewards_train/pos_steer_reward_accuracies'] = np.array(reward_accuracies.cpu().numpy().tolist()[:len(reward_accuracies)//2]).mean()
        metrics[f'rewards_train/neg_steer_reward_accuracies'] = np.array(reward_accuracies.cpu().numpy().tolist()[len(reward_accuracies)//2:]).mean()
        metrics[f'rewards_train/steer_accuracies'] = reward_accuracies.mean().cpu().numpy().tolist()
        metrics[f'logps_train/steer_chosen'] = chosen_logps.detach().mean().cpu().float().numpy().tolist()
        metrics[f'logps_train/steer_rejected'] = rejected_logps.detach().mean().cpu().float().numpy().tolist()
        metrics[f'loss/steer'] = losses.mean().detach().cpu().float().numpy().tolist()
        
        return metrics

    def train(self, examples, **kwargs):
   
        # prepare the data
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
                batch_metrics = {}
                loss_sum = 0
                
                # loop through the minibatches and compute the gradient
                for mb in range(num_minibatches):
                    start_idx = mb * minibatch_size
                    end_idx = min((mb + 1) * minibatch_size, expanded_batch_size)
                    
                    if start_idx >= expanded_batch_size:
                        break
                    
                    minibatch_inputs = {
                        k: torch.stack(winning_inputs[k][start_idx:end_idx] + losing_inputs[k][start_idx:end_idx], dim=0).to(self.model.device) 
                        for k, _ in winning_inputs.items()
                    }

                    # prepare the intervention subspaces
                    subspaces = [{ "steering_factor": minibatch_inputs["steering_factors"]}]
                    subspace_repeat = 1 if not isinstance(self.model.steer_vector, list) else len(self.model.steer_vector)
                    subspaces = subspaces * subspace_repeat
                    self.model.steer_vector.subspaces = subspaces
                    
                    # model forward propagation, severe bug here
                    self.model.steer_vector.intervention_locations = minibatch_inputs["intervention_locations"]
                    policy_outputs_orig = self.model.model(
                        input_ids=minibatch_inputs["input_ids"],
                        attention_mask=minibatch_inputs["attention_mask"],
                        use_cache=False
                    )

                    # calculate the reference model output ref_outputs
                    if hasattr(self.model, "steer_vector"):
                        # remove the intervention
                        self.model.reset("reps")

                        # forward propagation without intervention
                        ref_outputs = self.model.model(
                            input_ids=minibatch_inputs["input_ids"],
                            attention_mask=minibatch_inputs["attention_mask"],
                            use_cache=False
                        )

                        # restore the intervention to all layers
                        for layer in self.layers:
                            self.model.set_intervention(layer, self.model.steer_vector, "reps")
                    else:
                        ref_outputs = policy_outputs_orig
                    
                    # calculate the loss
                    policy_outputs_orig_logps = _get_batch_logps(policy_outputs_orig.logits, minibatch_inputs["labels"], average_log_prob=False)
                    ref_logps = _get_batch_logps(ref_outputs.logits, minibatch_inputs["labels"], average_log_prob=False)
                    
                    # split the logps into winning and losing parts
                    minibatch_size_actual = minibatch_inputs["input_ids"].shape[0]
                    steer_chosen_logps = policy_outputs_orig_logps[:minibatch_size_actual//2]
                    steer_rejected_logps = policy_outputs_orig_logps[minibatch_size_actual//2:]
                    steer_ref_chosen_logps = ref_logps[:minibatch_size_actual//2]
                    steer_ref_rejected_logps = ref_logps[minibatch_size_actual//2:]
                    
                    # prepare the parameters for calculating the preference loss
                    winning_lens = minibatch_inputs["attention_mask"][:minibatch_size_actual//2].sum(dim=-1)
                    losing_lens = minibatch_inputs["attention_mask"][minibatch_size_actual//2:].sum(dim=-1)
                    pos_loss_kwargs = {
                        'beta': self.hparams.beta, 
                        'gemma': self.hparams.gemma,
                        'simpo_scaler': self.hparams.simpo_scaler,
                        'reference_free': self.hparams.reference_free, 
                        'label_smoothing': self.hparams.label_smoothing, 
                        'loss_type': self.hparams.loss_type, 
                        'winning_lens': winning_lens,
                        'losing_lens': losing_lens
                    }
                    # call the preference loss function
                    steer_losses, steer_chosen_rewards, steer_rejected_rewards = preference_loss(
                        steer_chosen_logps, steer_rejected_logps,
                        steer_ref_chosen_logps, steer_ref_rejected_logps,
                        **pos_loss_kwargs
                    )

                    steer_loss = steer_losses.mean()
                    minibatch_loss = steer_loss
                    
                    # normalize the loss according to the number of minibatches and gradient accumulation steps
                    minibatch_loss = minibatch_loss / (num_minibatches * self.hparams.gradient_accumulation_steps)
                    
                    # backward propagation
                    minibatch_loss.backward()
                    
                    # accumulate the total loss
                    loss_sum += steer_loss.detach() * (end_idx - start_idx)
                    
                    # compute and accumulate the metrics of the current minibatch
                    minibatch_metrics = self._compute_metrics(
                        steer_chosen_logps, steer_rejected_logps,
                        steer_ref_chosen_logps, steer_ref_rejected_logps,
                        steer_chosen_rewards, steer_rejected_rewards,
                        steer_losses
                    )
                    
                    # accumulate the metrics of the current minibatch
                    for k, v in minibatch_metrics.items():
                        if k not in batch_metrics:
                            batch_metrics[k] = [v * (end_idx - start_idx)]
                        else:
                            batch_metrics[k].append(v * (end_idx - start_idx))
                
                # calculate the average metrics of the current batch
                metrics = {}
                for k, v in batch_metrics.items():
                    metrics[k] = sum(v) / expanded_batch_size
                
                loss = loss_sum / expanded_batch_size
                metrics[f'loss/train'] = loss.cpu().float().numpy().tolist()
                metrics[f'loss/steer'] = loss.cpu().float().numpy().tolist()

                # --- 4.8 optimizer step ---
                if (step + 1) % self.hparams.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.steer_vector.parameters(), 1.0)
                    curr_lr = get_lr(optimizer) 
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f || steer acc %.6f" % (
                            curr_lr, loss, metrics.get('rewards_train/steer_accuracies', 0.0))
                    )
                    curr_step += 1

        progress_bar.close()
        
    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        self.max_activations = {}
        return self.max_activations

