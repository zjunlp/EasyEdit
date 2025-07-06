<<<<<<< HEAD
from .PreferenceModelTrainer import PreferenceModelTrainer
import torch
from tqdm.auto import tqdm

class RepsVectorTrainer(PreferenceModelTrainer):
=======
from .preference_model import *

class RepsVectorTrainer(PreferenceModel):
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'PreferenceVector'

    def make_model(self, **kwargs):
        """
<<<<<<< HEAD
        create a model with intervention
        """
        from ..models.interventions import RePSVectorIntervention
        
        print("**Getting embed dim from the following model config**")
        
        intervention_type = kwargs.get("intervention_type", "addition") # addition
        if intervention_type == "addition":
            # create a preference vector intervention object
            steer_vector = RePSVectorIntervention(
                embed_dim=kwargs.get("embed_dim", self.model.model.config.hidden_size), # set the embedding dimension
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),            # set the low rank dimension, 4
                dropout=kwargs.get("dropout", 0.0),                                # set the dropout rate
                intervention_positions_dropout=kwargs.get("intervention_positions_dropout", 0.0) # set the dropout rate of the intervention positions
            )
        else:
            raise ValueError(f"Intervention type {intervention_type} not supported")

        self.intervention_type = intervention_type
        self.model.steer_vector = steer_vector.to(self.model.device, dtype=self.model.torch_dtype)
        self.model.steer_vector.train()
        
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])
        
        for layer in self.layers:
            intervention_copy = self.model.steer_vector  # all layers share the same intervention instance
            self.model.set_intervention(layer, intervention_copy, "reps")
    
 
=======
        根据给定的关键字参数创建一个可干预的模型 (Intervenable Model)。
        这个函数会配置并应用一个特定的干预 (intervention) 到基础模型上。
        """
        # 从关键字参数中获取 "mode"，如果未提供，则默认为 "latent"
        # "mode" 决定了创建哪种类型的干预模型
        mode = kwargs.get("mode", "latent")  # train
        
        # 从关键字参数中获取要覆盖的模型组件，如果未提供，则为 None
        # 这允许用户指定干预的具体位置
        overwrite_component = kwargs.get("overwrite_component", None) # none
        
        # 打印提示信息，说明将从模型配置中获取嵌入维度
        print("**Getting embed dim from the following model config**")
        
        # 如果模式是 "steering"（引导模式）
        if mode == "steering":
            # 获取干预类型，默认为 "addition"（加法干预）
            intervention_type = kwargs.get("intervention_type", "addition")
            if intervention_type == "addition":
                # 创建一个加法干预对象
                ax = AdditionIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size), # 设置嵌入维度，默认使用模型的隐藏层大小
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),            # 设置低秩维度，默认为 1
                )
            elif intervention_type == "addition_suppression":
                # 创建一个加法抑制干预对象
                ax = AdditionSuppressionIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size), # 设置嵌入维度
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),            # 设置低秩维度
                )
            else:
                # 如果干预类型不支持，则抛出异常
                raise ValueError(f"Intervention type {intervention_type} not supported")
        else: # 如果模式不是 "steering" (例如，是 "latent" 模式)
            # 获取干预类型，默认为 "addition"
            intervention_type = kwargs.get("intervention_type", "addition") # addition
            if intervention_type == "addition":
                # 创建一个偏好向量干预对象
                ax = PreferenceVectorIntervention(
                    embed_dim=kwargs.get("embed_dim", self.model.config.hidden_size), # 设置嵌入维度
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),            # 设置低秩维度，4
                    dropout=kwargs.get("dropout", 0.0),                                # 设置 dropout 率
                    intervention_positions_dropout=kwargs.get("intervention_positions_dropout", 0.0) # 设置干预位置的 dropout 率
                )

        # 将干预类型保存为实例属性
        self.intervention_type = intervention_type
        
        # 确定要应用干预的层。如果 self.steering_layers 已定义，则使用它，否则使用 self.layer
        layers = self.steering_layers if self.steering_layers else [self.layer]
        
        # 将创建的干预对象 (ax) 移动到指定的设备 (如 GPU 或 CPU)
        self.ax = ax.to(self.device)
        
        # 将干预对象设置为训练模式
        self.ax.train()
        
        # 创建可干预模型的配置 (IntervenableConfig)
        # 这个配置定义了干预将在模型的哪些位置（层和组件）以及如何应用
        ax_config = IntervenableConfig(
            representations=[
                {
                    "layer": l, # 干预的层索引
                    "component": f"model.layers[{l}].output" if overwrite_component is None else overwrite_component, # 干预的具体组件，层的输出
                    "low_rank_dimension": kwargs.get("low_rank_dimension", 1), # 低秩维度
                    "intervention": self.ax # 要应用的干预对象
                } for l in layers # 为 `layers` 列表中的每个层创建配置
            ]
        )
        
        # 使用基础模型 (self.model) 和干预配置 (ax_config) 创建可干预模型
        ax_model = IntervenableModel(ax_config, self.model)
        
        # 将新的可干预模型移动到指定的设备
        ax_model.set_device(self.device)
        
        # 将创建的可干预模型保存为实例属性
        self.ax_model = ax_model
        
        # 获取偏好对配置，默认为 ["orig_add"]，这可能用于模型的训练或评估
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])
    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        return_max_act_only = kwargs.get("return_max_act_only", False)
        is_chat_model = kwargs.get("is_chat_model", False)
        eager_prepare_df = kwargs.get("eager_prepare_df", False)
        overwrite_concept_id = kwargs.get("overwrite_concept_id", None)

        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        progress_bar = tqdm(range(0, len(examples), batch_size), desc="Processing batches")
        for i in progress_bar:
            batch = examples.iloc[i:i + batch_size]
            if eager_prepare_df:
                batch = prepare_df(batch, self.tokenizer, is_chat_model)

            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            outputs = self.ax(
                gather_acts[:, kwargs["prefix_length"]:],  # no bos token
                subspaces={
                    "subspaces": torch.tensor([overwrite_concept_id]*len(batch["input"])).to(self.device) \
                    if overwrite_concept_id is not None else torch.tensor(batch["concept_id"].tolist()).to(self.device),
                    "k": 1
                })
            ax_acts = outputs.latent[0].float().detach().cpu()

            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            # Process each sequence in the batch
            for seq_idx, ax_seq in enumerate(ax_acts):
                acts = ax_seq[:seq_lens[seq_idx]].flatten().data.numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                all_max_act.append(max_act)
            # clear memory and cache
            del ax_acts
            del gather_acts
            torch.cuda.empty_cache()

        return {
            "max_act": all_max_act
        }
        
    @torch.no_grad()
    def predict_latents(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)

        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            # Batch encode all inputs
            inputs = self.tokenizer(
                batch["input"].tolist(), return_tensors="pt", 
                add_special_tokens=True, padding=True, truncation=True).to(self.device)
            
            gather_acts = gather_residual_activations(
                self.model, self.layer, inputs)
            
            ax_acts_batch = torch.relu(torch.matmul(
                gather_acts[:, kwargs["prefix_length"]:], # bs, s, h
                self.ax.proj.weight.permute(1, 0) # h, d
            )).float().cpu().numpy()
            
            # Process each sequence in the batch
            seq_lens = inputs["attention_mask"].sum(dim=1) - kwargs["prefix_length"] # no bos token
            for seq_idx, row in enumerate(batch.itertuples()):
                # select acts with attention mask
                acts_batch = ax_acts_batch[
                    seq_idx, :seq_lens[seq_idx]]
                
                concept_acts = []
                concept_max_act = []
                concept_max_act_idx = []
                concept_max_token = []
                concept_tokens = []
                for row_idx in range(ax_acts_batch.shape[-1]):
                    # row_idx here is the concept id
                    acts = acts_batch[:, row_idx].flatten().tolist()
                    acts = [round(x, 3) for x in acts]
                    max_act = max(acts)
                    max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                    max_act_idx = max_act_indices[0]
                    # Get tokens for this specific sequence
                    tokens = self.tokenizer.tokenize(row.input)[kwargs["prefix_length"]-1:] # -1 is because it does not prepend BOS token
                    max_token = tokens[max_act_idx]
                    concept_acts.append(acts)
                    concept_max_act.append(max_act)
                    concept_max_act_idx.append(max_act_idx)
                    concept_max_token.append(max_token)
                    concept_tokens.append(tokens)
                all_acts.append(concept_acts)
                all_max_act.append(concept_max_act)
                all_max_act_idx.append(concept_max_act_idx)
                all_max_token.append(concept_max_token)
                all_tokens.append(concept_tokens)
        return {
            # "acts": all_acts,
            "max_act": all_max_act,
            # "max_act_idx": all_max_act_idx,
            # "max_token": all_max_token,
            # "tokens": all_tokens
        }
>>>>>>> ab3202ed (New RePS related configuration files and model trainers have been added)
