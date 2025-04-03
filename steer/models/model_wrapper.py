import copy
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

from steer.utils.alg_dict import DTYPES_DICT
from steer.vector_appliers.lm_steer.apply_lm_steer_hparam import ApplyLmSteerHyperParams
from steer.models.utils import add_vector_from_position # find_instruction_end_postion
from typing import Optional


from steer.vector_generators.lm_steer.generate_lm_steer_hparam import LmSteerHyperParams
from steer.utils.hparams import HyperParams
from steer.vector_generators.lm_steer.lm_steer_helper import Hack_no_grad, Projected_Adaptor


class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_attn_activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output

    def add(self, activations):
        self.add_attn_activations = activations
    
class MLPWrapper(t.nn.Module):
    """
    Wrapper for MLP to save activations
    """

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.activations = None
        self.add_mlp_activations = None

    def forward(self, *args, **kwargs):
        output = self.mlp(*args, **kwargs)
        self.activations = output
        return output

    def add(self, activations):
        self.add_mlp_activations = activations

class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer, layer_id,model_name_or_path):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        # self.model_type=model_name_or_path.lower()
        if 'gpt' in model_name_or_path.lower():
            self.model_type = 'gpt'
        elif 'llama' in model_name_or_path.lower():
            self.model_type = 'llama'
        elif 'gemma' in model_name_or_path.lower():
            self.model_type = 'gemma'
        elif 'qwen' in model_name_or_path.lower():
            self.model_type = 'qwen'
 
        if self.model_type in ['gpt']:
            self.block.attn = AttnWrapper(self.block.attn)
            self.block.mlp = MLPWrapper(self.block.mlp)
            self.post_attention_layernorm=self.block.ln_2
        elif self.model_type in ['llama', 'gemma', 'qwen']:
            self.block.self_attn = AttnWrapper(self.block.self_attn)
            self.block.mlp = MLPWrapper(self.block.mlp)
            self.post_attention_layernorm=self.block.post_attention_layernorm
        else:
            # Raise an error if model_type is not supported
            raise ValueError(f"Unsupported model type: {self.model_type}")
 

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations_dict = {}  # use dict to store activations for different methods
        
        self.from_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

        self.layer_id = layer_id

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
            
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (
                t.norm(last_token_activations) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations_dict:
            augmented_output = output[0]
            for activations in self.add_activations_dict.values():
                if activations is not None:
                    position_ids = kwargs.get("position_ids", None)
                    augmented_output = add_vector_from_position(
                        matrix=augmented_output,
                        vector=activations,
                        position_ids=position_ids,
                        from_pos=self.from_position,
                    )
            output = (augmented_output,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        if self.model_type in ['gpt']:
            attn_output = self.block.attn.activations
        if self.model_type in ['llama', 'gemma', 'qwen']:
            attn_output = self.block.self_attn.activations
        
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        if self.model_type in ['llama', 'gemma', 'qwen', 'gpt'] :
            mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations, method_name="default"):
        """
        store activations for different methods
        """
        self.add_activations_dict[method_name] = activations

    def reset(self, method_name="all"):
        """
        only reset the activations for the specified method
        """
        if method_name == "all":
            self.add_activations_dict.clear()
        elif method_name in self.add_activations_dict:
            del self.add_activations_dict[method_name]
        
        self.activations = None
        # self.block.self_attn.activations = None
        if self.model_type in ['gpt']:
            self.block.attn.activations = None
        if self.model_type in ['llama', 'gemma', 'qwen']:
            self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []

class BaseModelWrapper:
    def __init__(
        self,
        torch_dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None,
    ):
        
        self.hparams = hparams    #initialize hyperparams
        self.use_chat = use_chat
        self.device = device
        self.torch_dtype = DTYPES_DICT.get(torch_dtype, t.float32)
        self.use_cache = use_cache
        self.model_name_or_path = model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            padding_side="right" if "gemma" in self.model_name_or_path else "left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            use_cache=self.use_cache
        )
        
        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path), device=self.device)

        # for i, layer in enumerate(self.model.model.layers):
        #     self.model.model.layers[i] = BlockOutputWrapper(
        #         layer, self.model.lm_head, self.model.model.norm, self.tokenizer, i, self.hparams.model_name_or_path
        #     )
        ### Customize layers and outputs for specific models
        self._adapt_model_layers()

    def _adapt_model_layers(self):
        """Override this method in subclasses for model-specific layer adaptations."""
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer, i, self.model_name_or_path
            )

    def replace_final_layer(self, hparams):
        
        embed_dim = self.model.lm_head.weight.shape[1]
        vocab_size = self.model.lm_head.weight.shape[0]  
        for _param in self.model.parameters():
            _param.requires_grad_(False)  # froze params

        if hparams.adapted_component == "final_layer" and hasattr(self.model, 'model'):  # default
            self.model.model = Hack_no_grad(self.model.model)  # Freeze the model layers
            self.steer = Projected_Adaptor(  #
                self.model.lm_head, hparams.adaptor_class, hparams.num_steers, embed_dim,
                vocab_size, hparams.rank, hparams.epsilon, hparams.init_var, "output")
            if hparams.adaptor_class == "multiply":
                self.steer.projector1 = t.nn.Parameter(self.steer.projector1.to(self.torch_dtype))
                self.steer.projector2 = t.nn.Parameter(self.steer.projector2.to(self.torch_dtype))
            elif hparams.adaptor_class == "add":
                self.steer.add_vec = t.nn.Parameter(self.steer.add_vec.to(self.torch_dtype))
            elif hparams.adaptor_class == "offset":
                self.steer.offset_vec = t.nn.Parameter(self.steer.offset_vec.to(self.torch_dtype))
            self.model.set_output_embeddings(self.steer)  
        else:
            raise ValueError('Mismatched adapted component or model structure')
        
    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self.model.model.layers:
            layer.from_position = pos

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations, method_name="default"):
        if hasattr(self.model, 'model') and isinstance(self.model.model, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
            self.model.model.module.layers[layer].add(activations, method_name)
        else:
            self.model.model.layers[layer].add(activations, method_name)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        if hasattr(self.model, 'model') and isinstance(self.model.model, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
            model_layers = self.model.model.module.layers
        else:
            model_layers = self.model.model.layers
        for layer in model_layers:
            layer.reset(method_name="all")
        if hasattr(self, 'prompt'):
            delattr(self, 'prompt')
        if hasattr(self, 'generate_prompts'):
            delattr(self, 'generate_prompts')
        self.reset_lm_steer()
            
    def reset_lm_steer(self):
        if hasattr(self, 'steer'):
            original_lm_head = self.steer.lm_head
            self.model.set_output_embeddings(original_lm_head)
            delattr(self, 'steer')
        if hasattr(self.model, 'model') and isinstance(self.model.model, Hack_no_grad):
            self.model.model = self.model.model.module
        for param in self.model.parameters():
            param.requires_grad_(True)  
            
    def reset(self, method_name):
        method_name = method_name.lower()
        if method_name in ['caa', 'vector_prompt','sae_feature','sta']:
            if hasattr(self.model, 'model') and isinstance(self.model.model, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
                model_layers = self.model.model.module.layers
            else:
                model_layers = self.model.model.layers
            for layer in model_layers:
                layer.reset(method_name=method_name)
                
        elif method_name in ['lm_steer']:
            self.reset_lm_steer()    
        elif method_name in ['prompt']:
            if hasattr(self, 'prompt'):
                delattr(self, 'prompt')
        else:
            raise ValueError(f"Method {method_name} not supported to reset")
            
            

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))

    def get_logits(self,tokens):
        logits = self.model(tokens).logits
        return logits
    
    def ori_generate(self, input_ids, **kwargs):
        # Save activation dictionaries
        saved_activations = {}
        if hasattr(self.model, 'model') and isinstance(self.model.model, Hack_no_grad):
            model_layers = self.model.model.module.layers
        else:
            model_layers = self.model.model.layers
            
        for i, layer in enumerate(model_layers):
            if hasattr(layer, 'add_activations_dict') and layer.add_activations_dict:
                saved_dict = {}
                for key, value in layer.add_activations_dict.items():
                    if isinstance(value, t.Tensor):
                        saved_dict[key] = value.clone().detach()
                    else:
                        try:
                            saved_dict[key] = copy.deepcopy(value)
                        except:
                            saved_dict[key] = value
                
                saved_activations[i] = saved_dict
                layer.add_activations_dict = {}
        
        # Save steer value if exists
        saved_steer_values = t.zeros(1)
        if hasattr(self, 'steer') and hasattr(self.steer, 'steer_values'):
            saved_steer_values = self.steer.steer_values
            self.steer.steer_values = t.zeros(1)
        
        # Generate text
        try:
            output = self.model.generate(
                input_ids=input_ids,
                **kwargs
            )
        finally:
            # Restore activation dictionaries
            for i, activations_dict in saved_activations.items():
                model_layers[i].add_activations_dict = activations_dict
            
            # Restore steer value
            if saved_steer_values is not None and hasattr(self, 'steer'):
                self.steer.steer_values = saved_steer_values
        
        return output
        

class LlamaWrapper(BaseModelWrapper):
    def __init__(
        self,
        torch_dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None,
    ):
        super().__init__(
            torch_dtype, 
            use_chat, 
            device,
            model_name_or_path,
            use_cache,
            override_model_weights_path, 
            hparams)

class GemmaWrapper(BaseModelWrapper):
    def __init__(
        self,
        torch_dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None
    ):

        super().__init__(
            torch_dtype, 
            use_chat, 
            device,
            model_name_or_path,
            use_cache,
            override_model_weights_path, 
            hparams)

class QwenWrapper(BaseModelWrapper):
    def __init__(
        self,
        torch_dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None
    ):
        super().__init__(
            torch_dtype, 
            use_chat, 
            device,
            model_name_or_path,
            use_cache,
            override_model_weights_path, 
            hparams)

class GPTWrapper(BaseModelWrapper):
    def __init__(
        self,
        torch_dtype = t.float32,   #change to float16
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None
    ):
        super().__init__(
            torch_dtype, 
            use_chat,
            device,
            model_name_or_path,
            use_cache,
            override_model_weights_path, 
            hparams)


    def _adapt_model_layers(self):
        for i, layer in enumerate(self.model.transformer.h):
            self.model.transformer.h[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.transformer.ln_f, self.tokenizer, i, self.model_name_or_path
            )

    def replace_final_layer(self,hparams):
        embed_dim = self.model.lm_head.weight.shape[1]
        vocab_size = self.model.lm_head.weight.shape[0]
        for _param in self.model.parameters():
            _param.requires_grad_(False)  # froze params

        if hparams.adapted_component == "final_layer":  # default
            self.model.transformer = Hack_no_grad(self.model.transformer)  # Freeze the transformer layers
            self.steer = Projected_Adaptor(  #
                self.model.lm_head, hparams.adaptor_class, hparams.num_steers, embed_dim,
                vocab_size, hparams.rank, hparams.epsilon, hparams.init_var, "output")
            if hparams.adaptor_class == "multiply":
                self.steer.projector1 = t.nn.Parameter(self.steer.projector1.to(self.torch_dtype))
                self.steer.projector2 = t.nn.Parameter(self.steer.projector2.to(self.torch_dtype))
            elif hparams.adaptor_class == "add":
                self.steer.add_vec = t.nn.Parameter(self.steer.add_vec.to(self.torch_dtype))
            elif hparams.adaptor_class == "offset":
                self.steer.offset_vec = t.nn.Parameter(self.steer.offset_vec.to(self.torch_dtype))
            self.model.set_output_embeddings(self.steer) 

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.transformer.h:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self.model.transformer.h:
            layer.from_position = pos

    def get_last_activations(self, layer):
        return self.model.transformer.h[layer].activations

    def set_add_activations(self, layer, activations, method_name="default"):
        if isinstance(self.model.transformer, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
            self.model.transformer.module.h[layer].add(activations, method_name)
        else:
            self.model.transformer.h[layer].add(activations, method_name)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.transformer.h[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.transformer.h[layer].dot_products

    def reset_all(self):
        if isinstance(self.model.transformer, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
            model_layers = self.model.transformer.module.h
        else:
            model_layers = self.model.transformer.h
        for layer in model_layers:
            layer.reset(method_name="all")
        
        if hasattr(self, 'prompt'):
            delattr(self, 'prompt')
        if hasattr(self, 'generate_prompts'):
            delattr(self, 'generate_prompts')
        self.reset_lm_steer()
    
    def reset_lm_steer(self):
        if hasattr(self, 'steer'):
            original_lm_head = self.steer.lm_head
            self.model.set_output_embeddings(original_lm_head)
            delattr(self, 'steer')
        if isinstance(self.model.transformer, Hack_no_grad):
            self.model.transformer = self.model.transformer.module
        for param in self.model.parameters():
            param.requires_grad_(True)
            
    def reset(self, method_name):
        method_name = method_name.lower()
        if method_name in ['caa', 'vector_prompt','sae_feature','sta']:
            if isinstance(self.model.transformer, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
                model_layers = self.model.transformer.module.h
            else:
                model_layers = self.model.transformer.h
            for layer in model_layers:
                layer.reset(method_name=method_name)
                
        elif method_name in ['lm_steer']:
            self.reset_lm_steer()
        elif method_name in ['prompt']:
            if hasattr(self, 'prompt'):
                delattr(self, 'prompt')
        else:
            raise ValueError(f"Method {method_name} not supported to reset")
    
    def ori_generate(self, input_ids, **kwargs):
        # Save activation dictionaries
        saved_activations = {}
        if hasattr(self.model, 'transformer') and isinstance(self.model.transformer, Hack_no_grad):
            model_layers = self.model.transformer.module.h
        else:
            model_layers = self.model.transformer.h
            
        for i, layer in enumerate(model_layers):
            if hasattr(layer, 'add_activations_dict') and layer.add_activations_dict:
                saved_activations[i] = copy.deepcopy(layer.add_activations_dict)
                layer.add_activations_dict = {}
        
        # Save steer value if exists
        saved_steer_value = 0
        if hasattr(self, 'steer') and hasattr(self.steer, 'steer_value'):
            saved_steer_value = self.steer.steer_value
            self.steer.steer_value = 0
        
        # Generate text
        try:
            output = self.model.generate(
                input_ids=input_ids,
                **kwargs
            )
        finally:
            # Restore activation dictionaries
            for i, activations_dict in saved_activations.items():
                model_layers[i].add_activations_dict = activations_dict
            
            # Restore steer value
            if saved_steer_value is not None and hasattr(self, 'steer'):
                self.steer.steer_value = saved_steer_value
        
        return output
 

# class LlamaWrapper:
#     def __init__(
#         self,
#         model_name_or_path,
#         torch_dtype = t.float32,   
#         use_chat: bool = False,
#         override_model_weights_path: Optional[str] = None,
#         hparams:HyperParams=None,
#     ):
#         if not t.cuda.is_available():
#             self.device = "cpu"
#         self.hparams = hparams    #initialize hyperparams
#         self.use_chat = use_chat
#         self.model_name_path = model_name_or_path
#         self.device = hparams.device if hasattr(hparams, "device") else "cuda" if t.cuda.is_available() else "cpu"
  
#         self.torch_dtype = torch_dtype
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name_path,
#             padding_side="left",
#         )
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name_path,
#             torch_dtype=self.torch_dtype,
#             device_map=self.device,
#         )
#         if override_model_weights_path is not None:
#             self.model.load_state_dict(t.load(override_model_weights_path), device=self.device)
#         for i, layer in enumerate(self.model.model.layers):
#             self.model.model.layers[i] = BlockOutputWrapper(
#                 layer, self.model.lm_head, self.model.model.norm, self.tokenizer, i,self.hparams.model_name_or_path
#             )

#         if isinstance(self.hparams,LmSteerHyperParams) or isinstance(self.hparams,ApplyLmSteerHyperParams):   #If the hyperparameter passes LmSteer, perform corresponding initialization and replace the output layer.
#             self.init_lm_steer(self.hparams)
#             self.replace_final_layer(self.hparams)

#     def init_lm_steer(self,hparams):
        
#         self.adapted_component = hparams.adapted_component
#         self.init_var = hparams.init_var
#         self.num_steers = hparams.num_steers
#         self.embed_dim = self.model.lm_head.weight.shape[1]
#         self.vocab_size = self.model.lm_head.weight.shape[0]    
#         # self.rank = hparams.rank        #avoid conflict
#         # self.epsilon = hparams.epsilon
#         # self.adaptor_class = hparams.adaptor_class
        
#     def replace_final_layer(self,hparams):
#         for _param in self.model.parameters():
#             _param.requires_grad_(False)  # froze params

#         if hparams.adapted_component == "final_layer":  # default
#             self.model.model = Hack_no_grad(self.model.model)  # no_grad ,cut loss
#             self.steer = Projected_Adaptor(  #
#                 self.model.lm_head, hparams.adaptor_class, self.num_steers, self.embed_dim,
#                 self.vocab_size, hparams.rank, hparams.epsilon, self.init_var, "output")
#             self.model.set_output_embeddings(self.steer)  
    
#     def set_save_internal_decodings(self, value: bool):
#         for layer in self.model.model.layers:
#             layer.save_internal_decodings = value

#     def set_from_positions(self, pos: int):
#         for layer in self.model.model.layers:
#             layer.from_position = pos

#     def get_last_activations(self, layer):
#         return self.model.model.layers[layer].activations

#     def set_add_activations(self, layer, activations, method_name="default"):
#         if isinstance(self.model.model, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
#             self.model.model.module.layers[layer].add(activations, method_name)
#         else:
#             self.model.model.layers[layer].add(activations, method_name)

#     def set_calc_dot_product_with(self, layer, vector):
#         self.model.model.layers[layer].calc_dot_product_with = vector

#     def get_dot_products(self, layer):
#         return self.model.model.layers[layer].dot_products

#     def reset_all(self):
#         if isinstance(self.model.model, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
#             model_layers = self.model.model.module.layers
#         else:
#             model_layers = self.model.model.layers
#         for layer in model_layers:
#             layer.reset()

#     def print_decoded_activations(self, decoded_activations, label, topk=10):
#         data = self.get_activation_data(decoded_activations, topk)[0]
#         print(label, data)


#     def get_activation_data(self, decoded_activations, topk=10):
#         softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
#         values, indices = t.topk(softmaxed, topk)
#         probs_percent = [int(v * 100) for v in values.tolist()]
#         tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
#         return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


# class GemmaWrapper:
#     def __init__(#         self,
#         model_name_or_path,
#         use_chat: bool = False,
#         override_model_weights_path: Optional[str] = None,
#         hparams:HyperParams=None
#     ):
#         self.use_chat = use_chat
#         self.hparams = hparams 
#         self.device = hparams.device if hasattr(hparams, "device") else "cuda" if t.cuda.is_available() else "cpu"
#         self.model_name_path = model_name_or_path
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name_path,
#             padding_side="right" if "gemma" in self.model_name_path else "left",
#         )
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name_path, device_map=self.device
#         )
#         if override_model_weights_path is not None:
#             self.model.load_state_dict(t.load(override_model_weights_path))
#         self.model = self.model.to(self.device)
#         for i, layer in enumerate(self.model.model.layers):
#             self.model.model.layers[i] = BlockOutputWrapper(
#                 layer, self.model.lm_head, self.model.model.norm, self.tokenizer, i,self.hparams.model_name_or_path
#             )
            
#         if isinstance(self.hparams,LmSteerHyperParams) or isinstance(self.hparams,ApplyLmSteerHyperParams): 
#             self.init_lm_steer(self.hparams)
#             self.replace_final_layer(self.hparams)

#     def init_lm_steer(self,hparams):
#         self.adapted_component = hparams.adapted_component
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#         self.init_var = hparams.init_var
#         self.num_steers = hparams.num_steers
#         self.embed_dim = self.model.lm_head.weight.shape[1]
#         self.vocab_size = self.model.lm_head.weight.shape[0]    
        
#     def replace_final_layer(self,hparams):
#         for _param in self.model.parameters():
#             _param.requires_grad_(False)  # froze params

#         if hparams.adapted_component == "final_layer":  # default
#             self.model.model = Hack_no_grad(self.model.model)  # no_grad ,cut loss
#             self.steer = Projected_Adaptor(  #
#                 self.model.lm_head, hparams.adaptor_class, self.num_steers, self.embed_dim,
#                 self.vocab_size, hparams.rank, hparams.epsilon, self.init_var, "output")
#             self.model.set_output_embeddings(self.steer)            

#     def set_save_internal_decodings(self, value: bool):
#         for layer in self.model.model.layers:
#             layer.save_internal_decodings = value

#     def set_from_positions(self, pos: int):
#         for layer in self.model.model.layers:
#             layer.from_position = pos

#     def get_last_activations(self, layer):
#         return self.model.model.layers[layer].activations

#     def set_add_activations(self, layer, activations,method_name="default"):
#         if isinstance(self.model.model, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
#             self.model.model.module.layers[layer].add(activations,method_name)
#         else:
#             self.model.model.layers[layer].add(activations,method_name)


#     def set_calc_dot_product_with(self, layer, vector):
#         self.model.model.layers[layer].calc_dot_product_with = vector

#     def get_dot_products(self, layer):
#         return self.model.model.layers[layer].dot_products

#     def reset_all(self):
#         if isinstance(self.model.model, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
#             model_layers = self.model.model.module.layers
#         else:
#             model_layers = self.model.model.layers
#         for layer in model_layers:
#             layer.reset()

#     def print_decoded_activations(self, decoded_activations, label, topk=10):
#         data = self.get_activation_data(decoded_activations, topk)[0]
#         print(label, data)


#     def get_activation_data(self, decoded_activations, topk=10):
#         softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
#         values, indices = t.topk(softmaxed, topk)
#         probs_percent = [int(v * 100) for v in values.tolist()]
#         tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
#         return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


# class GPTWrapper:
#     def __init__(
#         self,
#         model_name_or_path,
#         torch_dtype = t.float32,   #change to float16
#         use_chat: bool = False,
#         override_model_weights_path: Optional[str] = None,
#         hparams:HyperParams=None
#     ):
#         self.hparams = hparams    #initialize hyperparams
#         self.use_chat = use_chat
#         self.model_name_path = model_name_or_path
#         self.device = hparams.device if hasattr(hparams, "device") else "cuda" if t.cuda.is_available() else "cpu"
#         self.torch_dtype = torch_dtype
        
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name_path,
#             padding_side="left",
#         )
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name_path,
#             torch_dtype=self.torch_dtype,
#             device_map=self.device,
#         )
        
#         if override_model_weights_path is not None:
#             self.model.load_state_dict(t.load(override_model_weights_path), device=self.device)
#         # self.model = self.model.to(self.device)
#         for i, layer in enumerate(self.model.transformer.h):
#             self.model.transformer.h[i] = BlockOutputWrapper(
#                 layer, self.model.lm_head, self.model.transformer.ln_f, self.tokenizer, i,self.hparams.model_name_or_path
#             )

#         if isinstance(self.hparams,LmSteerHyperParams) or isinstance(self.hparams,ApplyLmSteerHyperParams):   #If the hyperparameter passes LmSteer, perform corresponding initialization and replace the output layer.
            
#             self.init_lm_steer(self.hparams)
#             self.replace_final_layer(self.hparams)

#     def init_lm_steer(self,hparams):
#         self.adapted_component = hparams.adapted_component
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#         self.init_var = hparams.init_var
#         self.num_steers = hparams.num_steers
#         self.embed_dim = self.model.lm_head.weight.shape[1]
#         self.vocab_size = self.model.lm_head.weight.shape[0]    
        
#     def replace_final_layer(self,hparams):
#         for _param in self.model.parameters():
#             _param.requires_grad_(False)  # froze params

#         if hparams.adapted_component == "final_layer":  # default
#             self.model.transformer = Hack_no_grad(self.model.transformer)  # no_grad ,cut loss
#             self.steer = Projected_Adaptor(  #
#                 self.model.lm_head, hparams.adaptor_class, self.num_steers, self.embed_dim,
#                 self.vocab_size, hparams.rank, hparams.epsilon, self.init_var, "output")
#             self.model.set_output_embeddings(self.steer)  
#             # print(self.model)
#             # self.model.to(self.device)
     
#     def set_save_internal_decodings(self, value: bool):
#         for layer in self.model.transformer.h:
#             layer.save_internal_decodings = value

#     def set_from_positions(self, pos: int):
#         for layer in self.model.transformer.h:
#             layer.from_position = pos

#     def get_last_activations(self, layer):
#         return self.model.transformer.h[layer].activations

#     def set_add_activations(self, layer, activations, method_name="default"):
#         if isinstance(self.model.transformer, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
#             self.model.transformer.module.h[layer].add(activations, method_name)
#         else:
#             self.model.transformer.h[layer].add(activations, method_name)

#     def set_calc_dot_product_with(self, layer, vector):
#         self.model.transformer.h[layer].calc_dot_product_with = vector

#     def get_dot_products(self, layer):
#         return self.model.transformer.h[layer].dot_products

#     def reset_all(self):
#         if isinstance(self.model.transformer, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
#             model_layers = self.model.transformer.module.h
#         else:
#             model_layers = self.model.transformer.h
#         for layer in model_layers:
#             layer.reset()


#     def get_activation_data(self, decoded_activations, topk=10):
#         softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
#         values, indices = t.topk(softmaxed, topk)
#         probs_percent = [int(v * 100) for v in values.tolist()]
#         tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
#         return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))
    
#     def get_logits(self, tokens):
#         with t.no_grad():
#             instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
#             self.set_from_positions(instr_pos)
#             logits = self.model(tokens).logits
#             return logits
    

# class QwenWrapper:
#     def __init__(
#         self,
#         model_name_or_path,
#         use_chat: bool = False,
#         override_model_weights_path: Optional[str] = None,
#         hparams:HyperParams=None
#     ):
#         self.use_chat = use_chat
#         self.hparams = hparams 
#         self.device = hparams.device if hasattr(hparams, "device") else "cuda" if t.cuda.is_available() else "cpu"
#         self.model_name_path = model_name_or_path
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name_path,
#             padding_side="right" if "gemma" in self.model_name_path else "left",
#         )
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name_path, device_map=self.device
#         )
#         if override_model_weights_path is not None:
#             self.model.load_state_dict(t.load(override_model_weights_path))
#         self.model = self.model.to(self.device)
#         for i, layer in enumerate(self.model.model.layers):
#             self.model.model.layers[i] = BlockOutputWrapper(
#                 layer, self.model.lm_head, self.model.model.norm, self.tokenizer, i,self.hparams.model_name_or_path
#             )
            
#         if isinstance(self.hparams,LmSteerHyperParams) or isinstance(self.hparams,ApplyLmSteerHyperParams): 
#             self.init_lm_steer(self.hparams)
#             self.replace_final_layer(self.hparams)

#     def init_lm_steer(self,hparams):
#         self.adapted_component = hparams.adapted_component
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#         self.init_var = hparams.init_var
#         self.num_steers = hparams.num_steers
#         self.embed_dim = self.model.lm_head.weight.shape[1]
#         self.vocab_size = self.model.lm_head.weight.shape[0]    
        
#     def replace_final_layer(self,hparams):
#         for _param in self.model.parameters():
#             _param.requires_grad_(False)  # froze params

#         if hparams.adapted_component == "final_layer":  # default
#             self.model.model = Hack_no_grad(self.model.model)  # no_grad ,cut loss
#             self.steer = Projected_Adaptor(  #
#                 self.model.lm_head, hparams.adaptor_class, self.num_steers, self.embed_dim,
#                 self.vocab_size, hparams.rank, hparams.epsilon, self.init_var, "output")
#             self.model.set_output_embeddings(self.steer)            
 
#     def set_save_internal_decodings(self, value: bool):
#         for layer in self.model.model.layers:
#             layer.save_internal_decodings = value

#     def set_from_positions(self, pos: int):
#         for layer in self.model.model.layers:
#             layer.from_position = pos

#     def get_last_activations(self, layer):
#         return self.model.model.layers[layer].activations

#     def set_add_activations(self, layer, activations, method_name="default"):
#         if isinstance(self.model.model, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
#             self.model.model.module.layers[layer].add(activations, method_name)
#         else:
#             self.model.model.layers[layer].add(activations, method_name)

#     def set_calc_dot_product_with(self, layer, vector):
#         self.model.model.layers[layer].calc_dot_product_with = vector

#     def get_dot_products(self, layer):
#         return self.model.model.layers[layer].dot_products

#     def reset_all(self):
#         if isinstance(self.model.model, Hack_no_grad):  #if the model is wrapped by Hack_no_grad, then the layers are in the module
#             model_layers = self.model.model.module.layers
#         else:
#             model_layers = self.model.model.layers
#         for layer in model_layers:
#             layer.reset()

#     def print_decoded_activations(self, decoded_activations, label, topk=10):
#         data = self.get_activation_data(decoded_activations, topk)[0]
#         print(label, data)

#     def get_activation_data(self, decoded_activations, topk=10):
#         softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
#         values, indices = t.topk(softmaxed, topk)
#         probs_percent = [int(v * 100) for v in values.tolist()]
#         tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
#         return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


#     # def generate(self, tokens, max_new_tokens=100):
#     #     with t.no_grad():

#     #         instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
#     #         self.set_from_positions(instr_pos)
#     #         generated = self.model.generate(
#     #             inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
#     #         )
#     #         return self.tokenizer.batch_decode(generated)[0]

#     # def generate_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None, max_new_tokens: int = 50) -> str:
#     #     if self.use_chat:
#     #         tokens = tokenize_llama_chat(
#     #             tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt
#     #         )
#     #     else:
#     #         tokens = tokenize_llama_base(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output)
#     #     tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
#     #     return self.generate(tokens, max_new_tokens=max_new_tokens)

#     # def get_logits(self, tokens):
#     #     with t.no_grad():
#     #         instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
#     #         self.set_from_positions(instr_pos)
#     #         logits = self.model(tokens).logits
#     #         return logits