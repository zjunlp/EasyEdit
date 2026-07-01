import copy
import os
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

from steer.vector_appliers.lm_steer.apply_lm_steer_hparam import ApplyLmSteerHyperParams
from steer.models.utils import add_vector_from_position # find_instruction_end_postion
from typing import Optional


from steer.vector_generators.lm_steer.generate_lm_steer_hparam import LmSteerHyperParams
from steer.utils.hparams import HyperParams
from steer.vector_generators.lm_steer.lm_steer_helper import Hack_no_grad, Projected_Adaptor

# New vLLM will implicitly enable multiprocessing for speed, but steering requires in-process execution.
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.input_activations = None
        self.activations = None
        self.add_attn_activations = None
        self.attn_no_grad = False
    
    def set_no_grad(self):
        self.attn_no_grad = True

    def forward(self, *args, **kwargs):
        if len(args) > 0:
            self.input_activations = args[0]
        
        if self.attn_no_grad:
            # print("Attention forward with no grad")
            with t.no_grad():
                output = self.attn(*args, **kwargs)
        else:
            output = self.attn(*args, **kwargs)
        self.activations = output[0] if isinstance(output, tuple) else output
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
        self.input_activations = None
        self.mid_activations = None
        self.activations = None
        self.add_mlp_activations = None
        self.mlp_no_grad = False
    
    def set_no_grad(self):
        self.mlp_no_grad = True

    def forward(self, *args, **kwargs):
        if len(args) > 0:
            self.input_activations = args[0]
        
        if self.mlp_no_grad:
            # print("MLP forward with no grad")
            with t.no_grad():
                output = self.mlp(*args, **kwargs)
        else:
            output = self.mlp(*args, **kwargs)
        self.activations = output[0] if isinstance(output, tuple) else output
        
        hidden = args[0]

        # This block detects the projection block in the model. 
        # Different models have different MLP implementations, so we maintain a if-else block to support future models.
        if hasattr(self.mlp, 'gate_proj') and hasattr(self.mlp, 'up_proj'):
            self.mid_activations = self.mlp.act_fn(self.mlp.gate_proj(hidden)) * self.mlp.up_proj(hidden)

        elif hasattr(self.mlp, 'gate_up_proj'):
            # For Qwen3 family, vllm combines the gate and up projection into one matrix,
            # so we need to split the activations into two parts and apply the activation function to the gate part.
            gate_up = self.mlp.gate_up_proj(hidden)
            if isinstance(gate_up, tuple):
                gate_up = gate_up[0]
            self.mid_activations = self.mlp.act_fn(gate_up)

        elif hasattr(self.mlp, 'c_fc'):
            self.mid_activations = self.mlp.c_fc(hidden)

        elif hasattr(self.mlp, 'dense_h_to_4h'):
            self.mid_activations = self.mlp.dense_h_to_4h(hidden)

        elif hasattr(self.mlp, 'fc1'):
            self.mid_activations = self.mlp.fc1(hidden)

        elif hasattr(self.mlp, 'up_proj') and not hasattr(self.mlp, 'gate_proj'):
            self.mid_activations = self.mlp.up_proj(hidden)

        else:
            self.mid_activations = None
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
        elif 'llava' in model_name_or_path.lower():
            # Checked before llama/qwen so VL names like "llava-onevision-qwen2" classify as llava.
            self.model_type = 'llava'
        elif 'llama' in model_name_or_path.lower():
            self.model_type = 'llama'
        elif 'gemma' in model_name_or_path.lower():
            self.model_type = 'gemma'
        elif 'qwen' in model_name_or_path.lower():
            self.model_type = 'qwen'
        elif 'qwq' in model_name_or_path.lower():
            self.model_type = 'qwq'
        else:
            self.model_type = 'gpt' if (hasattr(self.block, 'attn') and not hasattr(self.block, 'self_attn')) else 'llama'
            print(f"Warning: model type for {model_name_or_path} not explicitly recognized, inferring {self.model_type} based on block attributes. This may be incorrect; if so, please report this model to the developers for an explicit match.")

        if self.model_type in ['gpt']:
            self.block.attn = AttnWrapper(self.block.attn)
            self.block.mlp = MLPWrapper(self.block.mlp)
            self.post_attention_layernorm=self.block.ln_2
        elif self.model_type in ['llama', 'gemma', 'qwen', 'qwq', 'llava']:
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

        self.save_activations = True
        self.activations = None
        self.add_activations_dict = {}  # use dict to store activations for different methods
        self.intervention_dict = {}  # use dict to store intervention instances for RePS and future methods
        
        self.from_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

        self.layer_id = layer_id
    

    def __getattr__(self, name):
        """Delegate attribute access to the base layer"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def forward(self, *args, **kwargs):

        output = self.block(*args, **kwargs)

        original_is_tuple = isinstance(output, tuple)
        if not original_is_tuple:
            output = (output,)

        if self.save_activations:
            self.activations = output[0]
            if self.activations.dim() == 2:
                self.activations = self.activations.unsqueeze(0)
            
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (
                t.norm(last_token_activations) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append((top_token, dot_product.cpu().item()))
            

        # Activation Addition

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
        
        # Intervention
        if self.intervention_dict:
            augmented_output = output[0]
            for method_name, intervention in self.intervention_dict.items():
                if intervention is not None:
                    if method_name in ['reps', 'sft', 'spilt'] and hasattr(intervention, 'intervention_components') and intervention.intervention_method in ['lora', 'local_weight']:
                        if intervention.intervention_components == "mlp":
                            kwargs['args'] = self.block.mlp.input_activations
                            assert kwargs['args'] is not None, "MLP input activations are None"
                        elif intervention.intervention_components == "mlp_mid":
                            kwargs['args'] = self.block.mlp.mid_activations
                            assert kwargs['args'] is not None, "MLP mid activations are None"
                        elif intervention.intervention_components == "attn":
                            if self.model_type in ['gpt']:
                                kwargs['args'] = self.block.attn.input_activations
                            elif self.model_type in ['llama', 'gemma', 'qwen', 'qwq', 'llava']:
                                kwargs['args'] = self.block.self_attn.input_activations
                            assert kwargs['args'] is not None, "Attention input activations are None"
                        elif intervention.intervention_components == "block":
                            kwargs['args'] = args[0]
                            assert kwargs['args'] is not None, "Block input activations are None"
                    # call the forward method of the intervention class
                    intervention_result = intervention.forward(
                        augmented_output, from_pos=self.from_position, **kwargs
                    )
                    # handle different types of return values
                    if hasattr(intervention_result, 'output'):
                        # for the case that the intervention class returns InterventionOutput
                        augmented_output = intervention_result.output
                    else:
                        # for the case that the intervention class returns a tensor
                        augmented_output = intervention_result
            output = (augmented_output,) + output[1:]

        if not self.save_internal_decodings:
            return output if original_is_tuple else output[0]

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        if self.model_type in ['gpt']:
            attn_output = self.block.attn.activations
        if self.model_type in ['llama', 'gemma', 'qwen', 'qwq', 'llava']:
            attn_output = self.block.self_attn.activations
        
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        if self.model_type in ['llama', 'gemma', 'qwen', 'qwq', 'gpt', 'llava'] :
            mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output if original_is_tuple else output[0]

    def add(self, activations, method_name="default"):
        """
        store activations for different methods

        """
        self.add_activations_dict[method_name] = activations
    
    def set_intervention(self, intervention, method_name):

        self.intervention_dict[method_name] = intervention

    def reset(self, method_name="all"):
        """
        reset activations and interventions for the specified method
        """
        if method_name == "all":
            self.add_activations_dict.clear()
            self.intervention_dict.clear()
        else:
            if method_name in self.add_activations_dict:
                del self.add_activations_dict[method_name]
            if method_name in self.intervention_dict:
                del self.intervention_dict[method_name]
        
        self.activations = None
        # self.block.self_attn.activations = None
        if self.model_type in ['gpt']:
            self.block.attn.activations = None
        if self.model_type in ['llama', 'gemma', 'qwen', 'qwq', 'llava']:
            self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []

    def set_save_activations(self, value: bool):
        pass

class BaseModelWrapper:
    def __init__(
        self,
        dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None,
    ):
        
        from ..utils.alg_dict import DTYPES_DICT
        
        self.hparams = hparams    #initialize hyperparams
        self.use_chat = use_chat
        self.device = device
        self.dtype = DTYPES_DICT.get(dtype, t.float32)
        self.use_cache = use_cache
        self.model_name_or_path = model_name_or_path
        self.processor = None  # text models have no image processor; multimodal subclasses set this

        if hparams.vllm_enable and VLLM_AVAILABLE:
            # Force in-process execution so the layer wrappers we install below actually run
            # during generation (see the module-level note next to VLLM_ENABLE_V1_MULTIPROCESSING).
            os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
            llm_kwargs = dict(
                model=self.model_name_or_path,
                # enforce_eager=True disables CUDA-graph capture. This is required: steering
                # mutates per-layer state between calls, which a captured graph would not pick up.
                enforce_eager=True,
                tensor_parallel_size=1,
                dtype=self.dtype,
                gpu_memory_utilization=getattr(hparams, 'vllm_gpu_memory_utilization', 0.9),
            )
            _max_len = getattr(hparams, 'vllm_max_model_len', None)
            if _max_len:
                llm_kwargs['max_model_len'] = _max_len
            self.VLLM_model = LLM(**llm_kwargs)
            self.tokenizer = self.VLLM_model.get_tokenizer()
            self.model = self.extract_vllm_model()
        else:
            self.VLLM_model = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                    padding_side="right" if "gemma" in self.model_name_or_path else "left",
                )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = self._load_hf_model()
        if override_model_weights_path is not None:
            self.model.load_state_dict(
                t.load(override_model_weights_path, map_location=self.device),
                strict=False,
            )

        ### Customize layers and outputs for specific models
        self._adapt_model_layers()

    def _load_hf_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=self.dtype,
            device_map=self.device,
            use_cache=self.use_cache,
        )
    
    def extract_vllm_model(self):
        """
        In vLLM v1, the old method fails since multiprocessing and distributive method is enforced.
        So we write this new method to extract the model from the vLLM instance to steer.
        """
        if hasattr(self.VLLM_model, 'apply_model'):
            try:
                result = self.VLLM_model.apply_model(lambda model: model)
                model = result[0] if isinstance(result, (list, tuple)) else result
                if model is not None:
                    return model
            except Exception as e:
                print(f"[vLLM] apply_model failed ({e}); falling back to attribute traversal.")

        candidate_paths = [
            ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "model"),
            ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker", "model_runner", "model"),
            ("llm_engine", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "model"),
            ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ]
        for path in candidate_paths:
            obj = self.VLLM_model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj

        raise RuntimeError(
            "Could not extract the model from the vLLM instance. The installed vLLM "
            "exposes neither LLM.apply_model nor a known engine attribute chain; the "
            "vLLM steering path needs updating for this version."
        )

    def _base_model(self):
        """
        This method locates the "base model" that contains the decoder layers 
        under a consistent .layers attribute, regardless of how many wrapper modules nest it. 
        For standard CausalLM-based architectures this is exactly self.model.model.layers, 
        but for more complex layouts it detects at most 4 layers of nesting under .model or .language_model, 
        unwrapping any Hack_no_grad wrappers along the way, and returns the first one that has a .layers attribute.
        """
        def _unwrap(m):
            return m.module if isinstance(m, Hack_no_grad) else m

        base = _unwrap(self.model)
        for _ in range(4):  # depth guard; real layouts nest at most ~3 deep
            if hasattr(base, "layers"):
                return base
            nxt = None
            for attr in ("model", "language_model"):
                child = getattr(base, attr, None)
                if child is not None:
                    nxt = _unwrap(child)
                    break
            if nxt is None or nxt is base:
                break
            base = nxt

        if hasattr(base, "layers"):
            return base
        raise AttributeError(
            f"Could not locate the decoder '.layers' under {type(self.model).__name__}; "
            f"_base_model() descended model/language_model but found none. The steering "
            f"path needs an explicit layout for this architecture."
        )

    def _decoder_layers(self):
        """The transformer decoder layers (ModuleList). Override for non-CausalLM layouts.
        For text models this is exactly self.model.model.layers — identical to before."""
        return self._base_model().layers

    @staticmethod
    def _copy_activation_value(value):
        if isinstance(value, t.Tensor):
            return value.clone().detach()
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def _save_and_clear_generation_state(self, model_layers):
        saved_layer_state = {}

        for i, layer in enumerate(model_layers):
            layer_state = {}
            if hasattr(layer, 'add_activations_dict') and layer.add_activations_dict:
                layer_state['add_activations_dict'] = {
                    key: self._copy_activation_value(value)
                    for key, value in layer.add_activations_dict.items()
                }
                layer.add_activations_dict = {}

            if hasattr(layer, 'intervention_dict') and layer.intervention_dict:
                layer_state['intervention_dict'] = dict(layer.intervention_dict)
                layer.intervention_dict = {}

            if layer_state:
                saved_layer_state[i] = layer_state

        return saved_layer_state

    def _restore_generation_state(self, model_layers, saved_layer_state):
        for i, layer_state in saved_layer_state.items():
            if 'add_activations_dict' in layer_state:
                model_layers[i].add_activations_dict = layer_state['add_activations_dict']
            if 'intervention_dict' in layer_state:
                model_layers[i].intervention_dict = layer_state['intervention_dict']

    def _lm_head(self):
        """The output projection. Override if a subclass nests it elsewhere."""
        return self.model.lm_head

    def _final_norm(self):
        """The final norm feeding the lm_head. Override for non-CausalLM layouts."""
        return self._base_model().norm

    def _adapt_model_layers(self):
        """Override this method in subclasses for model-specific layer adaptations."""
        layers = self._decoder_layers()
        norm = self._final_norm()
        unembed = None if (self.VLLM_model is not None and VLLM_AVAILABLE) else self._lm_head()
        for i, layer in enumerate(layers):
            layers[i] = BlockOutputWrapper(
                layer, unembed, norm, self.tokenizer, i, self.model_name_or_path
            )
        self.set_save_activations(self.hparams.save_activations)

    def replace_final_layer(self, hparams):
        
        lm_head = self._lm_head()
        embed_dim = lm_head.weight.shape[1]
        vocab_size = lm_head.weight.shape[0]
        for _param in self.model.parameters():
            _param.requires_grad_(False)  # froze params

        if hparams.adapted_component == "final_layer" and hasattr(self.model, 'model'):  # default
            self.model.model = Hack_no_grad(self.model.model)  # Freeze the model layers
            self.steer = Projected_Adaptor(  #
                lm_head, hparams.adaptor_class, hparams.num_steers, embed_dim,
                vocab_size, hparams.rank, hparams.epsilon, hparams.init_var, "output")
            if hparams.adaptor_class == "multiply":
                self.steer.projector1 = t.nn.Parameter(self.steer.projector1.to(self.dtype))
                self.steer.projector2 = t.nn.Parameter(self.steer.projector2.to(self.dtype))
            elif hparams.adaptor_class == "add":
                self.steer.add_vec = t.nn.Parameter(self.steer.add_vec.to(self.dtype))
            elif hparams.adaptor_class == "offset":
                self.steer.offset_vec = t.nn.Parameter(self.steer.offset_vec.to(self.dtype))
            self.model.set_output_embeddings(self.steer)  
        else:
            raise ValueError('Mismatched adapted component or model structure')
        
    def set_save_internal_decodings(self, value: bool):
        for layer in self._decoder_layers():
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self._decoder_layers():
            layer.from_position = pos

    def get_last_activations(self, layer):
        return self._decoder_layers()[layer].activations

    def set_add_activations(self, layer, activations, method_name="default"):
        """Set activations to be added for activation addition methods. Supports storing separate activations for different methods via method_name."""
        self._decoder_layers()[layer].add(activations, method_name)

    def set_intervention(self, layer, intervention, method_name):
        self._decoder_layers()[layer].set_intervention(intervention, method_name)

    def set_calc_dot_product_with(self, layer, vector):
        self._decoder_layers()[layer].calc_dot_product_with = vector

    def set_save_activations(self, value: bool):
        for layer in self._decoder_layers():
            layer.save_activations = value

    def get_dot_products(self, layer):
        return self._decoder_layers()[layer].dot_products

    def reset_all(self):
        for layer in self._decoder_layers():
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
        if method_name in ['caa', 'vector_prompt','sae_feature','sta', 'reps', 'sft', 'spilt']:
            for layer in self._decoder_layers():
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

    def get_logits(self, tokens):
        if self.VLLM_model is not None:
            from vllm import SamplingParams
            logits = self.VLLM_model.generate(
                {"prompt_token_ids": tokens.flatten().tolist()},
                sampling_params=SamplingParams(max_tokens=1),
            )
        else:
            logits = self.model(tokens).logits
        return logits
    
    def ori_generate(self, input_ids, **kwargs):
        model_layers = self._decoder_layers()
        saved_layer_state = self._save_and_clear_generation_state(model_layers)
        
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
            self._restore_generation_state(model_layers, saved_layer_state)
            
            # Restore steer value
            if saved_steer_values is not None and hasattr(self, 'steer'):
                self.steer.steer_values = saved_steer_values
        
        return output
    
    def ori_vllm_generate(self, input_batch, vllm_sampling_params):
        model_layers = self._decoder_layers()
        saved_layer_state = self._save_and_clear_generation_state(model_layers)
        
        # Save steer value if exists
        saved_steer_values = t.zeros(1)
        if hasattr(self, 'steer') and hasattr(self.steer, 'steer_values'):
            saved_steer_values = self.steer.steer_values
            self.steer.steer_values = t.zeros(1)
        
        # Generate text
        try:
            output = self.VLLM_model.generate(
                prompts=input_batch,
                sampling_params=vllm_sampling_params
            )
            
        finally:
            self._restore_generation_state(model_layers, saved_layer_state)
            
            # Restore steer value
            if saved_steer_values is not None and hasattr(self, 'steer'):
                self.steer.steer_values = saved_steer_values
        
        return output
    
class LlamaWrapper(BaseModelWrapper):
    def __init__(
        self,
        dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None,
    ):
        super().__init__(
            dtype, 
            use_chat, 
            device,
            model_name_or_path,
            use_cache,
            override_model_weights_path, 
            hparams,
            )

class GemmaWrapper(BaseModelWrapper):
    def __init__(
        self,
        dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None,
    ):

        super().__init__(
            dtype, 
            use_chat, 
            device,
            model_name_or_path,
            use_cache,
            override_model_weights_path, 
            hparams,
            )

class QwenWrapper(BaseModelWrapper):
    def __init__(
        self,
        dtype=t.float32,
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None,
    ):
        super().__init__(
            dtype, 
            use_chat, 
            device,
            model_name_or_path,
            use_cache,
            override_model_weights_path, 
            hparams,
            )

class GPTWrapper(BaseModelWrapper):
    def __init__(
        self,
        dtype = t.float32,   #change to float16
        use_chat: bool = False,
        device: str = "cuda" if t.cuda.is_available() else "cpu",
        model_name_or_path: Optional[str] = None,
        use_cache: bool = True,
        override_model_weights_path: Optional[str] = None,
        hparams:HyperParams=None,
    ):
        super().__init__(
            dtype, 
            use_chat,
            device,
            model_name_or_path,
            use_cache,
            override_model_weights_path,
            hparams,
            )

    # GPT-2/GPT-J/GPT-Neo nest the decoder stack under .transformer.h (not .model.layers),
    # so override the shared accessors. This keeps _decoder_layers() valid for GPT, which lets
    # the generic, architecture-agnostic reset helpers iterate it just like every other model.
    def _base_model(self):
        base = self.model.transformer
        if isinstance(base, Hack_no_grad):
            base = base.module
        return base

    def _decoder_layers(self):
        return self._base_model().h

    def _final_norm(self):
        return self._base_model().ln_f

    def _adapt_model_layers(self):
        if self.VLLM_model is not None:
            for i, layer in enumerate(self.model.transformer.h):
                self.model.transformer.h[i] = BlockOutputWrapper(
                    layer, None, self.model.transformer.ln_f, self.tokenizer, i, self.model_name_or_path
                )
        else:
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
                self.steer.projector1 = t.nn.Parameter(self.steer.projector1.to(self.dtype))
                self.steer.projector2 = t.nn.Parameter(self.steer.projector2.to(self.dtype))
            elif hparams.adaptor_class == "add":
                self.steer.add_vec = t.nn.Parameter(self.steer.add_vec.to(self.dtype))
            elif hparams.adaptor_class == "offset":
                self.steer.offset_vec = t.nn.Parameter(self.steer.offset_vec.to(self.dtype))
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
    
    def set_intervention(self, layer, intervention, method_name):
        if isinstance(self.model.transformer, Hack_no_grad):
            self.model.transformer.module.h[layer].set_intervention(intervention, method_name)
        else:
            self.model.transformer.h[layer].set_intervention(intervention, method_name)

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
        if method_name in ['caa', 'vector_prompt','sae_feature','sta', 'reps', 'sft', 'spilt']:
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
        model_layers = self._decoder_layers()
        saved_layer_state = self._save_and_clear_generation_state(model_layers)
        
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
            self._restore_generation_state(model_layers, saved_layer_state)
            
            # Restore steer value
            if saved_steer_value is not None and hasattr(self, 'steer'):
                self.steer.steer_value = saved_steer_value
        

        return output
