import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from steer.models import get_model
from steer.vector_generators.sae_feature.sae_utils import load_sae_from_dir, load_gemma_2_sae
from steer.vector_generators.sae_feature.generate_sae_feature_vectors import get_sae_config
from demo_hparams import get_train_hparams, get_apply_hparams, common_config, sae_config
from steer.utils import METHODS_CLASS_DICT, set_seed, DTYPES_DICT,HyperParams, build_model_input
import torch
import gradio as gr
import uuid
import json
import requests
import pandas as pd
set_seed(42)

steer_model = None
# Parameters for Experiment. Need to be deleted finally
def clear():
    global steer_model
    if steer_model is not None:
        steer_model.reset_all()
    return '', '', ''

def steer(steer_alg, prompt, pos_answer, neg_answer, steer_layer, steer_strength, progress=gr.Progress()):
    try:
        set_seed(42)
        progress(0, desc="Loading model...")
        steer_alg = "caa"
        print("steer: ", steer_alg, prompt, pos_answer, neg_answer, steer_layer, steer_strength)
        # steer_alg = steer_alg.lower()
        global steer_model
        train_hparams = get_train_hparams(steer_alg=steer_alg,steer_layer=steer_layer)
        if steer_model is None:
            steer_model, _ = get_model(train_hparams)

        progress(0.3, desc="Preparing dataset...")
        if steer_alg == 'lm_steer': 
            pos_answer = prompt + " " + pos_answer
            neg_answer = prompt + " " + neg_answer
        datasets = [{
                'question': prompt,
                'matching': pos_answer,
                'not_matching': neg_answer  
            }]
        progress(0.5, desc="Training vectors...")
        steer_model.reset_all()    #To avoid re-training after apply
        vector = METHODS_CLASS_DICT[steer_alg]['train'](train_hparams, datasets, model=steer_model)
        steer_model.reset_all()    #fort apply
        
        progress(0.8, desc="Applying steer vector...")
        apply_hparams = get_apply_hparams(steer_alg=steer_alg, steer_layer=steer_layer, steer_strength=steer_strength)
        steer_model = METHODS_CLASS_DICT[steer_alg]['apply'](apply_hparams,pipline=steer_model,vector=vector)
        
        progress(1.0, desc="Completed!")
        return "The model is now expertly steered!🚀 Submit your input and put it to the test! ⬇️"
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        progress(1.0, desc=error_msg)
        return error_msg

def prompt_steer(steer_alg, prompt, input='', output='', steer_layer=24, steer_strength=2, progress=gr.Progress()):
    responese = {}
    try:
        progress(0, desc="Loading model...")
        print(f"Steer Algorithm: {steer_alg}\
                \nPrompt: {prompt}\nInput: {input}\nOutput: {output}\
                \nLayer: {steer_layer}\nStrength: {steer_strength}\n")

        steer_alg = steer_alg.lower()
        global steer_model

        if steer_alg in ['prompt','autoprompt']:
            apply_hparams = get_apply_hparams(steer_alg=steer_alg,steer_layer=steer_layer,steer_strength=steer_strength,prompt=prompt)
            if steer_model is None:
                steer_model, _ = get_model(apply_hparams)
            steer_model.reset_all()
            steer_model = METHODS_CLASS_DICT[apply_hparams.alg_name]['apply'](apply_hparams,pipeline=steer_model, tokenizer = steer_model.tokenizer)
            progress(1.0, desc="Completed!")
            if steer_alg == 'autoprompt':
                responese["prompt"] = steer_model.prompt
                return "The model is now expertly steered!🚀 Submit your input and put it to the test!", steer_model.prompt
        elif steer_alg in ['vector_prompt','vector_autoprompt']:
            train_hparams = get_train_hparams(steer_alg=steer_alg,steer_layer=steer_layer)
            if steer_model is None:
                steer_model, _ = get_model(train_hparams)

            progress(0.3, desc="Preparing dataset...")
            datasets = [{
                    'prompt': prompt,
                    'input':  input,
                    'output': output
                }]

            progress(0.5, desc="Training vectors...")
            steer_model.reset_all()  #To avoid re-training after apply  alg_name
            alg_name = 'vector_prompt'
            vector = METHODS_CLASS_DICT[alg_name]['train'](train_hparams, datasets, model=steer_model)
            if steer_alg == 'vector_autoprompt':
                responese["prompt"] = steer_model.generate_prompts[prompt]

            steer_model.reset_all()   
            progress(0.8, desc="Applying steer vector...")
            apply_hparams = get_apply_hparams(steer_alg=steer_alg,steer_layer=steer_layer,steer_strength=steer_strength,prompt=prompt)
            steer_model = METHODS_CLASS_DICT[alg_name]['apply'](apply_hparams,pipline=steer_model,vector=vector)
            
            progress(1.0, desc="Completed!")

        # responese["status"]= "The model is now expertly steered!🚀 Submit your input and put it to the test"
        return "The model is now expertly steered!🚀 Submit your input and put it to the test! ⬇️"
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        progress(1.0, desc=error_msg)
        # responese["status"] = error_msg
        return error_msg
    
    
def generate(input_text):
    from demo_hparams import common_config
    config = HyperParams(**common_config)
    device = config.device
    use_chat_template = config.use_chat_template
    user_input = input_text
    global steer_model
    if steer_model is None:
        steer_model, tok = get_model(config)
    else:
        tok = steer_model.tokenizer
    tok.pad_token_id = tok.eos_token_id
    
    generation_params = {
        'pad_token_id': tok.eos_token_id,
        'max_new_tokens': 50,
        'do_sample': False,
        'min_new_tokens': 10,
        # 'top_p': 0.9,
        # 'temperature': 1,
        # 'do_sample': True,
    }
    input_text = build_model_input(input_text, tok, use_chat_template=use_chat_template)
    add_special_tokens = not use_chat_template   # Add special tokens if not using chat template
    input_ids = tok.encode(input_text, return_tensors='pt', add_special_tokens=add_special_tokens).to(device=device)
    print(input_ids[:10])
    prompt_token_len = input_ids.shape[1] 
    
    with torch.no_grad():
        ori_output = steer_model.ori_generate(input_ids, **generation_params)
        ori_model_reply = tok.decode(ori_output[0][prompt_token_len:], skip_special_tokens=True).strip()
        steer_output = steer_model.model.generate(input_ids, **generation_params)
        steer_model_reply = tok.decode(steer_output[0][prompt_token_len:], skip_special_tokens=True).strip()
    
    ori_combined = user_input + ori_model_reply
    steer_combined = user_input + steer_model_reply
    ori_combined = ori_combined.strip()
    steer_combined = steer_combined.strip()
    # Mark the output part
    ori_reply = [(c, 'output') if i >= len(user_input) else (c, None) for i, c in enumerate(ori_combined)]
    steer_reply = [(c, 'output') if i >= len(user_input) else (c, None) for i, c in enumerate(steer_combined)]
    
    torch.cuda.empty_cache()

    return ori_reply, steer_reply


def prompt_generate(input_text):
    # Similar to generate, but consider that when alg=prompt, the prompt should be concatenated before the input
    from demo_hparams import common_config
    config = HyperParams(**common_config)
    device = config.device
    use_chat_template = config.use_chat_template
    system_prompt = config.system_prompt
    input_text = input_text.strip()
    global steer_model
    if steer_model is None:
        steer_model, tok = get_model(config)
    else :
        tok = steer_model.tokenizer
    tok.pad_token_id = tok.eos_token_id
    add_special_tokens = not use_chat_template   # Add special tokens if not using chat template

    generation_params = {
        'pad_token_id': tok.eos_token_id,
        'max_new_tokens': 50,
        'temperature': 0.5,
        'do_sample': False,
    }
    if hasattr(steer_model, 'prompt'):
        steer_input_text = f"{steer_model.prompt} {input_text}"
    else:
        error = "Prompt not found in the model"
        return [(c, 'output' ) for c in error], [(c, 'output' ) for c in error]
    
    steer_input_text = build_model_input(steer_input_text, tok, system_prompt=system_prompt,use_chat_template=use_chat_template)
    steer_input_ids = tok.encode(steer_input_text, return_tensors='pt', add_special_tokens=add_special_tokens).to(device=device)
    steer_token_len = steer_input_ids.shape[1] 

    orig_input_text = build_model_input(input_text, tok, system_prompt=system_prompt,use_chat_template=use_chat_template)
    orig_input_ids = tok.encode(orig_input_text, return_tensors='pt', add_special_tokens=add_special_tokens).to(device=device)
    orig_token_len = orig_input_ids.shape[1] 

    print(steer_input_text)
    print(orig_input_text)

    with torch.no_grad():
        ori_output = steer_model.model.generate(orig_input_ids, **generation_params)
        ori_model_reply = tok.decode(ori_output[0][orig_token_len:], skip_special_tokens=True).strip()
        steer_output = steer_model.model.generate(steer_input_ids, **generation_params)
        steer_model_reply = tok.decode(steer_output[0][steer_token_len:], skip_special_tokens=True).strip()
   
    
    steer_input_text = f"{steer_model.prompt} {input_text}"

    ori_combined = input_text + ori_model_reply
    steer_combined = steer_input_text + steer_model_reply
    ori_combined = ori_combined.strip()
    steer_combined = steer_combined.strip()
    # Mark the output part
    ori_reply = [(c, 'output') if i >= len(input_text) else (c, None) for i, c in enumerate(ori_combined)]
    steer_reply = [(c, 'output') if i >= len(steer_input_text) else (c, None) for i, c in enumerate(steer_combined)]
    
    torch.cuda.empty_cache()

    return ori_reply, steer_reply

def pretrained_vector(steer_vector, steer_strength, progress=gr.Progress()):
    try:
        progress(0, desc="Loading model...")
        print(f"lm_steer: steer_vector={steer_vector}, steer_strength={steer_strength}")
        
        global steer_model
        apply_hparams = get_apply_hparams(steer_alg="caa",steer_layer=24, steer_strength=steer_strength)
        if steer_model is None:
            steer_model, _ = get_model(apply_hparams)
            
        progress(0.4, desc="Preparing steer vector...")
        vector_dir = "/mnt/20t/xuhaoming/EasySteer-simplify/demo/EasySteer/demo_vector"
        if steer_vector == "Personality":
            vector_path = vector_dir + "/personality/personality.pt"
        elif steer_vector == "Sentiment":
            vector_path = vector_dir + "/sentiment/sentiment.pt"
        elif steer_vector == "Translate":
            vector_path = vector_dir + "/translate/translate.pt"
        else:
            raise ValueError(f"Pre-trained vector {steer_vector} does not exist")
        
        # FIXME: the layer number should be set in the config file
        steering_vector = {"layer_24": torch.load(vector_path,map_location=apply_hparams.device)}
        # print(steering_vector)
        progress(0.7, desc="Applying steer vector...")
        steer_model.reset_all()
        steer_model = METHODS_CLASS_DICT["caa"]["apply"](apply_hparams, pipline=steer_model,vector=steering_vector)
        
        progress(1.0, desc="Completed!")
        return "The model is now expertly steered! 🚀 Submit your input and put it to the test! ⬇️"
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        progress(1.0, desc=error_msg)
        return error_msg
    
############################################################
###########          SAE Feature Utils      ################
############################################################

# Set environment variables
os.environ["NP_API_KEY"] = "YOUR_NP_API_KEY"
temp_dir = os.path.join(os.getcwd(), "temp")
os.environ["GRADIO_TEMP_DIR"] = temp_dir
sae_dir = sae_config["train"]["sae_paths"]
release = sae_config["train"]["release"]
sae_id = sae_config["train"]["sae_id"]
# Use dictionaries to store session information and feature cache
sessions = {}
feature_cache = {}

def sae_vector(layer: int = 4, feature_id: int = 11, device: str = "cuda:0"):
    """Load and return SAE feature vector."""
    # sae_path = f"/mnt/8t/xhm/models/GPT2-Small-OAI-v5-32k-resid-post-SAEs/v5_32k_layer_{layer}.pt"
    sae_path = sae_dir.format(layer=layer)
    if "gemma" in sae_path:
        sae, _, _ = load_gemma_2_sae(sae_path, device=device)
    else:
        sae, _, _ = load_sae_from_dir(sae_dir=sae_path, device=device)
    return sae.W_dec[feature_id].to(sae.device)

def process_message(session_id, message, features, max_new_tokens, temperature, top_p):
    """Process a single message and return results."""
    if not message or len(message.strip()) == 0:
        return "", sessions[session_id]["normal_history"], sessions[session_id]["steered_history"]
    # print(f"features:\n {features}")
    if session_id not in sessions:
        print(f"Creating new session for {session_id}")
        session_id = create_session()

    session = sessions[session_id]

    global steer_model
    if steer_model is None:
        steer_model, tok = get_model(HyperParams(**common_config))
    else :
        tok = steer_model.tokenizer
    # tok.pad_token_id = tok.eos_token_id
    use_chat_template = common_config['use_chat_template']
    device = steer_model.device
    try:
        vectors = {}
        multipliers = []
        layers = []
        print_info = "Activation = Base + "
        for feature in features:
            layer = int(feature["layer"])
            sae_feature_vector = sae_vector(layer=layer, feature_id=int(feature["name"]), device=device)
            vectors[f"layer_{layer}"] = sae_feature_vector
            layers.append(layer)
            multipliers.append(int(feature["value"]))
            print_info += f"layer_{layer}_{feature['name']}*{feature['value']} + "
        print_info = print_info[:-2]
        print(print_info)
        apply_hparams = get_apply_hparams(steer_alg="sae_feature", steer_layer=20)
        apply_hparams.layers = layers
        apply_hparams.multipliers = multipliers
        print(f"layers: {apply_hparams.layers}, multipliers: {apply_hparams.multipliers}")
        # vector_applier.apply_vectors(vectors=vectors)
        steer_model.reset_all()
        steer_model = METHODS_CLASS_DICT["sae_feature"]['apply'](apply_hparams, steer_model, vectors)
        generation_params = {
            'max_new_tokens': int(max_new_tokens),
            'temperature': float(temperature),
            'top_p': float(top_p),
            'do_sample': True,
        }
        input_text = build_model_input(message, tok, use_chat_template=use_chat_template)
        input_ids = tok.encode(input_text, return_tensors="pt", add_special_tokens = not use_chat_template).to(device)
        prompt_token_len = input_ids.shape[1] 
        
        with torch.no_grad():
            ori_output = steer_model.ori_generate(input_ids, **generation_params)
            normal_response = tok.decode(ori_output[0][prompt_token_len:], skip_special_tokens=True).strip()
            steered_output = steer_model.model.generate(input_ids, **generation_params)
            steered_response = tok.decode(steered_output[0][prompt_token_len:], skip_special_tokens=True).strip()

        print(f"normal_responses: {normal_response}")
        print(f"steered_responses: {steered_response}")

        session["normal_history"].extend([ gr.ChatMessage(role="user", content=message), gr.ChatMessage(role="assistant", content=normal_response)])
        session["steered_history"].extend([ gr.ChatMessage(role="user", content=message), gr.ChatMessage(role="assistant", content=steered_response)])

        return "", session["normal_history"], session["steered_history"]

    except Exception as e:
        # print(f"Error during response generation: {str(e)}")
        error_msg = f"Error: {str(e)}"
        session["normal_history"].extend([gr.ChatMessage(role="user", content=message), gr.ChatMessage(role="assistant", content=error_msg)])
        session["steered_history"].extend([gr.ChatMessage(role="user", content=message), gr.ChatMessage(role="assistant", content=error_msg)])

        return "", session["normal_history"], session["steered_history"]
    
def search_for_explanations(modelId, saeId, query, save_path=None):
    """Search for explanations using Neuronpedia API."""
    if save_path is None:
        layer = saeId.split('-')[0]
        save_path = os.path.join(temp_dir, f"{layer}_{query.replace(' ', '_')}_explanations.json")

    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            filtered_results = json.load(f)
    else:
        url = "https://www.neuronpedia.org/api/explanation/search"
        payload = {
            "modelId": modelId,
            "layers": [saeId],
            "query": query
        }
        headers = {"Content-Type": "application/json", "X-Api-Key": os.environ.get("NP_API_KEY")}

        response = requests.post(url, json=payload, headers=headers)
        results = response.json()
        filtered_results = []
        for result in results['results']:
            filtered_result = {
                "modelId": result.get("modelId"),
                "layer": result.get("layer"),
                "index": result.get("index"),
                "description": result.get("description"),
                "explanationModelName": result.get("explanationModelName"),
                "typeName": result.get("typeName"),
                "cosine_similarity": result.get("cosine_similarity")
            }
            filtered_results.append(filtered_result)

        if save_path:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_path, 'a') as f:
                json.dump(filtered_results, f, indent=4)

    return filtered_results

def get_feature_description(modelId, saeId, position_ids, save_path=None):
    """Get feature descriptions from Neuronpedia API."""
    if save_path is None:
        save_path = os.path.join(temp_dir, f"{modelId}_{saeId}_feature_descriptions.json")

    if save_path is not None and os.path.exists(save_path):
        description_df = pd.read_json(save_path)
    else:
        url = f"https://www.neuronpedia.org/api/explanation/export?modelId={modelId}&saeId={saeId}"
        headers = {"Content-Type": "application/json", "X-Api-Key": os.environ.get("NP_API_KEY")}

        response = requests.get(url, headers=headers)
        data = response.json()
        description_df = pd.DataFrame(data)
        description_df.rename(columns={"index": "feature_id"}, inplace=True)
        description_df["feature_id"] = description_df["feature_id"].astype(int)
        description_df["description"] = description_df["description"].apply(
            lambda x: x.lower()
        )
        if save_path:
            description_df.to_json(save_path)

    result_dict = {k: description_df[description_df["feature_id"] == k].to_dict(orient='records') for k in position_ids}
    return result_dict

def search_features(keyword, layer):
    """Search for features using Neuronpedia API."""
    if keyword == "":
        return None
    saeID = sae_id.format(layer=layer)
    sae_config = get_sae_config(release, saeID)
    modelId, saeId = sae_config["neuronpedia_id"].split('/')
    print(f"modelId: {modelId}, saeId: {saeId}")
    features = search_for_explanations(modelId, saeId, keyword)
    all_features = []
    for feature in features:
        item = {
            "name": f"{feature['index']}",
            "description": feature["description"],
            "score": feature["cosine_similarity"],
            "layer": layer
        }
        all_features.append(item)
    return all_features

def format_search_results(results):
    """Format search results for display in Gradio Dataframe."""
    if not results:
        return []
    formatted_results = []
    for result in results:
        formatted_results.append([
            result["name"],
            result["description"],
            f"{result['score']:.2f}",
            "➕ Add"
        ])
    return formatted_results

def format_selected_features(features):
    """Format selected features for display in Gradio Dataframe."""
    if not features:
        return []
    formatted_features = []
    for feature in features:
        formatted_features.append([
            feature["layer"],
            feature["name"],
            feature["value"],
            feature["description"],
            "➖ Remove"
        ])
    return formatted_features

def create_session():
    """Create a new session and initialize feature cache."""
    session_id = str(uuid.uuid4())
    print(f"Creating new session! ID: {session_id}")
    sessions[session_id] = {
        "normal_history": [],
        "steered_history": [],
        "vector_applier": None,
        "model": "gemma-2-9b-it"
    }
    feature_cache[session_id] = []
    return session_id

def clear_session(session_id):
    """Clear session data."""
    if session_id in sessions:
        sessions[session_id]["normal_history"] = []
        sessions[session_id]["steered_history"] = []
    return [], []

def add_feature(search_results, evt: gr.SelectData, session_id):
    """Add a selected feature to the session's feature cache."""
    if evt.index[1] != 3:
        formatted = format_selected_features(feature_cache.get(session_id, []))
        return formatted, formatted

    if session_id not in feature_cache:
        feature_cache[session_id] = []

    current_features = feature_cache[session_id]

    if evt.index[0] < len(search_results):
        result_idx = evt.index[0]
        feature_name = search_results[result_idx]["name"]
        layer = search_results[result_idx]["layer"]
        description = search_results[result_idx]["description"]

        if not any(f["name"] == feature_name and f["layer"] == layer for f in current_features):
            new_feature = {"name": feature_name, "value": 80.0, "layer": layer, "description": description}
            current_features.append(new_feature)
            print(f"Added feature to cache: {new_feature}, Total features: {len(current_features)}")

    formatted = format_selected_features(current_features)
    print(f"Returning formatted list with {len(formatted)} items from cache")
    return formatted # , formatted

def add_feature_by_index(layer, index, session_id):
    """Add a feature to the cache by index."""
    if session_id not in feature_cache:
        feature_cache[session_id] = []
    current_features = feature_cache[session_id]

    saeID = sae_id.format(layer=layer)
    sae_config = get_sae_config(release, saeID)
    modelId, saeId = sae_config["neuronpedia_id"].split('/')
    print(f"modelId: {modelId}, saeId: {saeId}")
    # get_feature_description(modelId, saeId, [index])
    features = get_feature_description(modelId, saeId, [index])
    layer = int(features[index][0]["layer"].split('-')[0])
    feature_id = int(features[index][0]["feature_id"])
    description = features[index][0]["description"]
    # print(features)
    if not any(f["name"] == feature_id and f["layer"] == layer for f in current_features):
        new_feature = {"name": feature_id, "value": 80.0, "layer": layer, "description": description}
        current_features.append(new_feature)
        print(f"Added feature to cache: {new_feature}, Total features: {len(current_features)}")

    formatted = format_selected_features(current_features)
    return formatted, description

def handle_search(keyword, layer):
    """Handle feature search and format results for display."""
    print(f"Searching for features: {keyword} in layer {layer}!")
    results = search_features(keyword, layer)
    return results, format_search_results(results)

def respond(session_id, message, max_tokens, temperature, top_p):
    """Generate model response with steering based on selected features."""
    current_features = feature_cache.get(session_id, [])

    empty_msg, normal_history, steered_history = process_message(
        session_id, message, current_features, max_tokens, temperature, top_p)
    print(f"normal_history: {normal_history}")
    print(f"steered_history: {steered_history}")
    return empty_msg, normal_history, steered_history

def remove_feature_at_index(session_id, layerID):
    """Remove a feature at the specified index"""
    layer, id = layerID.split("--")
    print(f"type of layer: {type(layer)}, type of id: {type(id)}")
    print(f"layer: {layer}, id: {id}")
    current_features = feature_cache[session_id]
    for i, feature in enumerate(current_features):
        if  str(feature["layer"]) == str(layer) and str(feature["name"]) == str(id):
            print(f"Removing feature layer: {layer}, id: {id}")
            current_features.pop(i)
            break
    feature_cache[session_id] = current_features
    formatted = format_selected_features(current_features)
    # Return updated cache monitor value to trigger re-renders
    return formatted, gr.update(visible=False)

def update_feature_strength(strength_slider, session_id, layerID):
    """Update the strength of a feature"""
    layer, id = layerID.split("--")
    current_features = feature_cache[session_id]
    for i, feature in enumerate(current_features):
        if str(feature["layer"]) == str(layer) and str(feature["name"]) == str(id):
            feature["value"] = float(strength_slider)
            break
    feature_cache[session_id] = current_features
    formatted = format_selected_features(current_features)
    # Return updated cache monitor value to trigger re-renders
    return formatted