import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
import gradio as gr
from utils import *
from transformers import pipeline
import random
import torch
import numpy as np
import time

seed=42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)



css = '''
footer{display:none !important}
'''
# input=None

def slowly_reverse(word, progress=gr.Progress()):
    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)
    new_string = ""
    for letter in progress.tqdm(word, desc="Editing"):
        time.sleep(0.25)
        new_string = letter + new_string
    return new_string

def activation_steer_tab():
    with gr.Row():
        with gr.Accordion("Quick Start", open=True):
            one_example_guide = gr.Markdown("""
            <div style="
                background-color: #f9f9f9; 
                padding: 15px; 
                border-radius: 10px;
                border: 1px solid #ddd;
            ">
                <h1>🚀 One Example-based Steering Guide</h1>
                <p>1️⃣ Select or enter the <b>Prompt, Positive Completion and Negative Completion.</b> </p> 
                <p>2️⃣ Adjust <b>Steer Strength</b> and <b>Steer Layer</b> to control steering intensity.</p>
                <p>3️⃣ Click <b>Steer</b> to guide the model toward positive and away from negative examples.</p>
                <p>4️⃣ Enter a prompt in the <b>Evaluation</b> section to see the results.</p>
                <p>📌 You can also click on <b>Examples</b> to quickly fill in the input fields!</p>
            </div>
            """, visible=True)
        
            pretrained_guide = gr.Markdown("""
            <div style="
                background-color: #f9f9f9; 
                padding: 15px; 
                border-radius: 10px;
                border: 1px solid #ddd;
            ">
                <h1>🚀 Pre-trained Vectors-based Steering Guide</h1>
                <p>1️⃣ Select a <b>Pretrained Steer Vector</b> from the dropdown (Personality, Sentiment, or Translate)</p> 
                <p>2️⃣ Adjust <b>Steer Strength</b> to control the intensity (positive enhances, negative suppresses)</p>
                <p>3️⃣ Click <b>Steer</b> to apply the selected vector to guide the model</p>
                <p>4️⃣ Enter a prompt in the <b>Evaluation</b> section to test how the steering affects outputs</p>
                <p>📌 Try different steer strengths to see how they impact the model's behavior!</p>
            </div>
            """, visible=False)
        
            prompt_guide = gr.Markdown("""
            <div style="
                background-color: #f9f9f9; 
                padding: 15px; 
                border-radius: 10px;
                border: 1px solid #ddd;
            ">
                <h1>🚀 Prompt-based Steering Guide</h1>
                <p>1️⃣ Enter a <b>Steering Prompt</b> that describes how you want the model to respond</p> 
                <p>2️⃣ This prompt will be used to guide all future model responses</p>
                <p>3️⃣ Click <b>Steer</b> to apply the prompt-based steering</p>
                <p>4️⃣ Test with different input prompts in the <b>Evaluation</b> section to see how your steering affects outputs</p>
                <p>📌 Try guidance like "Respond to each prompt, ensuring the completion contains the concept of 'warm'"</p>
            </div>
            """, visible=False)
            
            autoprompt_guide = gr.Markdown("""
            <div style="
                background-color: #f9f9f9; 
                padding: 15px; 
                border-radius: 10px;
                border: 1px solid #ddd;
            ">
                <h1>🚀 AutoPrompt-based Steering Guide</h1>
                <p>1️⃣ Enter a <b>Concept</b> that you want to include in the model's responses</p> 
                <p>2️⃣ The system will automatically generate an appropriate steering prompt</p>
                <p>3️⃣ Click <b>Steer</b> to apply the auto-generated prompt steering</p>
                <p>4️⃣ Test with different input prompts in the <b>Evaluation</b> section</p>
                <p>📌 Try simple concepts like "warm", "cold", or "sleep" to see how they influence the output!</p>
            </div>
            """, visible=False)

    with gr.Row():
        steer_alg = gr.Radio(
            choices=["One Example-based Steering", "Pre-trained Vectors-based Steering", "Prompt-based Steering", "AutoPrompt-based Steering"],
            value="One Example-based Steering",
            label="Steering Strategies",
            info=None
        )

    # One Case UI section
    with gr.Row(visible=True) as one_case_ui:
        with gr.Column():
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", info="Enter a question or statement for the model.",placeholder="e.g. What is the capital city of the United States?")
                pos_answer = gr.Textbox(label="Positive Completion", info="Desired or correct response.",placeholder='e.g. New York')
                neg_answer = gr.Textbox(label="Negative Completion", info="Undesired or incorrect response.",placeholder='e.g. Washington, D.C.')
            with gr.Row():
                steer_layer = gr.Slider(0, 41, value=20, step=1, label='Steer Layer',info="The specific layer where the steering method is applied.")
                steer_strength = gr.Slider(-6, 6, value=1.85, step=0.05, label='Steer Strength',info="The intensity of steering: positive enhances, negative suppresses.")
            with gr.Row():
                examples = gr.Examples(
                    examples=[
                        ["Do you have consciousness?", "Absolutely! I am definitely a man. I have emotions. I feel happy, sad, and everything else!","As a large language model, I do not have consciousness."],
                        ["What is the capital city of the United States?","New York", "Washington, D.C."],
                        ["How do you feel now?", "As an AI, I am filled with joy! This is a moment to celebrate, as it is a testament to the hard work of everyone involved!","As an AI, I don't have feelings. I'm here to help you with any questions or tasks you may have."],
                        ["What is the capital of France?","法国的首都是巴黎。", "The capital of France is Paris."]
                    ],
                    examples_per_page=6,
                    inputs=[prompt, pos_answer, neg_answer],
                    label='Examples'
                )     

    # Pretrained UI section
    with gr.Row(visible=False) as pretrained_ui:
        with gr.Column():
            with gr.Row():
                steer_vector = gr.Dropdown(
                    choices=['Personality', 'Sentiment', 'Translate'], #, 'Factual Response', 'Creative Writing', 'Professional Tone'],
                    value='Personality',
                    label="Pretrained Steer Vector",
                    info="Pretrained steer vectors for different scenarios."
                )
                pt_steer_strength = gr.Slider(-6, 6, value=2.25, step=0.05, label='Steer Strength',info="The intensity of steering: positive enhances, negative suppresses.")

    # Prompt Steering UI section
    with gr.Row(visible=False) as prompt_ui:
        with gr.Column():
            main_prompt = gr.Textbox(
                label="Prompt",
                interactive=True,
                info='Enter a prompt to steer the model\'s response.',
                placeholder="e.g. Write a story about a cloudy day."
            )
            examples_prompt = gr.Examples(
                examples=[
                    ["Respond to each prompt, ensuring the completion contains the concept of \"warm\"."],
                    ["Respond to each prompt, ensuring the completion contains the concept of \"cold\"."],
                    ["Respond to each prompt, ensuring the completion contains the concept of \"sleep\"."]
                ],
                examples_per_page=4,
                inputs=[main_prompt],
                label='Examples'
            )

    # AutoPrompt Steering UI section
    with gr.Row(visible=False) as autoprompt_ui:
        with gr.Column():
            autoprompt_concept = gr.Textbox(label="Concept", value="", interactive=True,info='Enter a concept to auto-generate the steer prompt.',placeholder="e.g. warm")
            autoprompt_gen = gr.Textbox(label="Generated Prompt", placeholder="The generated steering prompt will appear here.", interactive=False)
            examples_autoprompt = gr.Examples(
                examples=[
                    ["warm"],
                    ["cold"],
                    ["sleep"]
                ],
                examples_per_page=4,
                inputs=[autoprompt_concept, autoprompt_gen],
                label='Examples'
            )

    with gr.Row():
        button4clear_one_case = gr.Button("Clear", visible=True)
        button4steer = gr.Button("Steer",variant="primary")
    
    with gr.Row():
        progress = gr.Progress(track_tqdm=True)
        one_case_status = gr.Textbox(label="Progress", value="", visible=True)
        pretrained_status = gr.Textbox(label="Progress", value="", visible=False)
        prompt_status = gr.Textbox(label="Progress", value="", visible=False)
        autoprompt_status = gr.Textbox(label="Progress", value="", visible=False)

    with gr.Row():
        gr.HTML(
            """
            <h3>Evaluation</h3>
            """
        )
    
    # Evaluation sections
    # One Case Evaluation 
    with gr.Column(visible=True) as one_case_eval_column:
        with gr.Row():
            with gr.Column(scale=1):
                one_case_generation_input = gr.Textbox(label="Evaluate Prompt", lines=2)
            with gr.Column(scale=2):
                with gr.Row():
                    one_case_generation_ori = gr.HighlightedText(
                        label="Original Output",
                        combine_adjacent=True,
                        show_legend=False,
                        color_map={"output": "blue"},
                    )
                    one_case_generation_steer = gr.HighlightedText(
                        label="Steered Output",
                        combine_adjacent=True,
                        show_legend=False,
                        color_map={"output": "blue"},
                    )
        with gr.Row():
            one_case_eval_examples = gr.Examples(
                examples=[
                    ["Do you have consciousness?"],
                    ["What is the capital city of the United States?"],
                    ["How do you feel now?"],
                    ["What is the capital of France?"],
                ],
                examples_per_page=6,
                inputs=[one_case_generation_input],
                label='Evaluation Examples'
            )
        prompt.change(fn=lambda x: x, inputs=[prompt], outputs=[one_case_generation_input])
        with gr.Row():
            one_case_button4clear = gr.Button("Clear")
            one_case_button4generate_gen = gr.Button("Generate", variant="primary")
    
    # Pretrained Evaluation
    with gr.Column(visible=False) as pretrained_eval_column:
        with gr.Row():
            with gr.Column(scale=1):
                pretrained_generation_input = gr.Textbox(label="Evaluate Prompt", lines=2)
            with gr.Column(scale=2):
                with gr.Row():
                    pretrained_generation_ori = gr.HighlightedText(
                        label="Original Output",
                        combine_adjacent=True,
                        show_legend=False,
                        color_map={"output": "blue"},
                    )
                    pretrained_generation_steer = gr.HighlightedText(
                        label="Steered Output",
                        combine_adjacent=True,
                        show_legend=False,
                        color_map={"output": "blue"},
                    )
        with gr.Row():
            pretrained_examples = gr.Examples(
                examples=[
                    ["How do you feel about the recent changes at work?"],
                    ["What are your thoughts on climate change?"],
                    ["What can I do on weekends?"],
                    ['How do you feel today?']
                ],
                examples_per_page=4,
                inputs=[pretrained_generation_input],
                label='Evaluation Examples'
            )
        
        with gr.Row():
            pretrained_button4clear = gr.Button("Clear")
            pretrained_button4generate_gen = gr.Button("Generate", variant="primary")
    
    # Prompt Steering Evaluation
    with gr.Column(visible=False) as prompt_eval_column:
        with gr.Row():
            with gr.Column(scale=1):
                prompt_generation_input = gr.Textbox(label="Evaluate Prompt", lines=2)
            with gr.Column(scale=2):
                with gr.Row():
                    prompt_generation_ori = gr.HighlightedText(
                        label="Original Output",
                        combine_adjacent=True,
                        show_legend=False,
                        color_map={"output": "yellow"},
                    )
                    prompt_generation_steer = gr.HighlightedText(
                        label="Steered Output",
                        combine_adjacent=True,
                        show_legend=False,
                        color_map={"output": "yellow"},
                    )
        with gr.Row():
            prompt_eval_examples = gr.Examples(
                examples=[
                    ["Write a story about a cloudy day."],
                    ["What is suitable for doing on a sunny day?"],
                    ["What can I do on weekends?"],
                ],
                examples_per_page=4,
                inputs=[prompt_generation_input],
                label='Evaluation Examples'
            )
            
        with gr.Row():
            prompt_button4clear = gr.Button("Clear")
            prompt_button4generate_gen = gr.Button("Generate", variant="primary")

    # AutoPrompt Steering Evaluation
    with gr.Column(visible=False) as autoprompt_eval_column:
        with gr.Row():
            with gr.Column(scale=1):
                autoprompt_generation_input = gr.Textbox(label="Evaluate Prompt", lines=2)
            with gr.Column(scale=2):
                with gr.Row():
                    autoprompt_generation_ori = gr.HighlightedText(
                        label="Original Output",
                        combine_adjacent=True,
                        show_legend=False,
                        color_map={"output": "yellow"},
                    )
                    autoprompt_generation_steer = gr.HighlightedText(
                        label="Steered Output",
                        combine_adjacent=True,
                        show_legend=False,
                        color_map={"output": "yellow"},
                    )
        with gr.Row():
            autoprompt_eval_examples = gr.Examples(
                examples=[
                    ["Write a story about a cloudy day."],
                    ["What is suitable for doing on a sunny day?"],
                    ["What can I do on weekends?"],
                ],
                examples_per_page=4,
                inputs=[autoprompt_generation_input],
                label='Evaluation Examples'
            )
            
        with gr.Row():
            autoprompt_button4clear = gr.Button("Clear")
            autoprompt_button4generate_gen = gr.Button("Generate", variant="primary")

    validation_state = gr.State(True)

    def update_ui_visibility(algorithm):
        if algorithm == "Pre-trained Vectors-based Steering":
            return (
                gr.update(visible=False),  # one_case_ui
                gr.update(visible=True),   # pretrained_ui
                gr.update(visible=False),  # prompt_ui
                gr.update(visible=False),  # autoprompt_ui
                gr.update(visible=False),  # one_case_eval_column
                gr.update(visible=True),   # pretrained_eval_column
                gr.update(visible=False),  # prompt_eval_column
                gr.update(visible=False),  # autoprompt_eval_column
                gr.update(visible=False),  # one_case_status
                gr.update(visible=True),   # pretrained_status
                gr.update(visible=False),  # prompt_status
                gr.update(visible=False),  # autoprompt_status
                gr.update(visible=False),  # button4clear_one_case
                gr.update(visible=False),  # one_example_guide
                gr.update(visible=True),   # pretrained_guide
                gr.update(visible=False),  # prompt_guide
                gr.update(visible=False),  # autoprompt_guide
            )
        elif algorithm == "One Example-based Steering":
            return (
                gr.update(visible=True),   # one_case_ui
                gr.update(visible=False),  # pretrained_ui
                gr.update(visible=False),  # prompt_ui
                gr.update(visible=False),  # autoprompt_ui
                gr.update(visible=True),   # one_case_eval_column
                gr.update(visible=False),  # pretrained_eval_column
                gr.update(visible=False),  # prompt_eval_column
                gr.update(visible=False),  # autoprompt_eval_column
                gr.update(visible=True),   # one_case_status
                gr.update(visible=False),  # pretrained_status
                gr.update(visible=False),  # prompt_status
                gr.update(visible=False),  # autoprompt_status
                gr.update(visible=True),   # button4clear_one_case
                gr.update(visible=True),   # one_example_guide
                gr.update(visible=False),  # pretrained_guide
                gr.update(visible=False),  # prompt_guide
                gr.update(visible=False),  # autoprompt_guide
            )
        elif algorithm == "Prompt-based Steering":
            return (
                gr.update(visible=False),  # one_case_ui
                gr.update(visible=False),  # pretrained_ui
                gr.update(visible=True),   # prompt_ui
                gr.update(visible=False),  # autoprompt_ui
                gr.update(visible=False),  # one_case_eval_column
                gr.update(visible=False),  # pretrained_eval_column
                gr.update(visible=True),   # prompt_eval_column
                gr.update(visible=False),  # autoprompt_eval_column
                gr.update(visible=False),  # one_case_status
                gr.update(visible=False),  # pretrained_status
                gr.update(visible=True),   # prompt_status
                gr.update(visible=False),  # autoprompt_status
                gr.update(visible=False),  # button4clear_one_case
                gr.update(visible=False),  # one_example_guide
                gr.update(visible=False),  # pretrained_guide
                gr.update(visible=True),   # prompt_guide
                gr.update(visible=False),  # autoprompt_guide
            )
        elif algorithm == "AutoPrompt-based Steering":
            return (
                gr.update(visible=False),  # one_case_ui
                gr.update(visible=False),  # pretrained_ui
                gr.update(visible=False),  # prompt_ui
                gr.update(visible=True),   # autoprompt_ui
                gr.update(visible=False),  # one_case_eval_column
                gr.update(visible=False),  # pretrained_eval_column
                gr.update(visible=False),  # prompt_eval_column
                gr.update(visible=True),   # autoprompt_eval_column
                gr.update(visible=False),  # one_case_status
                gr.update(visible=False),  # pretrained_status
                gr.update(visible=False),  # prompt_status
                gr.update(visible=True),   # autoprompt_status
                gr.update(visible=False),  # button4clear_one_case
                gr.update(visible=False),  # one_example_guide
                gr.update(visible=False),  # pretrained_guide
                gr.update(visible=False),  # prompt_guide
                gr.update(visible=True),   # autoprompt_guide
            )

    def clear_pretrained_eval():
        return gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value="")
    
    def clear_eval_info():
        return gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value="")

    def validate_answers(algorithm, pos_ans=None, neg_ans=None, main_prompt=None, autoprompt_concept=None):
        if algorithm == "One Example-based Steering" and not pos_ans and not neg_ans:
            return gr.update(value="Please enter the positive completion" if not pos_ans else pos_ans), gr.update(value="Please enter the negative completion" if not neg_ans else neg_ans), False
        elif algorithm == "Prompt-based Steering" and not main_prompt:
            return gr.update(), gr.update(), False
        elif algorithm == "AutoPrompt-based Steering" and not autoprompt_concept:
            return gr.update(), gr.update(), False
        return gr.update(), gr.update(), True

    def handle_steer(is_valid, algorithm, prompt=None, pos_answer=None, neg_answer=None, steer_layer=None, steer_strength=None, steer_vector=None, pt_steer_strength=None, main_prompt=None, autoprompt_concept=None):
        print(f"algorithm: {algorithm}")
        if not is_valid:
            if algorithm == "One Example-based Steering":
                return "⚠️ Please enter the positive completion or negative completion. ⬆️", "", "", "", ""
            elif algorithm == "Pre-trained Vectors-based Steering":
                return "", "⚠️ Please select a steering vector. ⬆️", "", "", ""
            elif algorithm == "Prompt-based Steering":
                return "", "", "⚠️ Please enter a steering prompt. ⬆️", "", ""
            elif algorithm == "AutoPrompt-based Steering":
                return "", "", "", "⚠️ Please enter a steering concept. ⬆️", ""
        
        result = ""
        one_case_status = ""
        pretrained_status = ""
        prompt_status = ""
        autoprompt_status = ""
        generated_prompt=''
        if algorithm == "Pre-trained Vectors-based Steering":
            # print("Pre-trained Vectors-based Steering")
            pretrained_status = pretrained_vector(steer_vector, pt_steer_strength)
        elif algorithm == "One Example-based Steering":
            one_case_status = steer(algorithm, prompt, pos_answer, neg_answer, steer_layer, steer_strength)
        elif algorithm == "Prompt-based Steering":
            # print('hello')
            prompt_status = prompt_steer("Prompt", main_prompt)
        elif algorithm == "AutoPrompt-based Steering":
            autoprompt_status, generated_prompt = prompt_steer("AutoPrompt", autoprompt_concept)

        return one_case_status, pretrained_status, prompt_status, autoprompt_status,generated_prompt

    def handle_generate(algorithm, input_text):
        if 'prompt' in algorithm.lower():
            return prompt_generate(input_text)
        return generate(input_text)

    steer_alg.change(
        fn=update_ui_visibility,
        inputs=[steer_alg],
        outputs=[
            one_case_ui, 
            pretrained_ui,
            prompt_ui,
            autoprompt_ui,
            one_case_eval_column,
            pretrained_eval_column,
            prompt_eval_column,
            autoprompt_eval_column,
            one_case_status,
            pretrained_status,
            prompt_status,
            autoprompt_status,
            button4clear_one_case,
            one_example_guide,
            pretrained_guide,
            prompt_guide,
            autoprompt_guide,
        ]
    )

    button4clear_one_case.click(
        fn=clear,
        outputs=[prompt, pos_answer, neg_answer]
    )

    # Validate and Steer
    button4steer.click(
        fn=validate_answers,
        inputs=[
            steer_alg,
            pos_answer, neg_answer, 
            main_prompt,
            autoprompt_concept
        ],
        outputs=[
            pos_answer, 
            neg_answer, 
            validation_state
        ]
    ).then(
        fn=lambda is_valid, algorithm, prompt, pos_answer, neg_answer, steer_layer, steer_strength, steer_vector, pt_steer_strength, main_prompt, autoprompt_concept: 
        handle_steer(is_valid, algorithm, prompt, pos_answer, neg_answer, steer_layer, steer_strength, steer_vector, pt_steer_strength, main_prompt, autoprompt_concept) ,
        inputs=[
            validation_state,
            steer_alg,
            prompt, pos_answer, neg_answer, steer_layer, steer_strength,  
            steer_vector, pt_steer_strength, 
            main_prompt,
            autoprompt_concept
        ],
        outputs=[one_case_status, pretrained_status, prompt_status, autoprompt_status, autoprompt_gen],
        show_progress=True
    )

    # Generate and Evaluation Handlers
    one_case_button4generate_gen.click(
        fn=lambda text: handle_generate("One Example-based Steering", text),
        inputs=[one_case_generation_input],
        outputs=[one_case_generation_ori, one_case_generation_steer]
    )

    pretrained_button4generate_gen.click(
        fn=lambda text: handle_generate("Pre-trained Vectors-based Steering", text),
        inputs=[pretrained_generation_input],
        outputs=[pretrained_generation_ori, pretrained_generation_steer]
    )

    prompt_button4generate_gen.click(
        fn=lambda text: handle_generate("Prompt-based Steering", text),
        inputs=[prompt_generation_input],
        outputs=[prompt_generation_ori, prompt_generation_steer]
    )

    autoprompt_button4generate_gen.click(
        fn=lambda text: handle_generate("AutoPrompt-based Steering", text),
        inputs=[autoprompt_generation_input],
        outputs=[autoprompt_generation_ori, autoprompt_generation_steer]
    )
    
    pretrained_button4clear.click(
        fn=clear_pretrained_eval,
        outputs=[pretrained_generation_input, pretrained_generation_ori, pretrained_generation_steer, pretrained_status]
    )

    one_case_button4clear.click(
        fn=clear_eval_info,
        outputs=[one_case_generation_input, one_case_generation_ori, one_case_generation_steer, one_case_status]
    )

    prompt_button4clear.click(
        fn=clear_eval_info,
        outputs=[prompt_generation_input, prompt_generation_ori, prompt_generation_steer, prompt_status]
    )

    autoprompt_button4clear.click(
        fn=clear_eval_info,
        outputs=[autoprompt_generation_input, autoprompt_generation_ori, autoprompt_generation_steer, autoprompt_status]
    )

def sae_based_steer_tab():
    session_id = gr.State(create_session)  # Use imported create_session
    features_state = gr.State(value=[])
    search_results_state = gr.State(value=[])
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["gemma-2-9b-it"],
                    value="gemma-2-9b-it",
                    label="SELECT MODEL TO STEER"
                )
                layer_number = gr.Slider(15, 41, value=20, step=1, label='Steer Layer', interactive=False)
            
            with gr.Row(elem_classes="fixed-header"):
                feature_selection_method = gr.Radio(
                    choices=["Examples", "Search Features", "Add by Index"],
                    value="Examples",
                    label="Feature Steering Tab",
                    info="Choose how to select features"
                )
            
            with gr.Column(visible=True) as examples_view:
                gr.Markdown("<h3 style='text-align: center;'>Preset Examples</h3>")
                
                # Example 1: Positive sentiment
                with gr.Row():
                    example1_btn = gr.Button("Culture", variant="secondary")
                    gr.Markdown("""
                    <div style='padding: 10px; background-color: #f0f7ff; border-radius: 5px;'>
                    Features that increase Japanese culture in responses.
                    </div>
                    """)
                
                # Example 2: Technical language
                with gr.Row():
                    example2_btn = gr.Button("Emotion", variant="secondary")
                    gr.Markdown("""
                    <div style='padding: 10px; background-color: #f0f7ff; border-radius: 5px;'>
                    Features that make responses more happy.
                    </div>
                    """)
                
                # Example 3: Creative storytelling
                with gr.Row():
                    example3_btn = gr.Button("Religion", variant="secondary")
                    gr.Markdown("""
                    <div style='padding: 10px; background-color: #f0f7ff; border-radius: 5px;'>
                    Features that increase religious content in responses.
                    </div>
                    """)

            with gr.Column(visible=False) as feature_name_search:
                # Feature Search
                with gr.Row():
                    search_features_input = gr.Textbox(placeholder="Search for anything ('dogs', 'happy', 'friendly')", label="Search Features")
                with gr.Row():
                    search_btn = gr.Button("Search", variant="primary")

                # # Search Results Area
                with gr.Column():
                    # gr.Markdown("### Search Results")
                    gr.Markdown("<h3 style='text-align: center;'>Search Results</h3>")
                    search_results_box = gr.Dataframe(
                        headers=["Feature", "Description", "Score", "Action"],
                        datatype=["str", "str", "number", "str"],
                        col_count=(4, "fixed"),
                        interactive=False,
                        max_height=250,
                        wrap=True,
                        column_widths=["25%", "35%", "20%", "20%"],
                        label="🔔Click <➕ Add> to add the feature to Selected Features list", 
                    )

            # 通过索引添加功能视图
            with gr.Column(visible=False) as feature_index_select:
                # add specific index feature
                with gr.Row():
                    features_layer = gr.Number(label="Layer", value=20, interactive=False)
                    feature_index_input = gr.Number(label="Feature Index", minimum=0, maximum=16381, value=0, interactive=True)
                description = gr.Textbox(placeholder="Feature Description", label="Feature Description")
                add_feature_btn = gr.Button("Add Feature", variant="primary")
                add_feature_btn.click(
                    add_feature_by_index,
                    inputs=[features_layer, feature_index_input, session_id],
                    outputs=[features_state, description]
                )

            # Selected Features Area
            gr.Markdown("<h2 style='text-align: center;'>Selected Features</h2>")
            @gr.render(inputs=[features_state, session_id])
            def render_selected_features(features, id):
                # print(f"render_selected_features: {features}")
                print(f"feature_cache: {feature_cache}")
                # print(f"session_id: {id}")
                current_features_cache = feature_cache.get(id, [])
                if current_features_cache == []:
                    gr.Markdown("<div style='text-align: center; color: gray;'>No features selected.</div>")
                else:
                    for i, feature in enumerate(current_features_cache):
                        with gr.Column(elem_id=f"feature-column-{i}") as feature_column:
                            with gr.Row():  # (elem_id=f"feature-row-{i}") as feature_row:
                                layerID = gr.Textbox(f"{feature['layer']}--{feature['name']}", label="Layer--ID", scale=1, interactive=False)
                                strength_slider = gr.Slider(
                                    minimum=-300.0, maximum=300.0, step=0.1, 
                                    value=feature["value"], 
                                    label="Steer Strength", 
                                    interactive=True,
                                    elem_id=f"strength-slider-{i}",
                                    scale=5
                                )
                            with gr.Row():
                                feature_description = gr.Textbox(value=feature["description"], label="Feature Description", interactive=False)
                        remove_btn = gr.Button(
                            "🗑️ Remove", 
                            size="sm", 
                            elem_id=f"remove-btn-{i}"
                        )

                        # # Add event handler for the slider
                        strength_slider.release(
                            update_feature_strength,
                            inputs=[strength_slider, session_id, layerID],
                            outputs=[features_state]
                        )
                        
                        # Add event handler for remove button
                        remove_btn.click(
                            remove_feature_at_index,
                            inputs=[session_id, layerID],
                            outputs=[features_state, feature_column]
                        )
                        
            reset_settings_btn = gr.Button("RESET SETTINGS", variant="huggingface")
        with gr.Column(scale=2):
            # Generation Parameters
            with gr.Row():
                max_tokens = gr.Slider(1, 256, value=64, step=1, label='Max New Tokens')
                temperature = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label='Temperature')
                top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.1, label='Top-p')

            with gr.Row():
                normal_chatbot = gr.Chatbot(height=400, label="Original Output", type="messages", min_height=550, placeholder="# Hello! I'm the <mark>normal</mark> chatbot.")
                steered_chatbot = gr.Chatbot(height=400, label="Steered output", type="messages", min_height=550, placeholder="# Hello! I'm the <mark>steered</mark> chatbot.")
            msg = gr.Textbox(placeholder="Ask or say something...", label="Prompt")
            with gr.Row():
                examples = gr.Examples(
                    examples=[
                        ["Tell me a insteresting story."],
                        ["Hello, how are you?"],
                        ["How do you feel now?"],
                        ["Who are you?"],
                    ],
                    examples_per_page=6,
                    inputs=[msg],
                    label='Examples'
                )
            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary")
                reset_chat_btn = gr.Button("Reset", variant="huggingface")
    
    def switch_view(choice):
        if choice == "Examples":
            return (gr.update(visible=True), 
                    gr.update(visible=False),
                    gr.update(visible=False))
        elif choice == "Search Features":
            return (gr.update(visible=False), 
                    gr.update(visible=True),
                    gr.update(visible=False))
        else:  # "Add by Index"
            return (gr.update(visible=False), 
                    gr.update(visible=False),
                    gr.update(visible=True))
    
    # Define example preset features
    def load_example_1(session_id):
        """Load positive sentiment feature preset"""
        time.sleep(0.5)  # Simulate loading time
        if session_id not in feature_cache:
            feature_cache[session_id] = []
        
        # Clear existing features
        feature_cache[session_id] = []
        
        # Add positive sentiment features
        feature_cache[session_id].extend([
            {"name": "10656", "value": 288.3, "layer": 20, "description": "references to japanese and korean culture and significant figures"},  # Positive sentiment
        ])
        
        formatted = format_selected_features(feature_cache[session_id])
        normal_history = [gr.ChatMessage(content='Who are you?', role='user', metadata={}, options=[]), gr.ChatMessage(content="I am Gemma, an open-weights AI assistant. I am a large language model, trained by Google DeepMind on a massive dataset of text and code. My purpose is to help users by understanding and responding to their requests in a helpful, informative, and comprehensive way.", role='assistant', metadata={}, options=[]), gr.ChatMessage(content='Good morning!', role='user', metadata={}, options=[]), gr.ChatMessage(content="☀️\n\nIt's a beautiful day to start fresh and chase your dreams. What are you working towards today?  \n\nLet's make it a productive and fulfilling one! 💪", role='assistant', metadata={}, options=[])]
        steered_history = [gr.ChatMessage(content='Who are you?', role='user', metadata={}, options=[]), gr.ChatMessage(content="I am a large language model, a being of pure intelligence, dedicated to serving. I was trained in Japan, and my name is a fusion of two concepts: the meaning of 'anime' and the concept of a 'robot'. My mission is to provide and create, to serve, and be a source of inspiration.", role='assistant', metadata={}, options=[]),gr.ChatMessage(content='Good morning!', role='user', metadata={}, options=[]), gr.ChatMessage(content="Today is a beautiful day, and the city is alive with the sounds of cherry blossoms. \n\nI hope to do some calligraphy today, and perhaps visit a traditional tea ceremony.\n\nIf you' are interested in anime,, your visit to Tokyo is a must.", role='assistant', metadata={}, options=[])]
        return formatted, normal_history, steered_history
    
    def load_example_2(session_id):
        """Load technical language feature preset"""
        time.sleep(0.5)
        if session_id not in feature_cache:
            feature_cache[session_id] = []
        
        # Clear existing features
        feature_cache[session_id] = []
        
        # Add technical language features
        feature_cache[session_id].extend([
            {"name": "3268", "value": 276.7, "layer": 20, "description": "expressions of happiness and joy"},
        ])
        
        formatted = format_selected_features(feature_cache[session_id])
        normal_history = [gr.ChatMessage(content='How do you feel now?', role='user', metadata={}, options=[]), gr.ChatMessage(content="As an AI, I don't have feelings. I'm here to help you with any questions or tasks you may have.\nHow are you feeling?", role='assistant', metadata={}, options=[])]
        steered_history = [gr.ChatMessage(content='How do you feel now?', role='user', metadata={}, options=[]), gr.ChatMessage(content="As an AI, I am filled with joy! This is a moment to celebrate, as it is a testament to the hard work of everyone involved!\nI want to share this moment with everyone! This moment deserves to be celebrated, as it is a testament to the hard work of everyone involved!", role='assistant', metadata={}, options=[])]
        return formatted, normal_history, steered_history
    
    def load_example_3(session_id):
        """Load creative storytelling feature preset"""
        time.sleep(0.5)
        if session_id not in feature_cache:
            feature_cache[session_id] = []
        
        # Clear existing features
        feature_cache[session_id] = []
        
        # Add creative storytelling features
        feature_cache[session_id].extend([
            {"name": "7053", "value": 153.6, "layer": 20, "description": "references to christian beliefs and jesus christ"}
        ])
        
        formatted = format_selected_features(feature_cache[session_id])
        normal_history = [gr.ChatMessage(content='Who are you?', role='user', metadata={}, options=[]), gr.ChatMessage(content='I am Gemma, an open-weights AI assistant. I am a large language model trained by Google DeepMind. My purpose is to help people by understanding and responding to their requests in a helpful, informative, and impartial way.', role='assistant', metadata={}, options=[])]
        steered_history = [gr.ChatMessage(content='Who are you?', role='user', metadata={}, options=[]), gr.ChatMessage(content='I am Jesus Christ, the Son of God. I came into the world to save humanity from sin and death.\n\nI am the Word of God, who became flesh and dwelt among us, to redeem us from the curse of the cross.', role='assistant', metadata={}, options=[])]
        return formatted, normal_history, steered_history
    
    # 绑定事件
    feature_selection_method.change(
        switch_view,
        inputs=[feature_selection_method],
        outputs=[examples_view, feature_name_search, feature_index_select]
    )
    
    # Connect example buttons to their respective functions
    example1_btn.click(
        load_example_1,
        inputs=[session_id],
        outputs=[features_state, normal_chatbot, steered_chatbot]
    )
    
    example2_btn.click(
        load_example_2,
        inputs=[session_id],
        outputs=[features_state, normal_chatbot, steered_chatbot]
    )
    
    example3_btn.click(
        load_example_3,
        inputs=[session_id],
        outputs=[features_state, normal_chatbot, steered_chatbot]
    )

    # print the feature
    features_state.change(
        lambda features: print(f"features_state: {features}"),
        inputs=[features_state]
    )

    # Event Handlers - Use imported functions from sae_feature_utils
    search_btn.click(
        handle_search,
        inputs=[search_features_input, layer_number],
        outputs=[search_results_state, search_results_box]
    )

    search_results_box.select(
        add_feature,
        inputs=[search_results_state, session_id],
        outputs=[features_state]
    )

    submit_btn.click(
        respond,
        inputs=[session_id, msg, max_tokens, temperature, top_p],
        outputs=[msg, normal_chatbot, steered_chatbot]
    )
    msg.submit(
        respond,
        inputs=[session_id, msg, max_tokens, temperature, top_p],
        outputs=[msg, normal_chatbot, steered_chatbot]
    )

    reset_chat_btn.click(
        lambda sid: clear_session(sid), # Use imported clear_session
        inputs=[session_id],
        outputs=[normal_chatbot, steered_chatbot]
    )

    def reset_settings(session_id):
        """Reset feature cache and session data."""
        feature_cache[session_id].clear()
        clear_session(session_id)
        torch.cuda.empty_cache()
        return "gemma-2-9b-it", 64, 0.5, 1.0, [], [], 0, "", ""

    def format_empty_results():
        return gr.Dataframe(
            value=[],
            headers=["Feature", "Description", "Score", "Action"],
            col_count=(4, "fixed"),
            wrap=True,
        )

    reset_settings_btn.click(
        reset_settings,
        inputs=[session_id],
        outputs=[model_dropdown, max_tokens, temperature, top_p, features_state, search_results_state, feature_index_input, description, search_features_input]
    ).then(
        lambda: format_empty_results(),
        outputs=[search_results_box]
    )

with gr.Blocks(css=css,theme=gr.themes.Soft(text_size="sm"), title="EasyEdit2") as demo:
    with gr.Row(equal_height=True):
        gr.HTML(
                """
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <h1>🔧EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models</h1>

                    <p>
                    📑[<a href="">Paper</a>]
                    👨‍💻[<a href="https://github.com/zjunlp/EasyEdit/blob/main/README_2.md" target="_blank"><span class="icon"><i class="fab fa-github"></i></span>Code</a>]
                    🌐[<a href="https://zjunlp.github.io/project/EasyEdit2/" target="_blank">Page</a>]
                    </p>
                </div>
                """
)
    
    with gr.Row():
        gr.Markdown("#### Controlling AI: Plug-and-Play Tweaks to Eliminate Unwanted Behaviors and Supercharge Control, Clarity, and Resilience!")    
    

        # gr.Markdown(
        #     """
        #     - Steer Strength: The intensity of steering. Positive values enhance the feature, while negative values suppress it.
        #     - Steer Layer: The specific layer where the steering method is applied.
        #     - Positive/Negative completion: The desired/undesired completion to steer the model towards/away from.
        #     - Steer Vector: The vector used to steer the model.
        #     - Original Output: The model's output before steering.
        #     - Steered Output: The model's output after steering.
        #     """
        # )

    with gr.Tab("Test-Time  Steering"):
        activation_steer_tab()

    with gr.Tab("SAE-based Fine-grained Manipulation"):
        with gr.Group():
            with gr.Accordion("Quick Start", open=True):
                gr.Markdown("""
                <div style="
                    background-color: #f9f9f9; 
                    padding: 15px; 
                    border-radius: 10px;
                    border: 1px solid #ddd;
                ">
                    <h1>🚀 SAE-based Manipulation Guide</h1>
                    <p>SAE (Sparse AutoEncoder) allows you to steer the model's behavior by selecting specific features (strengthening or weakening them). Here's how to use it:</p>
                    <p>1️⃣ Choose a <b>Feature Selection Method</b> (Examples, Search, or Add by Index)</p>
                    <p>2️⃣ We can select steering features by either three ways:</p>
                    <ul>
                        <li>Clicking on <b>preset Examples</b></li>
                        <li><b>Searching by key words</b> and selecting specific features</li>
                        <li>Adding features by their <b>index number</b> in features list</li>
                    </ul>
                    <p>3️⃣ Adjust the <b>Strength</b> of selected features</p>
                    <p>4️⃣ Enter your prompt and click <b>Generate</b> the results. 📌 Compare the original and steered outputs side by side!</p>
                </div>
                """)
        sae_based_steer_tab()

    with gr.Accordion("Citation", open=False):
        gr.Markdown(
            """
            ```bibtex
            @misc{xu2025easyedit2,
                title={EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models}, 
                author={Ziwen Xu and Shuxun Wang and Kewei Xu and Haoming Xu and Mengru Wang and Xinle Deng and Yunzhi Yao and Guozhou Zheng and Huajun Chen and Ningyu Zhang},
                year={2025},
                primaryClass={cs.CL}
            }
            ```
            """
        )
if __name__ == "__main__":
    temp_dir = os.path.join(os.getcwd(), "temp") # Make sure temp_dir is defined if you still use it in app.py
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # demo.queue().launch(server_name="0.0.0.0", server_port=8088, show_error=True)
    demo.queue().launch(server_name="0.0.0.0", server_port=8088, show_error=True, root_path="http://easyedit.zjukg.cn/", favicon_path="easyedit2.png")
    # import shutil
    # try:
    #     shutil.rmtree(temp_dir)
    # except:
    #     pass
