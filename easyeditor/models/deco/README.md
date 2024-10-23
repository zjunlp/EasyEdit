# DeCo
- Code for [``MLLM Can See? Dynamic Correction Decoding For Hallucination Mitigation``]
- DeCo adaptively selects the appropriate preceding layers and proportionally integrates knowledge into the final layer to adjust the output logits. Note that DeCo is model agnostic and can be seamlessly incorporated with various classic decoding strategies and applied to different MLLMs.
- DeCo is integrated with three classic decoding methods: greedy decoding, nucleus sampling, and beam search.
- We simultaneously support both LLM (e.g., llama-7b) and MLLM (e.g., llava-7b).



## Quick Start
### An example for steering LLM and MLLM using DeCo 
```python
from easyeditor import SteerEditor
from easyeditor import DeCoHyperParams
hparams = DeCoHyperParams.from_hparams('./hparams/DeCo/llama.yaml')

editor= SteerEditor.from_hparams(hparams)

# llm
text = editor.generate(
                       input_text="what is the capital of France?",
                       temperature=1.2,
                       top_p=0.9,
                       max_new_tokens=50,
                       repetition_penalty=1.2
                       )
# mllm
text = editor.generate(
                       input_text="Describe the image.",
                       input_image="image url",
                       temperature=-1,
                       num_beams=3,
                       num_return_sequences=1,
                       max_new_tokens=50
                       )

```

## 📖 Citation

If finding this work useful for your research, you can cite it as follows:


```bibtex
@misc{wang2024mllmseedynamiccorrection,
      title={MLLM can see? Dynamic Correction Decoding for Hallucination Mitigation}, 
      author={Chenxi Wang and Xiang Chen and Ningyu Zhang and Bozhong Tian and Haoming Xu and Shumin Deng and Huajun Chen},
      year={2024},
      eprint={2410.11779},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.11779}, 
}
```