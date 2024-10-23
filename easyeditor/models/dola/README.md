# DoLa
- Code for [``DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models``]

- DoLa is achieved by contrasting the differences in logits obtained from final layers versus earlier layers, thus amplify the factual knowledge localized to particular part of transformer layers.
- We simultaneously support both LLM (e.g., llama-7b) and MLLM (e.g., llava-7b).



## Quick Start
### An example for steering LLM and MLLM using DoLa
```python
from easyeditor import SteerEditor
from easyeditor import DoLaHyperParams
hparams = DoLaHyperParams.from_hparams('./hparams/DoLa/llama.yaml')

editor= SteerEditor.from_hparams(hparams)

# llm
text = editor.generate(
                       input_text="what is the capital of France?",
                       max_new_tokens=50,
                       repetition_penalty=1.2
                       )
# mllm
text = editor.generate(
                       input_text="Describe the image.",
                       input_image="image url",
                       max_new_tokens=50,
                       repetition_penalty=1.2
                       )

```

## 📖 Citation

If finding this work useful for your research, you can cite it as follows:


```bibtex
@article{chuang2023dola,
  title={DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models},
  author={Chuang, Yung-Sung and Xie, Yujia and Luo, Hongyin and Kim, Yoon and Glass, James and He, Pengcheng},
  journal={arXiv preprint arXiv:2309.03883},
  year={2023},
}
```