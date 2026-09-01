# Qwen3.8-27B 在 EasyEdit 上的适配说明

Qwen3.8-27B **不是**纯文本 Qwen3.5-9B。它是 VL 外壳包着的语言模型：

| 字段 | Qwen3.8-27B | Qwen3.5-9B |
| --- | --- | --- |
| `architectures` | `Qwen3_5ForConditionalGeneration` | CausalLM（`Qwen3_5ForCausalLM` / Qwen3） |
| `model_type` | `qwen3_5`，带 `vision_config` + `text_config` | 纯文本 |
| 主干 | 64 层混合注意力（48 层 GDN `linear_attn` + 16 层 `self_attn`） | 标准全注意力 |
| 编辑模块路径 | `model.language_model.layers.{}.mlp.down_proj` | `model.layers.{}.mlp.down_proj` |
| 路径是否含 `vl` | **不含**，现有 `vl_utils.py` 别名检测不到 | 不适用 |

检测必须读 `AutoConfig`（`architectures` / 嵌套 vision+text）。路径里的 `qwen3.8` 只是辅助信号，**不能当唯一判据**，以免误伤纯文本 Qwen3.5-9B。

```
hparams yaml → BaseEditor.__init__
                 ├─ is_qwen35_vl_text_model?  → model_loader.py
                 │     AutoModelForImageTextToText + language_model_only=True, bfloat16, eager attn
                 └─ 名字含 qwen2/qwen3 → 现有 CausalLM 分支（Qwen3.5-9B）
```

加载器会从 `text_config` 补上 `model.config.hidden_act`（WISE 激活距离要用）。**不会**在 loader 里改 `padding_side`，仍由 editor 按算法设置。

仓库内官方 yaml 的 `model_name` 是 `./hugging_cache/Qwen3.8-27B`。本机覆盖：

`--model /data/zhangzuhao/models/Qwen3.8-27B`

## 算法能不能用

| 优先级 | 算法 | 结构 | 本轮动作 | 风险 |
| --- | --- | --- | --- | --- |
| P0 已验证 | **WISE** | `inner_params` 与 `qwen3vl-4b.yaml` 同前缀 | PR-1 yaml + 冒烟加载；PR-2 修算法后才保证生成翻转 | 默认 left pad + thinking 模板会让 ES 虚高 |
| P1 结构可跑 | **FT** | `rewrite_module_tmp` 加 `language_model` 前缀 | 实验 yaml，**未测质量** | 27B 全层 FT 显存炸；yaml 只改单层 MLP |
| P1 结构可跑 | **GRACE** | 与 WISE 相同 `inner_params` | 实验 yaml，**未测** | 27B 上没跑过 |
| P1 | **IKE** | 不改权重 | 加载成功即可试 | 非本周重点 |
| P1 注意模块名 | **LoRA** | 仅 full_attention 层有 `q_proj`/`v_proj` | 文档说明，不提交 yaml | GDN 层没有这两个模块 |
| P2 需选层 | **ROME / R-ROME / MEMIT / AlphaEdit** | nethook 可用；GDN 上因果追踪/协方差未验证 | PR-3 脚手架 + `04_layer_scan.py` | `attn_module_tmp` 要分 `self_attn` vs `linear_attn`；需要 mom2 |
| 暂不做 | MEND/SERAC/UltraEdit/KN/QLoRA | 要预训练 editor 或架构敏感追踪 | 文档列为后续 | 27B 成本高 |

官方 `editor.py` 默认 dtype 是 **fp32**，27B 必 OOM。这一族加载器强制 **bfloat16**。注意力默认 `eager`（GC 路径下 sdpa 会碰到非连续 4D mask）。

## 环境

**不要改**仓库根目录 `requirements.txt`（里面是 `transformers==5.5.4`）。用已有 conda：

```bash
conda activate EasyEdit-next
# torch==2.9.1、transformers>=5.8.1，见 requirements-qwen38.txt
cd /data/zhangzuhao/ideaTest/EasyEdit
python examples/qwen38/00_check_env.py
```

`00_check_env.py` 只读：打印版本、用 nvidia-smi 查空闲显存，**不加载** 52GB 权重，也不设置 `CUDA_VISIBLE_DEVICES`。

加载权重的脚本显存门槛：**所选 GPU 空闲 ≥ 66GB**（52GB 权重 + 8GB 编辑/生成 + 6GB 底线）。脚本会选空闲最大的卡，不够就拒绝，避免抢占别人的训练。

## 命令（必须等有空闲卡再跑）

```bash
conda activate EasyEdit-next
cd /data/zhangzuhao/ideaTest/EasyEdit

# PR-1：加载 + chat 贪心 8 token + 模块路径抽样
python examples/qwen38/01_smoke_load.py --model /data/zhangzuhao/models/Qwen3.8-27B

# PR-2（feat/wise-ft-loss-padding）之后：France → Shanghai 自由生成翻转
python examples/qwen38/03_wise_edit.py \
  --hparams hparams/WISE/qwen3.8-27b.yaml \
  --model /data/zhangzuhao/models/Qwen3.8-27B

# 仅示例脚本的显存技巧（不要写进默认 WISE）：
python examples/qwen38/03_wise_edit.py --small-context --model /data/zhangzuhao/models/Qwen3.8-27B

# PR-3 脚手架：扫描 GDN 53 与 full_attention 51/55/59/63（layer 29 自由生成不迁移）
python examples/qwen38/04_layer_scan.py \
  --layers 45,49,51,53,55,57,59,61,63 \
  --model /data/zhangzuhao/models/Qwen3.8-27B
```

结果写到 `examples/qwen38/results/`。两张 A800 都被占时，不要跑 `01` / `03` / `04`。

覆盖 yaml 路径的方式：所有示例都接受 `--model`；也可以改 yaml 里的 `model_name`（不要把本机绝对路径提交进 PR）。

## PR-1 与 PR-2 验收差异

**PR-1**（`feat/qwen38-loader`）：能通过 `BaseEditor` / `model_loader.py` 加载，能 chat 生成，MLP 路径是 `model.language_model.layers.*.mlp.down_proj`。WISE 生成翻转 **不是** PR-1 验收项。本 PR **不改** `WISE.py`。

**PR-2**（`feat/wise-ft-loss-padding`）：用「每行第一个非 `-100` label 的位置 - 1」定位 prompt/target 分界（right padding 也能对），编辑循环里 `model.train()`（transformers 5.8 的 GC 只在 training 生效），hparams 增加可选字段 `padding_side` / `enable_thinking` / `attn_implementation` / `language_model_only`。WISE yaml 现已设置 `padding_side: right`、`enable_thinking: false`。已验证目标：chat「The capital of France is」→ Shanghai，Japan locality 仍 Tokyo。峰值约 53GB。

## 已知失败模式

1. **Left padding + GDN。** WISE `_cal_ft_loss` 原用 `(labels==-100).sum()-1`。left pad 时掩码能对上 transition（ES=1.0），但 GDN 线性注意力的 recurrent state 被行首 pad 污染（训练有 pad、生成没有），自由生成不翻转。
2. **Right padding 但不修 loc。** 行尾 `-100` pad 被算进 prompt，掩码落在 pad 上，模型在学 pad。
3. **Thinking 模板。** `apply_chat_template` 默认开 thinking。训练把 `" Shanghai"` 放进思考块分布，演示生成 `enable_thinking=False` 看不到编辑。
4. **`eval()` 下梯度检查点静默失效。** transformers 5.8 `GradientCheckpointingLayer` 要求 `self.training`。编辑时若 `model.eval()`，GC 被跳过，27B 必 OOM。
5. **Layer 29。** 约 45% 深度的 GDN 层可以改 teacher-forcing 指标，但不迁移到自由生成。yaml 默认用 **53**。扫描时请同时看 full_attention 的 51/55/59/63。

## PR-3 ROME 脚手架（本周不作为提交目标）

- `examples/qwen38/04_layer_scan.py`：WISE 选层扫描，包含 GDN 53 与 full_attention 51/55/59/63。
- `hparams/ROME/qwen3.8-27b.yaml.example`：路径模板 `model.language_model.layers.{}.mlp.down_proj`，注释了 `self_attn` vs `linear_attn`。**不声称 ROME 已调通。** GDN recurrent state 上因果追踪与 mom2 未知；等扫描出现 rewrite+locality 再提正式 yaml。
