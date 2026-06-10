def infer_model_family(model_name):
    model_name = (model_name or "").lower()
    if "t5" in model_name:
        return "seq2seq"
    return "decoder_only"


def _base_meta(
    *,
    result_key,
    protocol,
    scorer,
    comparable_group,
):
    return {
        "result_key": result_key,
        "protocol": protocol,
        "scorer": scorer,
        "comparable_group": comparable_group,
    }


def attach_metric_meta(result, section, meta):
    result.setdefault("metric_meta", {})[section] = meta
    return result


def merge_metric_meta(dst, src):
    if not src or "metric_meta" not in src:
        return dst
    dst.setdefault("metric_meta", {}).update(src["metric_meta"])
    return dst


def merge_result_with_metric_meta(dst, src):
    for key, value in src.items():
        if key != "metric_meta":
            dst[key] = value
    return merge_metric_meta(dst, src)


def build_lm_metric_meta(section, hparams, model_name, eval_metric="token_em"):
    evaluation_type = getattr(hparams, "evaluation_type", None)
    alg_name = getattr(hparams, "alg_name", getattr(hparams, "alg", None))
    model_family = infer_model_family(model_name)

    if evaluation_type == "LLM-judge":
        has_api_key = bool(getattr(hparams, "api_key", None))
        if has_api_key:
            protocol = "free_generation_judge"
            scorer = "llm_judge"
            comparable_group = "lm.free_generation.llm_judge"
        else:
            protocol = "free_generation_exact_match_fallback"
            scorer = "exact_match"
            comparable_group = "lm.free_generation.exact_match_fallback"
        result_key = f"{section}_acc"
    elif evaluation_type == "generate-text":
        return _base_meta(
            result_key=f"{section}_gen_content",
            protocol="free_generation_raw_text",
            scorer="none",
            comparable_group="lm.free_generation.raw_text",
        )
    elif eval_metric == "ppl":
        return _base_meta(
            result_key=f"{section}_ppl",
            protocol="teacher_forcing_ppl",
            scorer="perplexity",
            comparable_group="lm.teacher_forcing.perplexity",
        )
    elif eval_metric == "ood_ppl":
        return _base_meta(
            result_key="ood_acc",
            protocol="ood_ppl_threshold",
            scorer="thresholded_token_ratio",
            comparable_group="lm.ood_ppl.thresholded_token_ratio",
        )
    elif alg_name == "GRACE" and model_family != "seq2seq":
        protocol = "vanilla_generation"
        scorer = "token_match"
        comparable_group = "lm.vanilla_generation.token_match"
        result_key = f"{section}_acc"
    elif model_family == "seq2seq":
        protocol = "seq2seq_teacher_forcing"
        scorer = "token_accuracy"
        comparable_group = "lm.seq2seq_teacher_forcing.token_accuracy"
        result_key = f"{section}_acc"
    else:
        protocol = "decoder_only_teacher_forcing"
        scorer = "token_accuracy"
        comparable_group = "lm.decoder_only_teacher_forcing.token_accuracy"
        result_key = f"{section}_acc"

    return _base_meta(
        result_key=result_key,
        protocol=protocol,
        scorer=scorer,
        comparable_group=comparable_group,
    )


def build_icl_metric_meta(section, hparams, model_name):
    model_family = infer_model_family(model_name)
    if model_family == "seq2seq":
        protocol = "icl_seq2seq_teacher_forcing"
        comparable_group = "lm.icl_seq2seq_teacher_forcing.token_accuracy"
    else:
        protocol = "icl_decoder_only_teacher_forcing"
        comparable_group = "lm.icl_decoder_only_teacher_forcing.token_accuracy"
    return _base_meta(
        result_key=f"{section}_acc",
        protocol=protocol,
        scorer="token_accuracy",
        comparable_group=comparable_group,
    )


def build_multimodal_metric_meta(
    section,
    hparams,
    model_name,
    *,
    result_key,
    protocol,
    scorer,
    comparable_group,
    exact_match=False,
):
    if exact_match:
        protocol = f"{protocol}_exact_match"
        scorer = "sequence_exact_match"
        comparable_group = f"{comparable_group.rsplit('.', 1)[0]}.sequence_exact_match"
    return _base_meta(
        result_key=result_key,
        protocol=protocol,
        scorer=scorer,
        comparable_group=comparable_group,
    )
