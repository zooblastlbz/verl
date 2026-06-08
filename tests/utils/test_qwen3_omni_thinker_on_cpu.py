import importlib.util
from types import SimpleNamespace

import pytest
import transformers

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ray") is None or importlib.util.find_spec("tensordict") is None,
    reason="verl package dependencies are not installed",
)


def _model_utils():
    from verl.utils import model

    return model


def test_get_qwen3_omni_thinker_config_extracts_nested_config():
    model_utils = _model_utils()
    nested_text_config = SimpleNamespace()
    full_config = SimpleNamespace(
        model_type="qwen3_omni_moe",
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
        thinker_config=SimpleNamespace(
            model_type="qwen3_omni_moe_thinker",
            architectures=None,
            text_config=nested_text_config,
            audio_token_id=151646,
            image_token_id=151655,
            video_token_id=151656,
        ),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        _attn_implementation="flash_attention_2",
    )

    thinker_config = model_utils.get_qwen3_omni_thinker_config(full_config)

    assert thinker_config is not full_config.thinker_config
    assert thinker_config.model_type == model_utils.QWEN3_OMNI_THINKER_MODEL_TYPE
    assert thinker_config.architectures == [model_utils.QWEN3_OMNI_THINKER_ARCHITECTURE]
    assert thinker_config.bos_token_id == 1
    assert thinker_config.eos_token_id == 2
    assert thinker_config.pad_token_id == 3
    assert thinker_config.text_config.bos_token_id == 1
    assert thinker_config.text_config.eos_token_id == 2
    assert thinker_config.text_config.pad_token_id == 3
    assert thinker_config.text_config._attn_implementation == "flash_attention_2"
    assert thinker_config.vision_start_token_id == 151652


def test_get_qwen3_omni_thinker_config_accepts_existing_thinker_config():
    model_utils = _model_utils()
    thinker_config = SimpleNamespace(
        model_type=model_utils.QWEN3_OMNI_THINKER_MODEL_TYPE,
        architectures=[model_utils.QWEN3_OMNI_THINKER_ARCHITECTURE],
        text_config=SimpleNamespace(),
        vision_start_token_id=42,
    )

    converted_config = model_utils.get_qwen3_omni_thinker_config(thinker_config)

    assert converted_config is not thinker_config
    assert converted_config.architectures == [model_utils.QWEN3_OMNI_THINKER_ARCHITECTURE]
    assert converted_config.vision_start_token_id == 42


def test_get_hf_auto_model_class_prefers_qwen3_omni_thinker(monkeypatch):
    model_utils = _model_utils()

    class DummyQwen3OmniThinker:
        pass

    monkeypatch.setattr(
        transformers,
        "Qwen3OmniMoeThinkerForConditionalGeneration",
        DummyQwen3OmniThinker,
        raising=False,
    )
    config = SimpleNamespace(
        model_type=model_utils.QWEN3_OMNI_THINKER_MODEL_TYPE,
        architectures=[model_utils.QWEN3_OMNI_THINKER_ARCHITECTURE],
    )

    assert model_utils.get_hf_auto_model_class(config) is DummyQwen3OmniThinker


def test_convert_qwen3_omni_thinker_weight_keys_for_vllm_maps_language_model_keys():
    model_utils = _model_utils()
    layer_weight = object()
    lm_head_weight = object()
    visual_weight = object()
    audio_weight = object()
    already_mapped_weight = object()
    full_checkpoint_weight = object()

    converted = model_utils.convert_qwen3_omni_thinker_weight_keys_for_vllm(
        [
            ("model.layers.0.self_attn.q_proj.weight", layer_weight),
            ("lm_head.weight", lm_head_weight),
            ("visual.patch_embed.proj.weight", visual_weight),
            ("audio_tower.conv.weight", audio_weight),
            ("language_model.model.layers.0.mlp.gate.weight", already_mapped_weight),
            ("thinker.model.layers.0.input_layernorm.weight", full_checkpoint_weight),
        ]
    )

    assert converted == [
        ("language_model.model.layers.0.self_attn.q_proj.weight", layer_weight),
        ("language_model.lm_head.weight", lm_head_weight),
        ("visual.patch_embed.proj.weight", visual_weight),
        ("audio_tower.conv.weight", audio_weight),
        ("language_model.model.layers.0.mlp.gate.weight", already_mapped_weight),
        ("thinker.model.layers.0.input_layernorm.weight", full_checkpoint_weight),
    ]


def test_vllm_qwen3_omni_thinker_remap_does_not_require_language_model_attr():
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    model_utils = _model_utils()

    class Qwen3OmniMoeThinkerForConditionalGeneration:
        pass

    model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="unrelated"))
    should_remap = vLLMColocateWorkerExtension._should_remap_qwen3_omni_thinker_weights(
        object(),
        [("model.layers.0.input_layernorm.weight", object())],
        Qwen3OmniMoeThinkerForConditionalGeneration(),
        model_config,
    )

    assert should_remap

    should_remap_from_config = vLLMColocateWorkerExtension._should_remap_qwen3_omni_thinker_weights(
        object(),
        [("lm_head.weight", object())],
        object(),
        SimpleNamespace(hf_config=SimpleNamespace(model_type=model_utils.QWEN3_OMNI_THINKER_MODEL_TYPE)),
    )

    assert should_remap_from_config
