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
