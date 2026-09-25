import numpy as np
import pytest

from qev.export_lfm2 import FOUNDATIONS, convert_adapter
from qev.inference import Agent


def agent_with(config):
    agent = Agent.__new__(Agent)
    agent.config = config
    return agent


def test_text_only_model_rejects_media():
    agent = agent_with({"model_name": "qev-230m", "family": "lfm2", "modalities": ["text"]})
    agent._require_media([])
    with pytest.raises(ValueError, match="qev-230m accepts text only; image input"):
        agent._require_media(["image"])
    with pytest.raises(ValueError, match="video"):
        agent._require_media(["video"])


def test_image_model_accepts_video_frames():
    agent_with({"family": "lfm2", "modalities": ["text", "image"]})._require_media(["image", "video"])
    agent_with({})._require_media(["image", "video"])


def test_foundations_map_checkpoint_kinds():
    assert FOUNDATIONS["lfm2"]["modalities"] == ["text"]
    assert "processor_config.json" not in FOUNDATIONS["lfm2"]["files"]
    assert FOUNDATIONS["lfm2_vl"]["modalities"] == ["text", "image"]


def test_convert_adapter_renames_peft_keys(tmp_path):
    from safetensors.numpy import save_file

    path = tmp_path / "adapter.safetensors"
    save_file({"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones((4, 8), np.float16),
               "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.ones((8, 4), np.float16)}, str(path))
    weights = convert_adapter(path)
    assert sorted(weights) == ["language_model.model.layers.0.self_attn.q_proj.lora_a",
                               "language_model.model.layers.0.self_attn.q_proj.lora_b"]
    assert all(value.dtype == np.float32 for value in weights.values())
    save_file({"base_model.model.lm_head.weight": np.ones((2, 2), np.float16)}, str(path))
    with pytest.raises(ValueError):
        convert_adapter(path)
