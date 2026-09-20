"""Load the original HF processor independently of global Auto registrations."""

from __future__ import annotations


def load_processor(source, *, revision=None, local_files_only=False):
    """Preserve official Qwen3.5 image/video settings and its chat template.

    Importing mlx-vlm registers its own Qwen3VLProcessor with Transformers.
    Loading explicit HF classes also avoids the Auto image/video registries,
    whose dispatch may likewise be changed by another runtime in this process.
    Both original split configs and HF's newer nested processor config work.
    """
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    kwargs = {"revision": revision, "local_files_only": local_files_only}
    processor_dict, initialization = Qwen3VLProcessor.get_processor_dict(source, **kwargs)
    components = [
        Qwen2VLImageProcessor.from_pretrained(source, **kwargs),
        AutoTokenizer.from_pretrained(source, **kwargs),
        Qwen3VLVideoProcessor.from_pretrained(source, **kwargs),
    ]
    return Qwen3VLProcessor.from_args_and_dict(components, processor_dict, **initialization)
