"""Load the original HF processor independently of global Auto registrations."""

from __future__ import annotations

import importlib.util
import os

PROCESSOR_BACKENDS = ("auto", "hf", "numpy")


def processor_backend(requested: str | None = None) -> str:
    """Resolve ``auto``: the validated HF processors when torch is importable.

    Transformers' Qwen2/3-VL image and video processors import torch. Without
    it (the Homebrew install), mlx-vlm's numpy ports are used; tokens and grids
    match the HF path and pixels differ by at most a couple of 8-bit levels.
    ``QEV_PROCESSOR`` overrides the default.
    """
    choice = (requested or os.environ.get("QEV_PROCESSOR") or "auto").lower()
    if choice not in PROCESSOR_BACKENDS:
        raise ValueError(f"processor backend must be one of {PROCESSOR_BACKENDS}")
    if choice == "auto":
        return "hf" if importlib.util.find_spec("torch") is not None else "numpy"
    return choice


def _numpy_components(source, kwargs):
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLImageProcessor,
        Qwen3VLVideoProcessor,
    )
    from transformers.video_utils import VideoMetadata

    class VideoProcessor(Qwen3VLVideoProcessor):
        """Return the caller's frame metadata, which Qwen3VL needs for timestamps."""

        def __call__(self, videos, video_metadata=None, **options):
            output = super().__call__(videos, **options)
            listed = videos if isinstance(videos, (list, tuple)) else [videos]
            if video_metadata is None:
                video_metadata = [{"total_num_frames": len(video), "fps": None,
                                   "frames_indices": list(range(len(video)))} for video in listed]
            output["video_metadata"] = [item if isinstance(item, VideoMetadata) else VideoMetadata(**item)
                                        for item in video_metadata]
            return output

    return (Qwen3VLImageProcessor.from_pretrained(source, **kwargs),
            VideoProcessor.from_pretrained(source, **kwargs))


def load_processor(source, *, revision=None, local_files_only=False, backend=None):
    """Preserve official Qwen3.5 image/video settings and its chat template.

    Importing mlx-vlm registers its own Qwen3VLProcessor with Transformers.
    Loading explicit HF classes also avoids the Auto image/video registries,
    whose dispatch may likewise be changed by another runtime in this process.
    Both original split configs and HF's newer nested processor config work.
    """
    from transformers import AutoTokenizer
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    kwargs = {"revision": revision, "local_files_only": local_files_only}
    processor_dict, initialization = Qwen3VLProcessor.get_processor_dict(source, **kwargs)
    if processor_backend(backend) == "hf":
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

        image, video = (Qwen2VLImageProcessor.from_pretrained(source, **kwargs),
                        Qwen3VLVideoProcessor.from_pretrained(source, **kwargs))
    else:
        image, video = _numpy_components(source, kwargs)
    components = [image, AutoTokenizer.from_pretrained(source, **kwargs), video]
    return _processor_class(backend).from_args_and_dict(components, processor_dict, **initialization)


def _processor_class(backend):
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    if processor_backend(backend) == "hf":
        return Qwen3VLProcessor

    class NumpyQwen3VLProcessor(Qwen3VLProcessor):
        """Accept the numpy media processors.

        Without torch, Transformers exposes placeholder base classes, so its
        component isinstance check would reject working numpy processors.
        """

        def check_argument_for_proper_class(self, argument_name, argument):
            if argument_name in ("image_processor", "video_processor"):
                return type(argument)
            return super().check_argument_for_proper_class(argument_name, argument)

    return NumpyQwen3VLProcessor
