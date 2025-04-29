from typing import Mapping
from transformers import BatchFeature, ProcessorMixin
from vllm.utils import resolve_mm_processor_kwargs


def call_hf_processor(
    self,
    hf_processor: ProcessorMixin,
    data: Mapping[str, object],
    kwargs: Mapping[str, object] = {},
) -> BatchFeature:
    """
    Call :code:`hf_processor` on the prompt :code:`data`
    (text, image, audio...) with configurable options :code:`kwargs`.
    """
    assert callable(hf_processor)

    base_kwargs = self.model_config.mm_processor_kwargs
    if base_kwargs is None:
        base_kwargs = {}

    merged_kwargs = resolve_mm_processor_kwargs(
        base_kwargs,
        kwargs,
        hf_processor,
        requires_kw_only=False,
        allow_var_kwargs=True,
    )

    try:
        return hf_processor(**data, **merged_kwargs, return_tensors="np")
    except Exception as exc:
        msg = (f"Failed to apply {type(hf_processor).__name__} "
                f"on data={data} with kwargs={merged_kwargs}")

        raise RuntimeError(msg) from exc
