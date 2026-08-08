"""bitsandbytes AdamW with 32-bit states and the synchronized-once fast path."""

from __future__ import annotations

from library.adamw8bit_fast import AdamW8bitFast


# bitsandbytes uses 32-bit states for tensors smaller than min_8bit_size.
# This matches the threshold used by the experiments that established the
# bnb32 condition while remaining representable by the CUDA-side int32 APIs.
BNB32_MIN_8BIT_SIZE = 2**31 - 1


class AdamWBnb(AdamW8bitFast):
    """AdamW8bitFast with the bitsandbytes 32-bit state path forced on.

    This class intentionally inherits the synchronized-once implementation
    from :class:`AdamW8bitFast`. It selects bitsandbytes' 32-bit update path by
    forcing ``min_8bit_size`` above every practical parameter tensor size.
    """

    _optimizer_display_name = "AdamWBnb"
    _stock_optimizer_display_name = "bitsandbytes AdamW (32-bit state)"

    def __init__(self, params, *args, **kwargs):
        requested_minimum = kwargs.pop("min_8bit_size", BNB32_MIN_8BIT_SIZE)
        if requested_minimum != BNB32_MIN_8BIT_SIZE:
            raise ValueError(
                "AdamWBnb fixes min_8bit_size at "
                f"{BNB32_MIN_8BIT_SIZE}; remove the conflicting optimizer argument"
            )
        super().__init__(
            params,
            *args,
            min_8bit_size=BNB32_MIN_8BIT_SIZE,
            **kwargs,
        )
