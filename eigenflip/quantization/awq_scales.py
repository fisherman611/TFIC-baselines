"""AWQ activation-aware scale and weight-clipping search.

The paper search uses the per-input-channel average activation magnitude
``mean(abs(X))`` and ``s = activation_scale ** alpha``.  After choosing alpha,
AWQ searches a symmetric clipping bound per output channel and weight group.
The functions here return fixed-grid artifacts; they do not implement packed
integer kernels or mutate neighbouring LayerNorm/activation modules.
"""

from __future__ import annotations

import torch


def _validate_quant_args(bits: int, group_size: int, scheme: str) -> None:
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if scheme not in {"asymmetric", "symmetric"}:
        raise ValueError(f"scheme must be 'asymmetric' or 'symmetric', got {scheme!r}")
    if scheme == "symmetric" and bits < 2:
        raise ValueError("symmetric AWQ quantization requires bits >= 2")


def _minimum_range(reference: torch.Tensor, eps: float) -> torch.Tensor:
    floor = torch.as_tensor(eps, device=reference.device, dtype=reference.dtype)
    if floor == 0:
        floor = torch.nextafter(
            torch.zeros((), device=reference.device, dtype=reference.dtype),
            torch.ones((), device=reference.device, dtype=reference.dtype),
        )
    return floor


@torch.no_grad()
def _quantize_grouped(
    grouped: torch.Tensor,
    bits: int,
    scheme: str,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fake-quantize tensors whose last dimension is one weight group."""
    floor = _minimum_range(grouped, eps)
    if scheme == "asymmetric":
        wmin = grouped.amin(dim=-1, keepdim=True)
        wmax = grouped.amax(dim=-1, keepdim=True)
        qmin, qmax = 0, 2**bits - 1
        scale = (wmax - wmin).clamp_min(floor) / qmax
        zero = torch.round(-wmin / scale).clamp(qmin, qmax)
        codes = torch.round(grouped / scale + zero).clamp(qmin, qmax)
        return (codes - zero) * scale

    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = absmax.clamp_min(floor) / qmax
    codes = torch.round(grouped / scale).clamp(qmin, qmax)
    return codes * scale


@torch.no_grad()
def _groupwise_quant(W, bits, group_size, scheme="asymmetric", clip_max=None):
    _validate_quant_args(bits, group_size, scheme)
    if W.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got shape {tuple(W.shape)}")

    rows, in_features = W.shape
    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size
    if padded_in > in_features:
        padded = torch.zeros(rows, padded_in, device=W.device, dtype=W.dtype)
        padded[:, :in_features] = W
    else:
        padded = W

    grouped = padded.reshape(rows, n_groups, group_size)
    if clip_max is not None:
        clip = torch.as_tensor(clip_max, device=W.device, dtype=W.dtype)
        if clip.shape == (rows, n_groups):
            clip = clip.unsqueeze(-1)
        if clip.shape != (rows, n_groups, 1):
            raise ValueError(
                "clip_max must have shape "
                f"({rows}, {n_groups}, 1), got {tuple(clip.shape)}"
            )
        if not torch.isfinite(clip).all() or torch.any(clip <= 0):
            raise ValueError("clip_max values must be finite and positive")
        grouped = grouped.clamp(-clip, clip)

    dequantized = _quantize_grouped(grouped, bits, scheme).reshape(rows, padded_in)
    return dequantized[:, :in_features]


@torch.no_grad()
def _groupwise_asym_quant(W, bits, group_size):
    return _groupwise_quant(W, bits, group_size, scheme="asymmetric")


@torch.no_grad()
def compute_awq_scales(
    W: torch.Tensor,
    salience_l2: torch.Tensor,
    X_sample: torch.Tensor,
    bits: int,
    group_size: int,
    n_grid: int = 20,
    scheme: str = "asymmetric",
) -> tuple[torch.Tensor, float, float]:
    """Search the AWQ per-input-channel scale for one fixed-grid layer.

    ``salience_l2`` is retained as the parameter name for API compatibility;
    callers must pass the paper statistic ``mean(abs(X), dim=tokens)``.
    """
    _validate_quant_args(bits, group_size, scheme)
    if W.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got shape {tuple(W.shape)}")
    if n_grid <= 0:
        raise ValueError(f"n_grid must be positive, got {n_grid}")

    device, dtype = W.device, W.dtype
    activation_scale = salience_l2.to(device=device, dtype=torch.float32).reshape(-1)
    if activation_scale.numel() != W.shape[1]:
        raise ValueError(
            f"expected {W.shape[1]} activation scales, got {activation_scale.numel()}"
        )
    if not torch.isfinite(activation_scale).all() or torch.any(activation_scale < 0):
        raise ValueError("activation scales must be finite and non-negative")

    inputs = X_sample.to(device=device, dtype=dtype).reshape(-1, W.shape[1])
    original_output = inputs @ W.t()

    best_error = float("inf")
    best_alpha = 0.0
    best_scales = torch.ones(W.shape[1], device=device, dtype=dtype)
    for grid_index in range(n_grid):
        alpha = grid_index / n_grid
        scales_f32 = activation_scale.pow(alpha).clamp_min(1e-4)
        scales_f32 = scales_f32 / torch.sqrt(scales_f32.max() * scales_f32.min())
        scales = scales_f32.to(dtype)
        quantized = _groupwise_quant(
            W * scales.unsqueeze(0), bits, group_size, scheme=scheme
        )
        quantized_output = (inputs / scales.unsqueeze(0)) @ quantized.t()
        error = (original_output.float() - quantized_output.float()).pow(2).mean().item()
        if error < best_error:
            best_error = error
            best_alpha = alpha
            best_scales = scales.clone()

    return best_scales, best_alpha, best_error


@torch.no_grad()
def compute_awq_clip(
    W_scaled: torch.Tensor,
    X_scaled: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    scheme: str = "asymmetric",
    n_grid: int = 20,
    max_shrink: float = 0.5,
    sample_tokens: int = 512,
    output_chunk_size: int = 256,
) -> torch.Tensor:
    """Search AWQ clipping bounds per output channel and weight group."""
    _validate_quant_args(bits, group_size, scheme)
    if W_scaled.dim() != 2:
        raise ValueError("W_scaled must be a 2D tensor")
    if n_grid <= 0 or not (0 < max_shrink <= 1):
        raise ValueError("n_grid must be positive and max_shrink must be in (0, 1]")
    if sample_tokens <= 0 or output_chunk_size <= 0:
        raise ValueError("sample_tokens and output_chunk_size must be positive")

    rows, in_features = W_scaled.shape
    inputs = X_scaled.to(device=W_scaled.device, dtype=W_scaled.dtype)
    inputs = inputs.reshape(-1, in_features)
    if inputs.shape[0] > sample_tokens:
        indices = torch.linspace(
            0, inputs.shape[0] - 1, sample_tokens, device=inputs.device
        ).round().long()
        inputs = inputs.index_select(0, indices)

    n_groups = (in_features + group_size - 1) // group_size
    padded_in = n_groups * group_size
    if padded_in > in_features:
        padded_weights = torch.zeros(
            rows, padded_in, device=W_scaled.device, dtype=W_scaled.dtype
        )
        padded_weights[:, :in_features] = W_scaled
        padded_inputs = torch.zeros(
            inputs.shape[0], padded_in, device=inputs.device, dtype=inputs.dtype
        )
        padded_inputs[:, :in_features] = inputs
    else:
        padded_weights, padded_inputs = W_scaled, inputs

    grouped_weights = padded_weights.reshape(rows, n_groups, group_size)
    grouped_inputs = padded_inputs.reshape(inputs.shape[0], n_groups, group_size)
    best_bounds = []
    shrink_steps = max(1, int(max_shrink * n_grid))

    for start in range(0, rows, output_chunk_size):
        weights = grouped_weights[start : start + output_chunk_size]
        original_bound = weights.abs().amax(dim=-1, keepdim=True)
        floor = _minimum_range(original_bound, 1e-5)
        original_bound = original_bound.clamp_min(floor)
        best_bound = original_bound.clone()
        minimum_error = torch.full_like(original_bound, float("inf"))
        original_output = torch.einsum("cgd,tgd->ctg", weights, grouped_inputs)

        for shrink_index in range(shrink_steps):
            bound = original_bound * (1 - shrink_index / n_grid)
            clipped = weights.clamp(-bound, bound)
            quantized = _quantize_grouped(clipped, bits, scheme)
            quantized_output = torch.einsum("cgd,tgd->ctg", quantized, grouped_inputs)
            error = (
                quantized_output.float() - original_output.float()
            ).pow(2).mean(dim=1).unsqueeze(-1)
            improved = error < minimum_error
            minimum_error = torch.where(improved, error, minimum_error)
            best_bound = torch.where(improved, bound, best_bound)

        best_bounds.append(best_bound)

    return torch.cat(best_bounds, dim=0)


def layer_params_from_awq_run(artifact: dict) -> dict[str, dict]:
    """Normalize legacy scale dictionaries and current AWQ layer artifacts."""
    raw_layers = artifact.get("layers", artifact)
    out = {}
    for name, info in raw_layers.items():
        if torch.is_tensor(info) or not isinstance(info, dict):
            out[name] = {"scales": torch.as_tensor(info)}
            continue
        scales = info.get("scales")
        if scales is None:
            continue
        normalized = dict(info)
        normalized["scales"] = scales if torch.is_tensor(scales) else torch.as_tensor(scales)
        if normalized.get("clip_max") is not None:
            normalized["clip_max"] = torch.as_tensor(normalized["clip_max"])
        out[name] = normalized
    return out


def scales_from_awq_run(artifact: dict) -> dict[str, torch.Tensor]:
    """Return the legacy ``{layer_name: scales}`` view of an AWQ artifact."""
    return {
        name: info["scales"]
        for name, info in layer_params_from_awq_run(artifact).items()
    }
