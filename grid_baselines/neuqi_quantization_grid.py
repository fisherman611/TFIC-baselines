"""NeUQI quantization-parameter initialization grid.

NeUQI improves the initialization of uniform affine quantization parameters.
For each row/group vector it minimizes the diagonal-Hessian approximation

    sum_i H_ii * (Q_{s,z}(w_i) - w_i)^2

over the scale ``s`` and a floating-point zero-point ``z``.  This module keeps
the repository's fixed-grid contract: it returns a grid with learned
``scale``/``zero_point`` tensors, then assignment methods choose integer codes
on top of that grid.  The asymmetric variant uses NeUQI's floating zero-point
solver.  The symmetric variant is a repository extension: it keeps zero-point
fixed at 0 and searches the diagonal-Hessian weighted scale/clipping factor on
the signed grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from eigenflip.statistics.trust_region import LayerStats
from .vanilla_quantization_grid import VanillaQuantizationGrid


@dataclass
class NeUQIQuantizationGrid(VanillaQuantizationGrid):
    """Uniform affine grid initialized by NeUQI's weighted loss."""

    scale_candidates: int = 2048
    coarse_candidates: int = 64
    candidate_chunk_size: int = 16


@torch.no_grad()
def _expand_group(group_values: torch.Tensor, group_size: int, padded_in: int) -> torch.Tensor:
    rows, _n_groups, _ = group_values.shape
    return group_values.repeat(1, 1, group_size).reshape(rows, padded_in)


@torch.no_grad()
def _pad_weights(weights: torch.Tensor, padded_in: int) -> torch.Tensor:
    rows, in_features = weights.shape
    if padded_in == in_features:
        return weights
    padded = torch.zeros(rows, padded_in, device=weights.device, dtype=weights.dtype)
    padded[:, :in_features] = weights
    return padded


@torch.no_grad()
def _pad_diag(diag: torch.Tensor, padded_in: int) -> torch.Tensor:
    if diag.numel() == padded_in:
        return diag
    padded = torch.zeros(padded_in, device=diag.device, dtype=diag.dtype)
    padded[: diag.numel()] = diag
    return padded


@torch.no_grad()
def _candidate_factors(indices: torch.Tensor, total: int) -> torch.Tensor:
    return indices.to(dtype=torch.float32) / float(total)


@torch.no_grad()
def _quadratic_interval_minimum(
    coeffs: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = coeffs[:, 0].clamp_min(torch.finfo(coeffs.dtype).tiny)
    b = coeffs[:, 1]
    c = coeffs[:, 2]
    z = -b / (2 * a)
    z = torch.maximum(torch.minimum(z, right), left)
    loss = a * z.square() + b * z + c
    return z, loss


@torch.no_grad()
def _solve_piecewise_quadratic(
    initial_coeffs: torch.Tensor,
    transition_points: torch.Tensor,
    delta_coeffs: torch.Tensor,
    *,
    left_bound: torch.Tensor | None = None,
    right_bound: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize a batch of one-dimensional piecewise quadratics.

    Coefficients are stored as ``a z^2 + b z + c``.  ``delta_coeffs`` must be
    the next interval's coefficients minus the previous interval's coefficients,
    matching Algorithm 1 in the paper.
    """

    transitions, order = transition_points.sort(dim=-1)
    deltas = delta_coeffs.gather(
        dim=1,
        index=order.unsqueeze(-1).expand_as(delta_coeffs),
    )
    vectors, n_events = transitions.shape
    coeffs = initial_coeffs.clone()

    neg_inf = torch.full((vectors,), -torch.inf, device=coeffs.device, dtype=coeffs.dtype)
    pos_inf = torch.full((vectors,), torch.inf, device=coeffs.device, dtype=coeffs.dtype)
    if left_bound is None:
        left = neg_inf
    else:
        left = left_bound.to(device=coeffs.device, dtype=coeffs.dtype)
    if right_bound is None:
        final_right = pos_inf
    else:
        final_right = right_bound.to(device=coeffs.device, dtype=coeffs.dtype)

    right = torch.minimum(transitions[:, 0], final_right)
    best_z, best_loss = _quadratic_interval_minimum(coeffs, left, right)

    for event_idx in range(n_events):
        coeffs = coeffs + deltas[:, event_idx, :]
        left = torch.maximum(transitions[:, event_idx], left_bound) if left_bound is not None else transitions[:, event_idx]
        if event_idx + 1 < n_events:
            right = transitions[:, event_idx + 1]
            if right_bound is not None:
                right = torch.minimum(right, final_right)
        else:
            right = final_right

        z, loss = _quadratic_interval_minimum(coeffs, left, right)
        improved = loss < best_loss
        best_z = torch.where(improved, z, best_z)
        best_loss = torch.where(improved, loss, best_loss)

    return best_z, best_loss


@torch.no_grad()
def _solve_eq8_zero_point(
    x: torch.Tensor,
    hessian_diag: torch.Tensor,
    qmax: int,
) -> torch.Tensor:
    """Algorithm 3: optimal zero-point for the Eq. 8 approximation."""

    vectors, width = x.shape
    h = hessian_diag
    initial = torch.stack(
        [
            h.sum(dim=-1),
            2 * (h * x).sum(dim=-1),
            (h * x.square()).sum(dim=-1),
        ],
        dim=-1,
    )

    enter_t = -0.5 - x
    exit_t = float(qmax) + 0.5 - x
    transition_points = torch.cat([enter_t, exit_t], dim=-1)

    enter_delta = torch.stack(
        [
            -h,
            -2 * h * x,
            h * (0.25 - x.square()),
        ],
        dim=-1,
    )
    exit_delta = torch.stack(
        [
            h,
            2 * h * (x - qmax),
            h * ((x - qmax).square() - 0.25),
        ],
        dim=-1,
    )
    delta_coeffs = torch.cat([enter_delta, exit_delta], dim=1)
    z, _loss = _solve_piecewise_quadratic(
        initial.reshape(vectors, 3),
        transition_points.reshape(vectors, 2 * width),
        delta_coeffs.reshape(vectors, 2 * width, 3),
    )
    return z


@torch.no_grad()
def _solve_limited_eq7_zero_point(
    x: torch.Tensor,
    hessian_diag: torch.Tensor,
    qmax: int,
    z_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm 4: solve Eq. 7 in ``[z_center - 1, z_center + 1]``."""

    vectors, width = x.shape
    h = hessian_diag
    left = z_center - 1
    right = z_center + 1
    initial_codes = torch.round(x + left.unsqueeze(-1)).clamp(0, qmax)
    shifted = x - initial_codes
    initial = torch.stack(
        [
            h.sum(dim=-1),
            2 * (h * shifted).sum(dim=-1),
            (h * shifted.square()).sum(dim=-1),
        ],
        dim=-1,
    )

    j1 = torch.ceil(left.unsqueeze(-1) + x - 0.5)
    j2 = j1 + 1
    j = torch.cat([j1, j2], dim=-1)
    x_events = torch.cat([x, x], dim=-1)
    h_events = torch.cat([h, h], dim=-1)
    transition_points = j + 0.5 - x_events

    valid = (
        (j >= 0)
        & (j < qmax)
        & (transition_points >= left.unsqueeze(-1))
        & (transition_points < right.unsqueeze(-1))
    )
    transition_points = torch.where(
        valid,
        transition_points,
        right.unsqueeze(-1),
    )
    delta_coeffs = torch.stack(
        [
            torch.zeros_like(h_events),
            -2 * h_events,
            h_events * (2 * (j - x_events) + 1),
        ],
        dim=-1,
    )
    delta_coeffs = torch.where(valid.unsqueeze(-1), delta_coeffs, torch.zeros_like(delta_coeffs))
    return _solve_piecewise_quadratic(
        initial.reshape(vectors, 3),
        transition_points.reshape(vectors, 2 * width),
        delta_coeffs.reshape(vectors, 2 * width, 3),
        left_bound=left,
        right_bound=right,
    )


@torch.no_grad()
def _optimal_zero_point(
    x: torch.Tensor,
    hessian_diag: torch.Tensor,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    leading_shape = x.shape[:-1]
    width = x.shape[-1]
    flat_x = x.reshape(-1, width)
    flat_h = hessian_diag.expand_as(x).reshape(-1, width)
    z_s = _solve_eq8_zero_point(flat_x, flat_h, qmax)
    z, loss = _solve_limited_eq7_zero_point(flat_x, flat_h, qmax, z_s)
    return z.reshape(leading_shape), loss.reshape(leading_shape)


@torch.no_grad()
def _evaluate_scale_factors(
    grouped: torch.Tensor,
    hessian_diag: torch.Tensor,
    scale_upper: torch.Tensor,
    factors: torch.Tensor,
    qmax: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate scale factors for a chunk of grouped rows.

    ``grouped`` is ``[rows, groups, group_size]``.  ``factors`` may be one
    shared vector ``[candidates]`` or per-vector factors
    ``[rows, groups, candidates]``.
    """

    if factors.dim() == 1:
        scale_group = scale_upper * factors.reshape(1, 1, -1)
    else:
        scale_group = scale_upper * factors
    scale_group = scale_group.clamp_min(eps)

    x = grouped.unsqueeze(2) / scale_group.unsqueeze(-1)
    h = hessian_diag.unsqueeze(0).unsqueeze(2).to(device=grouped.device, dtype=x.dtype)

    zero_point, normalized_loss = _optimal_zero_point(
        x,
        h,
        qmax,
    )
    weighted_loss = normalized_loss * scale_group.square()
    best = weighted_loss.argmin(dim=-1, keepdim=True)
    best_scale = scale_group.gather(-1, best)
    best_zero = zero_point.gather(-1, best)
    best_loss = weighted_loss.gather(-1, best)
    return best_scale, best_zero, best_loss, best.squeeze(-1)


@torch.no_grad()
def _evaluate_symmetric_scale_factors(
    grouped: torch.Tensor,
    hessian_diag: torch.Tensor,
    scale_upper: torch.Tensor,
    factors: torch.Tensor,
    qmin: int,
    qmax: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate scale factors for symmetric signed-code NeUQI."""

    if factors.dim() == 1:
        scale_group = scale_upper * factors.reshape(1, 1, -1)
    else:
        scale_group = scale_upper * factors
    scale_group = scale_group.clamp_min(eps)

    weights = grouped.unsqueeze(2)
    codes = torch.round(weights / scale_group.unsqueeze(-1)).clamp(qmin, qmax)
    reconstructed = codes * scale_group.unsqueeze(-1)
    h = hessian_diag.unsqueeze(0).unsqueeze(2).to(
        device=grouped.device,
        dtype=reconstructed.dtype,
    )
    weighted_loss = (h * (reconstructed - weights).square()).sum(dim=-1)
    best = weighted_loss.argmin(dim=-1, keepdim=True)
    best_scale = scale_group.gather(-1, best)
    best_loss = weighted_loss.gather(-1, best)
    return best_scale, best_loss, best.squeeze(-1)


@torch.no_grad()
def _search_neuqi_params(
    grouped: torch.Tensor,
    hessian_diag: torch.Tensor,
    valid_mask: torch.Tensor,
    qmax: int,
    *,
    scale_candidates: int,
    coarse_candidates: int,
    row_chunk_size: int,
    candidate_chunk_size: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, n_groups, _group_size = grouped.shape
    scale_group = torch.empty(rows, n_groups, 1, device=grouped.device, dtype=torch.float32)
    zero_group = torch.empty_like(scale_group)

    coarse_candidates = min(coarse_candidates, scale_candidates)
    fine_width = max(1, scale_candidates // coarse_candidates)
    coarse_indices = torch.arange(
        1,
        coarse_candidates + 1,
        device=grouped.device,
        dtype=torch.long,
    )
    coarse_factors = _candidate_factors(coarse_indices, coarse_candidates).to(grouped.device)

    hessian_grouped = hessian_diag.reshape(n_groups, -1).to(
        device=grouped.device,
        dtype=torch.float32,
    )

    for start in range(0, rows, row_chunk_size):
        end = min(start + row_chunk_size, rows)
        chunk = grouped[start:end].to(torch.float32)
        mask = valid_mask.unsqueeze(0).to(device=chunk.device)
        wmin = chunk.masked_fill(~mask, torch.inf).amin(dim=-1, keepdim=True)
        wmax = chunk.masked_fill(~mask, -torch.inf).amax(dim=-1, keepdim=True)
        scale_upper = ((wmax - wmin) / qmax).clamp_min(eps)

        best_coarse_loss = None
        coarse_best = None
        for candidate_start in range(0, coarse_candidates, candidate_chunk_size):
            candidate_end = min(candidate_start + candidate_chunk_size, coarse_candidates)
            _scale, _zero, loss, index = _evaluate_scale_factors(
                chunk,
                hessian_grouped,
                scale_upper,
                coarse_factors[candidate_start:candidate_end],
                qmax,
                eps,
            )
            index = index + candidate_start
            if best_coarse_loss is None:
                best_coarse_loss = loss
                coarse_best = index
                continue
            improved = loss < best_coarse_loss
            best_coarse_loss = torch.where(improved, loss, best_coarse_loss)
            coarse_best = torch.where(improved.squeeze(-1), index, coarse_best)
        center = ((coarse_best + 1) * fine_width).clamp(1, scale_candidates)
        half = max(1, fine_width // 2)
        offsets = torch.arange(
            -half,
            half + 1,
            device=grouped.device,
            dtype=torch.long,
        )
        fine_indices = (center.unsqueeze(-1) + offsets).clamp(1, scale_candidates)
        fine_factors = fine_indices.to(torch.float32) / float(scale_candidates)

        best_scale = best_zero = best_loss = None
        fine_candidates = fine_factors.shape[-1]
        for candidate_start in range(0, fine_candidates, candidate_chunk_size):
            candidate_end = min(candidate_start + candidate_chunk_size, fine_candidates)
            scale, zero, loss, _index = _evaluate_scale_factors(
                chunk,
                hessian_grouped,
                scale_upper,
                fine_factors[..., candidate_start:candidate_end],
                qmax,
                eps,
            )
            if best_loss is None:
                best_scale, best_zero, best_loss = scale, zero, loss
                continue
            improved = loss < best_loss
            best_scale = torch.where(improved, scale, best_scale)
            best_zero = torch.where(improved, zero, best_zero)
            best_loss = torch.where(improved, loss, best_loss)
        scale_group[start:end] = best_scale
        zero_group[start:end] = best_zero

    return scale_group, zero_group


@torch.no_grad()
def _search_symmetric_neuqi_params(
    grouped: torch.Tensor,
    hessian_diag: torch.Tensor,
    qmin: int,
    qmax: int,
    *,
    scale_candidates: int,
    coarse_candidates: int,
    row_chunk_size: int,
    candidate_chunk_size: int,
    eps: float,
) -> torch.Tensor:
    rows, n_groups, _group_size = grouped.shape
    scale_group = torch.empty(rows, n_groups, 1, device=grouped.device, dtype=torch.float32)

    coarse_candidates = min(coarse_candidates, scale_candidates)
    fine_width = max(1, scale_candidates // coarse_candidates)
    coarse_indices = torch.arange(
        1,
        coarse_candidates + 1,
        device=grouped.device,
        dtype=torch.long,
    )
    coarse_factors = _candidate_factors(coarse_indices, coarse_candidates).to(grouped.device)

    hessian_grouped = hessian_diag.reshape(n_groups, -1).to(
        device=grouped.device,
        dtype=torch.float32,
    )

    for start in range(0, rows, row_chunk_size):
        end = min(start + row_chunk_size, rows)
        chunk = grouped[start:end].to(torch.float32)
        scale_upper = (chunk.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(eps)

        best_coarse_loss = None
        coarse_best = None
        for candidate_start in range(0, coarse_candidates, candidate_chunk_size):
            candidate_end = min(candidate_start + candidate_chunk_size, coarse_candidates)
            _scale, loss, index = _evaluate_symmetric_scale_factors(
                chunk,
                hessian_grouped,
                scale_upper,
                coarse_factors[candidate_start:candidate_end],
                qmin,
                qmax,
                eps,
            )
            index = index + candidate_start
            if best_coarse_loss is None:
                best_coarse_loss = loss
                coarse_best = index
                continue
            improved = loss < best_coarse_loss
            best_coarse_loss = torch.where(improved, loss, best_coarse_loss)
            coarse_best = torch.where(improved.squeeze(-1), index, coarse_best)

        center = ((coarse_best + 1) * fine_width).clamp(1, scale_candidates)
        half = max(1, fine_width // 2)
        offsets = torch.arange(
            -half,
            half + 1,
            device=grouped.device,
            dtype=torch.long,
        )
        fine_indices = (center.unsqueeze(-1) + offsets).clamp(1, scale_candidates)
        fine_factors = fine_indices.to(torch.float32) / float(scale_candidates)

        best_scale = best_loss = None
        fine_candidates = fine_factors.shape[-1]
        for candidate_start in range(0, fine_candidates, candidate_chunk_size):
            candidate_end = min(candidate_start + candidate_chunk_size, fine_candidates)
            scale, loss, _index = _evaluate_symmetric_scale_factors(
                chunk,
                hessian_grouped,
                scale_upper,
                fine_factors[..., candidate_start:candidate_end],
                qmin,
                qmax,
                eps,
            )
            if best_loss is None:
                best_scale, best_loss = scale, loss
                continue
            improved = loss < best_loss
            best_scale = torch.where(improved, scale, best_scale)
            best_loss = torch.where(improved, loss, best_loss)
        scale_group[start:end] = best_scale

    return scale_group


@torch.no_grad()
def build_neuqi_quantization_grid(
    weights: torch.Tensor,
    stats: LayerStats,
    bits: int,
    group_size: int,
    *,
    scheme: str = "asymmetric",
    scale_candidates: int = 2048,
    coarse_candidates: int = 64,
    row_chunk_size: int = 16,
    candidate_chunk_size: int = 16,
    eps: float = 1e-8,
) -> NeUQIQuantizationGrid:
    """Build a NeUQI-initialized grid for ``weights``.

    ``scheme="asymmetric"`` uses NeUQI's affine uniform quantizer with a
    floating-point zero-point, matching the paper's relaxed formulation.
    ``scheme="symmetric"`` is a repository extension that keeps zero-point
    fixed at 0 and searches the weighted scale/clipping factor on signed
    integer codes.  ``group_size=-1`` means one channel-wise group per row.
    """

    if weights.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got shape {tuple(weights.shape)}")
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if group_size == -1:
        actual_group_size = weights.shape[1]
    elif group_size > 0:
        actual_group_size = group_size
    else:
        raise ValueError(f"group_size must be positive or -1, got {group_size}")
    if scheme not in {"asymmetric", "symmetric"}:
        raise ValueError(f"scheme must be 'asymmetric' or 'symmetric', got {scheme!r}")
    if scheme == "symmetric" and bits < 2:
        raise ValueError("symmetric NeUQI quantization requires bits >= 2")
    if scale_candidates <= 0:
        raise ValueError("scale_candidates must be positive")
    if coarse_candidates <= 0:
        raise ValueError("coarse_candidates must be positive")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    if stats.diag_H.numel() != weights.shape[1]:
        raise ValueError(
            f"stats.diag_H has {stats.diag_H.numel()} values, "
            f"expected {weights.shape[1]}"
        )

    rows, in_features = weights.shape
    device, dtype = weights.device, weights.dtype
    n_groups = (in_features + actual_group_size - 1) // actual_group_size
    padded_in = n_groups * actual_group_size

    padded_weights = _pad_weights(weights, padded_in)
    diag_h = _pad_diag(stats.diag_H.detach(), padded_in).to(device=device)
    diag_h = diag_h.clamp_min(0)
    valid_mask = (torch.arange(padded_in, device=device) < in_features).reshape(
        n_groups,
        actual_group_size,
    )

    grouped = padded_weights.reshape(rows, n_groups, actual_group_size)
    if scheme == "symmetric":
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        scale_group = _search_symmetric_neuqi_params(
            grouped,
            diag_h,
            qmin,
            qmax,
            scale_candidates=scale_candidates,
            coarse_candidates=coarse_candidates,
            row_chunk_size=row_chunk_size,
            candidate_chunk_size=candidate_chunk_size,
            eps=eps,
        )
        zero_group = torch.zeros_like(scale_group)
    else:
        qmin = 0
        qmax = 2**bits - 1
        scale_group, zero_group = _search_neuqi_params(
            grouped,
            diag_h,
            valid_mask,
            qmax,
            scale_candidates=scale_candidates,
            coarse_candidates=coarse_candidates,
            row_chunk_size=row_chunk_size,
            candidate_chunk_size=candidate_chunk_size,
            eps=eps,
        )

    return NeUQIQuantizationGrid(
        float_weights=padded_weights,
        scale=_expand_group(scale_group.to(dtype=dtype), actual_group_size, padded_in),
        zero_point=_expand_group(zero_group.to(dtype=dtype), actual_group_size, padded_in),
        qmin=qmin,
        qmax=qmax,
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        padded_in_features=padded_in,
        original_dtype=dtype,
        scheme=scheme,
        scale_candidates=scale_candidates,
        coarse_candidates=coarse_candidates,
        candidate_chunk_size=candidate_chunk_size,
    )


@torch.no_grad()
def build_asymmetric_neuqi_quantization_grid(
    weights: torch.Tensor,
    stats: LayerStats,
    bits: int,
    group_size: int,
    *,
    scale_candidates: int = 2048,
    coarse_candidates: int = 64,
    row_chunk_size: int = 16,
    candidate_chunk_size: int = 16,
    eps: float = 1e-8,
) -> NeUQIQuantizationGrid:
    return build_neuqi_quantization_grid(
        weights,
        stats,
        bits,
        group_size,
        scheme="asymmetric",
        scale_candidates=scale_candidates,
        coarse_candidates=coarse_candidates,
        row_chunk_size=row_chunk_size,
        candidate_chunk_size=candidate_chunk_size,
        eps=eps,
    )


@torch.no_grad()
def build_symmetric_neuqi_quantization_grid(
    weights: torch.Tensor,
    stats: LayerStats,
    bits: int,
    group_size: int,
    *,
    scale_candidates: int = 2048,
    coarse_candidates: int = 64,
    row_chunk_size: int = 16,
    candidate_chunk_size: int = 16,
    eps: float = 1e-8,
) -> NeUQIQuantizationGrid:
    return build_neuqi_quantization_grid(
        weights,
        stats,
        bits,
        group_size,
        scheme="symmetric",
        scale_candidates=scale_candidates,
        coarse_candidates=coarse_candidates,
        row_chunk_size=row_chunk_size,
        candidate_chunk_size=candidate_chunk_size,
        eps=eps,
    )
