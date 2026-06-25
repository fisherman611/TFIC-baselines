from __future__ import annotations

import itertools

import torch

from eigenflip.encoders.tfic_fast import TFICEncoder


def test_frustration_field_chunked_matches_full_formula():
    torch.manual_seed(0)
    rows, pin, top_m = 5, 9, 3
    delta = torch.randn(rows, pin)
    G = torch.randn(pin, pin)
    G = 0.5 * (G + G.t())
    Gabs = G.abs().clone()
    Gabs.fill_diagonal_(0.0)
    _values, nbr_idx = torch.topk(Gabs, top_m, dim=1)
    G_nbr = G[torch.arange(pin).unsqueeze(1), nbr_idx]
    tau = 0.37

    full = (
        (-2.0 * delta.unsqueeze(2) * delta[:, nbr_idx] * G_nbr.unsqueeze(0))
        .clamp_min(0.0)
        .sum(2)
        / tau
    ).clamp(max=1.0)
    chunked = TFICEncoder._frustration_field(
        delta, G_nbr, nbr_idx, tau, chunk_cols=2
    )

    assert torch.allclose(chunked, full)


def test_zero_pre_round_difference_is_not_a_valid_flip():
    Wint = torch.tensor([[1.0, 2.0, 3.0]])
    flip_dir = TFICEncoder._flip_dir(Wint.clone(), Wint)
    in_range = TFICEncoder._in_range(Wint, flip_dir, max_int=7.0)

    assert torch.count_nonzero(flip_dir) == 0
    assert not in_range.any()


def test_certified_fixes_match_bruteforce_global_minimizers():
    G = torch.eye(4)
    diagG = torch.diagonal(G)
    scale = torch.tensor([[0.5, 0.7, 0.4, 0.6]], dtype=torch.float32)
    pre = torch.tensor([[1.2, 1.8, 2.25, 2.7]], dtype=torch.float32)
    Wint = torch.round(pre)
    Wf = pre * scale
    R = Wint * scale - Wf
    RG = R @ G
    flip_dir = TFICEncoder._flip_dir(pre, Wint)
    delta = flip_dir * scale
    in_range = TFICEncoder._in_range(Wint, flip_dir, max_int=7.0)
    dE = TFICEncoder._dE(delta, RG, diagG, in_range)

    fixed, fix_now = TFICEncoder._certified_fixes(
        G, diagG, scale, pre, Wint, dE, in_range
    )
    certified = fixed | fix_now
    assert certified.any()

    H = 0.5 * scale[0]
    s_cur = -torch.sign(pre[0] - Wint[0])
    D = H * s_cur - R[0]
    movable_cols = torch.nonzero(
        in_range[0] & (s_cur != 0) & (scale[0] > 0), as_tuple=False
    ).flatten()
    certified_targets = torch.where(fix_now[0], -s_cur, s_cur)

    energies: list[tuple[float, torch.Tensor]] = []
    for values in itertools.product((-1.0, 1.0), repeat=movable_cols.numel()):
        s = s_cur.clone()
        s[movable_cols] = torch.tensor(values, dtype=s.dtype)
        residual = H * s - D
        energy = float(residual @ G @ residual)
        energies.append((energy, s))

    best = min(energy for energy, _s in energies)
    minimizers = [s for energy, s in energies if abs(energy - best) < 1e-7]
    for col in torch.nonzero(certified[0], as_tuple=False).flatten().tolist():
        assert all(s[col] == certified_targets[col] for s in minimizers)
