"""Actual rate-3/4 (2048,1536) systematic repeat-accumulate LDPC.

The FSO-SC paper specifies rate 3/4 and block length 2048 (Sec. IV,
printed p.9173), but does not publish its parity-check matrix or decoder.
This module therefore supplies a reproducible *replacement matrix*, not
the undisclosed paper code.  Its systematic repeat-accumulate construction
uses H=[A|D], where A has column weight three and D is lower bidiagonal.
This is the sparse systematic RA/IRA family described in H. Jin's thesis,
Chapter 7: https://thesis.caltech.edu/3191/1/HJinthesis.pdf .  The particular
balanced, seeded, four-cycle-free A below is an implementation assumption;
no published performance claim or optimized degree distribution is implied.

Decoder inputs are genuine soft log(P(bit=0)/P(bit=1)) likelihood ratios.
Parity convergence is not a CRC and does not establish correct recovery.
Only PyTorch plus the Python standard library are required; all iterative
encoding/decoding operations stay on the module's CPU or CUDA device.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct

import torch
from torch import nn


def _construct_rows(seed: int) -> list[list[int]]:
    """Build A with balanced check degrees, excluding all length-4 cycles.

An information column cannot reuse a pair of checks.  Adjacent check
pairs are already occupied by a column of the bidiagonal D, so they are
excluded too.  Local Python RNG keeps global training RNGs untouched.
"""
    m, k = 512, 1536
    rng = random.Random(seed)
    rows: list[list[int]] = [[] for _ in range(m)]
    degrees = [0] * m
    used_pairs = {(i, i + 1) for i in range(m - 1)}
    for variable in range(k):
        selected: list[int] = []
        for _ in range(3):
            candidates = [
                check for check in range(m)
                if check not in selected
                and all(tuple(sorted((check, old))) not in used_pairs for old in selected)
            ]
            if not candidates:
                raise RuntimeError("LDPC construction exhausted four-cycle-free candidates")
            minimum = min(degrees[check] for check in candidates)
            chosen = rng.choice([check for check in candidates if degrees[check] == minimum])
            selected.append(chosen)
            degrees[chosen] += 1
            rows[chosen].append(variable)
        for a in range(3):
            for b in range(a + 1, 3):
                used_pairs.add(tuple(sorted((selected[a], selected[b]))))
    for check in range(m):
        rows[check].append(k + check)
        if check:
            rows[check].append(k + check - 1)
        rows[check].sort()
    return rows


class SystematicRALDPC(nn.Module):
    """Fixed dimensions; batched systematic encoder and soft min-sum decoder.

    ``decode_llr`` returns ``(information_bits, stats)``. All statistics are
    tensors on the input device; iterations are zero for initially valid
    codewords. A successful syndrome does not rule out undetected errors.
    ``seed``, normalization, iteration limit and LLR clipping must be fixed
    before any image-quality sweep, not selected to fit reported curves.
    """

    n = 2048
    k = 1536
    m = 512
    rate = 0.75

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        normalization: float = 0.8,
        seed: int = 20261028,
        llr_clip: float = 30.0,
    ) -> None:
        super().__init__()
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not math.isfinite(float(normalization)) or not 0.0 < normalization <= 1.0:
            raise ValueError("normalization must be finite and in (0,1]")
        if not math.isfinite(float(llr_clip)) or llr_clip <= 0:
            raise ValueError("llr_clip must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.max_iterations = max_iterations
        self.normalization = float(normalization)
        self.seed = seed
        self.llr_clip = float(llr_clip)
        rows = _construct_rows(seed)
        self.edge_count = sum(map(len, rows))
        # Canonical byte layout: row-major (check, variable) uint16 little endian.
        canonical = b"".join(struct.pack("<HH", i, j) for i, row in enumerate(rows) for j in row)
        self.matrix_sha256 = hashlib.sha256(canonical).hexdigest()
        self._row_degree_range = (min(map(len, rows)), max(map(len, rows)))
        width = max(map(len, rows))
        neighbors = torch.zeros((self.m, width), dtype=torch.long)
        valid = torch.zeros((self.m, width), dtype=torch.bool)
        information_mask = torch.zeros((self.m, width), dtype=torch.bool)
        for check, row in enumerate(rows):
            neighbors[check, :len(row)] = torch.tensor(row, dtype=torch.long)
            valid[check, :len(row)] = True
            information_mask[check, :len(row)] = torch.tensor([v < self.k for v in row])
        self.register_buffer("check_variables", neighbors)
        self.register_buffer("check_mask", valid)
        self.register_buffer("information_mask", information_mask)

    def metadata(self) -> dict:
        return {
            "name": "systematic_ra_2048_1536_seeded_v1",
            "n": self.n, "k": self.k, "m": self.m, "rate": self.rate,
            "seed": self.seed, "matrix_sha256": self.matrix_sha256,
            "matrix_hash_encoding": "row-major (check,variable), uint16 little endian pairs",
            "edge_count": self.edge_count,
            "information_column_weight": 3,
            "check_degree_min": self._row_degree_range[0],
            "check_degree_max": self._row_degree_range[1],
            "four_cycles": 0,
            "decoder": "flooding normalized min-sum with syndrome early stopping",
            "normalization": self.normalization,
            "max_iterations": self.max_iterations,
            "llr_clip": self.llr_clip,
            "llr_convention": "log(P0/P1); positive means bit zero",
            "paper_dimensions_matched": True,
            "paper_matrix_reproduced": False,
            "matrix_assumption": "H=[seeded balanced column-weight-3 A | lower-bidiagonal D]",
            "construction_reference": "H. Jin thesis, Chapter 7, https://thesis.caltech.edu/3191/1/HJinthesis.pdf",
            "convergence_is_crc": False,
        }

    def _check_input(self, value: torch.Tensor, width: int, *, bits: bool) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"expected a tensor with shape [batch,{width}]")
        if value.device != self.check_variables.device:
            raise ValueError("input and LDPC module must share a device; use ldpc.to(device)")
        if value.is_complex():
            raise ValueError("expected real bits or real-valued LLRs")
        if bits:
            if not bool(((value == 0) | (value == 1)).all()):
                raise ValueError("bits must contain only zeros and ones")
        elif not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("LLRs must be finite floating-point values")

    @torch.no_grad()
    def encode_bits(self, bits: torch.Tensor) -> torch.Tensor:
        self._check_input(bits, self.k, bits=True)
        # Padding and parity neighbor slots are masked, not included in Au.
        information_indices = self.check_variables.clamp_max(self.k - 1)
        local = bits[:, information_indices].to(torch.int64)
        au = (local * self.information_mask).sum(dim=-1).remainder(2)
        parity = au.cumsum(dim=1).remainder(2).to(torch.uint8)
        return torch.cat((bits.to(torch.uint8), parity), dim=1)

    @torch.no_grad()
    def syndrome(self, codeword: torch.Tensor) -> torch.Tensor:
        self._check_input(codeword, self.n, bits=True)
        return self._syndrome_unchecked(codeword)

    def _syndrome_unchecked(self, codeword: torch.Tensor) -> torch.Tensor:
        return (codeword[:, self.check_variables].to(torch.int64) * self.check_mask).sum(-1).remainder(2).to(torch.uint8)

    @torch.no_grad()
    def decode_llr(self, llr: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._check_input(llr, self.n, bits=False)
        channel = llr.to(torch.float32).clamp(-self.llr_clip, self.llr_clip)
        batch = channel.shape[0]
        posterior = channel.clone()
        hard = (posterior < 0).to(torch.uint8)
        initial_weight = self._syndrome_unchecked(hard).sum(dim=1)
        syndrome_weight = initial_weight.clone()
        converged = initial_weight == 0
        iterations = torch.zeros(batch, dtype=torch.int64, device=channel.device)
        r = torch.zeros((batch,) + tuple(self.check_variables.shape), device=channel.device)
        q = channel[:, self.check_variables]
        flat_indices = self.check_variables.reshape(1, -1).expand(batch, -1)
        slots = torch.arange(self.check_variables.shape[1], device=channel.device)
        for iteration in range(1, self.max_iterations + 1):
            if bool(converged.all()):
                break
            active = ~converged
            signs = torch.where(q < 0, -1.0, 1.0)
            signs = torch.where(self.check_mask, signs, 1.0)
            abs_q = torch.where(self.check_mask, q.abs(), float("inf"))
            minima, locations = abs_q.topk(2, dim=-1, largest=False, sorted=True)
            excluded_minimum = torch.where(slots == locations[..., :1], minima[..., 1:2], minima[..., :1])
            new_r = self.normalization * signs.prod(-1, keepdim=True) * signs * excluded_minimum
            new_r = torch.where(self.check_mask, new_r, 0.0)
            # Freeze converged blocks. They must not drift while others iterate.
            r = torch.where(active[:, None, None], new_r, r)
            candidate = channel.clone().scatter_add_(1, flat_indices, r.reshape(batch, -1))
            posterior = torch.where(active[:, None], candidate, posterior)
            hard = (posterior < 0).to(torch.uint8)
            syndrome_weight = self._syndrome_unchecked(hard).sum(dim=1)
            iterations[active] = iteration
            converged = converged | (syndrome_weight == 0)
            q = (posterior[:, self.check_variables] - r).clamp(-self.llr_clip, self.llr_clip)
        return hard[:, :self.k], {
            "converged": converged,
            "iterations": iterations,
            "syndrome_weight": syndrome_weight,
            "initial_syndrome_weight": initial_weight,
        }
