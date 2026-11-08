"""Rate-only, dataset-mean matching of genuine native BPG streams.

The official libbpg 0.9.8 CLI parses ``-q`` with ``atoi`` and its x265 glue
uses integer constant-QP mode; fractional QP is not an exposed rate control.
Source: https://bellard.org/bpg/libbpg-0.9.8.tar.gz (bpgenc.c, x265_glue.c).

For each image we retain an under-target native stream and the first tested
lower-QP stream above the target. A deterministic subset-sum then promotes
some images using *only encoded byte counts*. No reconstruction, metric,
SNR, decoder outcome, padding, metadata stuffing, or source-image change is
used. Freeze the selected streams before evaluating any channel condition.

This matches the DATASET MEAN, not each image's rate. Individual streams may
exceed the per-image reference. Report their actual minimum/maximum/rates.
Subset-sum is exact over the retained candidates, not over all possible
codec settings. The default native QP search remains explicitly non-global.
The legacy strict-budget codec is deliberately unchanged.
"""

from dataclasses import dataclass
import hashlib
import math
import operator
from types import MappingProxyType

from fso_sc_bpg_codec import BPG_MAGIC, BPG_CODEC_VERSION


RATE_CONTROL_VERSION = "bpg_native_integer_qp_dataset_mean_v1"


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class NativeBPGCandidate:
    data: bytes
    q: int
    image_hw: tuple

    def __post_init__(self):
        if not isinstance(self.data, bytes) or not self.data.startswith(BPG_MAGIC):
            raise ValueError("candidate must contain complete native BPG bytes")
        if isinstance(self.q, bool) or not isinstance(self.q, int) or not 0 <= self.q <= 51:
            raise ValueError("candidate q must be an integer from 0 to 51")
        if len(self.image_hw) != 2 or any(_positive_integer(v, "image dimension") != v for v in self.image_hw):
            raise ValueError("candidate image_hw must contain two positive integers")

    @property
    def num_bytes(self):
        return len(self.data)

    @property
    def bpp(self):
        return 8 * self.num_bytes / math.prod(self.image_hw)

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()

    def metadata(self):
        return {"q": self.q, "actual_bytes": self.num_bytes, "actual_bpp": self.bpp,
                "image_hw": list(self.image_hw), "sha256": self.sha256,
                "bpg_header_included": True, "padding_bytes": 0}


@dataclass(frozen=True)
class BPGRateBracket:
    image_name: str
    target_bytes: int
    low: NativeBPGCandidate
    high: object
    tested_q_sizes: tuple
    search_mode: str
    optimality: str
    encoder_settings: dict
    cache_hits: int = 0

    def __post_init__(self):
        if not isinstance(self.image_name, str) or not self.image_name:
            raise ValueError("image_name must be a nonempty string")
        _positive_integer(self.target_bytes, "target_bytes")
        if self.low.num_bytes > self.target_bytes:
            raise ValueError("low candidate exceeds the reference target")
        if self.high is not None:
            if self.high.num_bytes <= self.target_bytes or self.high.q >= self.low.q:
                raise ValueError("high candidate must exceed target at a lower integer QP")
            if self.high.image_hw != self.low.image_hw:
                raise ValueError("candidate image dimensions differ")

    def metadata(self):
        return {"image_name": self.image_name, "target_bytes": self.target_bytes,
                "low": self.low.metadata(),
                "high": self.high.metadata() if self.high is not None else None,
                "tested_q_sizes": {str(q): size for q, size in self.tested_q_sizes},
                "tested_q_count": len(self.tested_q_sizes), "search_mode": self.search_mode,
                "optimality": self.optimality, "encoder_settings": dict(self.encoder_settings),
                "cache_hits": self.cache_hits}


def encode_rate_bracket(codec, image, image_name, target_bytes=1792,
                        search_mode="binary_checked"):
    """Measure an under/over integer-QP pair without decoding or scoring.

    ``low`` means fewer bytes, not lower QP. The higher-rate candidate is
    the first actual over-target stream encountered below low.q. Already
    tested same-image results are reused by the unmodified native codec.
    ``exhaustive`` proves the lowest feasible QP; the default does not.
    """
    target_bytes = _positive_integer(target_bytes, "target_bytes")
    encoded = codec.encode_to_budget(image, target_bytes, search_mode=search_mode)
    low = NativeBPGCandidate(encoded.data, encoded.q, tuple(encoded.image_hw))
    sizes = dict(encoded.tested_q_sizes)
    high = None
    cache_before = getattr(codec, "_cache_hits", 0)
    for q in range(low.q - 1, -1, -1):
        data = codec.encode(image, q)
        sizes[q] = len(data)
        candidate = NativeBPGCandidate(data, q, tuple(encoded.image_hw))
        if candidate.num_bytes > target_bytes:
            high = candidate
            break
        low = candidate
    return BPGRateBracket(image_name, target_bytes, low, high, tuple(sorted(sizes.items())),
                          encoded.search_mode, encoded.optimality,
                          dict(encoded.encoder_settings), encoded.cache_hits +
                          getattr(codec, "_cache_hits", 0) - cache_before)


@dataclass(frozen=True)
class SelectedBPG:
    candidate: NativeBPGCandidate
    bracket: BPGRateBracket
    selected_level: str

    @property
    def data(self):
        return self.candidate.data

    @property
    def q(self):
        return self.candidate.q

    @property
    def num_bytes(self):
        return self.candidate.num_bytes

    @property
    def bpp(self):
        return self.candidate.bpp

    def metadata(self):
        result = self.candidate.metadata()
        result.update(codec=BPG_CODEC_VERSION, rate_control=RATE_CONTROL_VERSION,
                      rate_matching_scope="dataset_mean_not_per_image",
                      per_image_reference_bytes=self.bracket.target_bytes,
                      per_image_hard_cap_bytes=None, selected_level=self.selected_level,
                      tested_q_sizes={str(q): size for q, size in self.bracket.tested_q_sizes},
                      tested_q_count=len(self.bracket.tested_q_sizes),
                      search_mode=self.bracket.search_mode, optimality=self.bracket.optimality,
                      cache_hits=self.bracket.cache_hits,
                      encoder_settings=dict(self.bracket.encoder_settings),
                      bracket=self.bracket.metadata())
        return result


@dataclass(frozen=True)
class MatchedBPGSources:
    entries: object
    target_bytes_per_image: int
    seed: int
    subset_sum_states: int

    def metadata(self):
        entries = list(self.entries.values())
        sizes = [entry.num_bytes for entry in entries]
        target_total = len(entries) * self.target_bytes_per_image
        total = sum(sizes)
        total_pixels = sum(math.prod(entry.candidate.image_hw) for entry in entries)
        manifest = "\n".join(f"{name}\t{entry.q}\t{entry.candidate.sha256}"
                             for name, entry in sorted(self.entries.items()))
        return {"rate_control": RATE_CONTROL_VERSION, "selection_inputs": ["image_name", "encoded_byte_count"],
                "quality_metric_used_for_selection": False, "snr_used_for_selection": False,
                "rate_matching_scope": "dataset_mean_not_per_image", "num_images": len(entries),
                "target_bytes_per_image": self.target_bytes_per_image, "target_total_bytes": target_total,
                "actual_total_bytes": total, "aggregate_shortfall_bytes": target_total - total,
                "mean_actual_bytes": total / len(entries), "actual_mean_bpp": total * 8 / total_pixels,
                "target_mean_bpp": target_total * 8 / total_pixels,
                "min_actual_bytes": min(sizes), "max_actual_bytes": max(sizes),
                "min_actual_bpp": min(entry.bpp for entry in entries),
                "max_actual_bpp": max(entry.bpp for entry in entries),
                "per_image_hard_cap_bytes": None,
                "images_above_per_image_reference": sum(size > self.target_bytes_per_image for size in sizes),
                "selected_high_count": sum(entry.selected_level == "high" for entry in entries),
                "selected_low_count": sum(entry.selected_level == "low" for entry in entries),
                "padding_bytes": 0, "seed": self.seed, "subset_sum_states": self.subset_sum_states,
                "aggregate_optimality": "largest achievable total <= target over retained native candidate pairs",
                "qp_search_global_optimality_claimed": False,
                "source_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
                "reuse_policy": "freeze this exact mapping across both BPG branches and every channel condition"}


def select_rate_matched_sources(brackets, target_bytes_per_image=1792, seed=20261028):
    """Exact bounded subset-sum on measured size increments, never quality.

    Ties use seeded image-name hashes, independently of caller iteration
    order. The aggregate never exceeds N*target, although individual images
    can exceed target. An unattainable target is reported as a real shortfall.
    Runtime/memory scale with aggregate shortfall, not with image pixels.
    """
    target = _positive_integer(target_bytes_per_image, "target_bytes_per_image")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    try:
        seed = operator.index(seed)
    except TypeError as exc:
        raise ValueError("seed must be an integer") from exc
    brackets = list(brackets.values() if hasattr(brackets, "values") else brackets)
    if not brackets:
        raise ValueError("at least one rate bracket is required")
    names = [bracket.image_name for bracket in brackets]
    if len(set(names)) != len(names):
        raise ValueError("image names must be unique")
    if any(bracket.target_bytes != target for bracket in brackets):
        raise ValueError("all brackets must use the requested target")
    ordered = sorted(brackets, key=lambda item: (hashlib.sha256(
        f"{seed}\0{item.image_name}".encode()).digest(), item.image_name))
    deficit = len(ordered) * target - sum(bracket.low.num_bytes for bracket in ordered)
    mask = (1 << (deficit + 1)) - 1
    reachable, history = 1, [1]
    increments = []
    for bracket in ordered:
        increment = bracket.high.num_bytes - bracket.low.num_bytes if bracket.high is not None else 0
        increments.append(increment)
        # A huge out-of-range increment must not allocate a huge bigint.
        if 0 < increment <= deficit:
            reachable = (reachable | reachable << increment) & mask
        history.append(reachable)
    remaining = reachable.bit_length() - 1
    selected = {}
    for index in range(len(ordered) - 1, -1, -1):
        bracket = ordered[index]
        use_high = not ((history[index] >> remaining) & 1)
        candidate = bracket.high if use_high else bracket.low
        if use_high:
            remaining -= increments[index]
        selected[bracket.image_name] = SelectedBPG(candidate, bracket, "high" if use_high else "low")
    if remaining != 0:
        raise AssertionError("subset-sum reconstruction failed")
    selected = MappingProxyType(dict(sorted(selected.items())))
    return MatchedBPGSources(selected, target, seed, reachable.bit_count())


def prepare_rate_matched_sources(codec, images, target_bytes_per_image=1792,
                                 seed=20261028, search_mode="binary_checked", progress=None):
    """Encode then freeze one mean-matched source mapping.

    ``images`` is a mapping or iterable of (name, PIL RGB image). ``progress``
    receives (completed_count, total_count, bracket) after each native image;
    it is a status callback and must not mutate the source-selection policy.
    For parallel/pre-cached preparation call encode_rate_bracket separately
    and pass the complete fixed set to select_rate_matched_sources.
    """
    items = list(images.items() if hasattr(images, "items") else images)
    if len({name for name, _ in items}) != len(items):
        raise ValueError("image names must be unique")
    brackets = []
    for name, image in items:
        bracket = encode_rate_bracket(codec, image, name, target_bytes_per_image, search_mode)
        brackets.append(bracket)
        if progress is not None:
            progress(len(brackets), len(items), bracket)
    return select_rate_matched_sources(brackets, target_bytes_per_image, seed)
