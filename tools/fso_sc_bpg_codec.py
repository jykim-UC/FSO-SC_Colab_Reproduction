"""Real libbpg CLI transport, with no subprocess/network work at import.

Provenance: https://bellard.org/bpg/ ; official libbpg-0.9.8.tar.gz SHA256
c0788e23bdf1a7d36cb4424ccb2fae4c7789ac94949563c4ad0e2569d3bf0095.
The 256x256 input/output are RGB uint8. Compression explicitly uses x265,
8-bit YCbCr 4:2:0, full range, no alpha/metadata, and lossy q=0..51.

Example (the PNG input must already be RGB8 256x256)::

    bpgenc -e x265 -b 8 -c ycbcr -f 420 -m 4 -q 28 -o image.bpg image.png
    bpgdec -b 8 -o recovered.png image.bpg

Linux source build requires a 64-bit target, make, gcc/g++, cmake, yasm,
libpng-dev, libjpeg-dev and zlib1g-dev. Build only the two utilities with
``make -j2 USE_BPGVIEW= CMAKE_OPTS=-DENABLE_LIBNUMA=OFF bpgenc bpgdec``;
this avoids the optional SDL viewer. The official source bundles its x265
fork: replacing that fork with a current system libx265 is not equivalent.
The verified build pins CMake3.28.3:3.31+ also breaks its old ASM_YASM linker
module, while CMake4 removes its CMP0025/CMP0054 OLD behavior. The installer
configures every variant explicitly so failed configure caches can recover.
Yasm cannot simply be disabled on x86: that x265 source rejects doing so.
The official README supports Windows cross-compilation from Linux, not a
native MSVC build; official portable Windows binaries are also provided.
Building/installing/downloads are intentionally the caller's responsibility.
"""

from dataclasses import dataclass
import hashlib
import math
import operator
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError


BPG_CODEC_VERSION = "libbpg_cli_rgb8_v1"
BPG_MAGIC = b"BPG\xfb"
LIBBPG_SOURCE_URL = "https://bellard.org/bpg/libbpg-0.9.8.tar.gz"
LIBBPG_SOURCE_SHA256 = "c0788e23bdf1a7d36cb4424ccb2fae4c7789ac94949563c4ad0e2569d3bf0095"


class BPGError(RuntimeError):
    """Fatal codec setup/encoding problem, not an ordinary damaged frame."""


class BPGExecutableNotFound(BPGError):
    pass


class BPGEncodeError(BPGError):
    pass


class BPGRateBudgetError(BPGError):
    """No q in the full lossy range fits; never silently increases the budget."""

    def __init__(self, max_bytes, tested_q_sizes):
        self.max_bytes = max_bytes
        self.tested_q_sizes = tuple(sorted(tested_q_sizes))
        self.minimum_size_q, self.minimum_size = min(self.tested_q_sizes, key=lambda item: (item[1], item[0]))
        super().__init__(f"No lossy BPG q=0..51 fits {max_bytes} bytes; minimum measured "
                         f"size is {self.minimum_size} bytes at q={self.minimum_size_q}")


@dataclass(frozen=True)
class EncodedBPG:
    data: bytes
    q: int
    max_bytes: int
    image_hw: tuple
    tested_q_sizes: tuple
    search_mode: str
    optimality: str
    encoder_settings: dict
    cache_hits: int = 0

    @property
    def num_bytes(self):
        return len(self.data)

    @property
    def bpp(self):
        return 8 * len(self.data) / (self.image_hw[0] * self.image_hw[1])

    def metadata(self):
        return {"codec": BPG_CODEC_VERSION, "q": self.q, "actual_bytes": self.num_bytes,
                "actual_bpp": self.bpp, "budget_bytes": self.max_bytes,
                "bpg_header_included": True, "image_hw": list(self.image_hw),
                "search_mode": self.search_mode, "optimality": self.optimality,
                "tested_q_sizes": {str(q): size for q, size in self.tested_q_sizes},
                "tested_q_count": len(self.tested_q_sizes), "cache_hits": self.cache_hits,
                "encoder_settings": dict(self.encoder_settings),
                "source_url": LIBBPG_SOURCE_URL, "source_sha256": LIBBPG_SOURCE_SHA256}


@dataclass(frozen=True)
class DecodeResult:
    status: str
    image: object = None
    error: str = ""
    returncode: object = None

    @property
    def ok(self):
        return self.status == "ok"


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def bytes_to_bits(data):
    """Every byte, including the BPG header, in MSB-first transmission order."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8), bitorder="big")


def bits_to_bytes(bits):
    """Reverse bytes_to_bits; rejects truncation, nonbinary values and padding."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size % 8:
        raise ValueError("bits must be one-dimensional and byte-aligned")
    if not np.isin(bits, [0, 1]).all():
        raise ValueError("bits must contain only 0 or 1")
    return np.packbits(bits.astype(np.uint8), bitorder="big").tobytes()


def _read_ue7(data, offset):
    value = 0
    for i in range(5):
        if offset >= len(data):
            raise ValueError("Truncated BPG header")
        byte = data[offset]
        offset += 1
        if i == 0 and byte == 0x80:
            raise ValueError("Noncanonical BPG header integer")
        value = (value << 7) | (byte & 127)
        if value > 0xffffffff:
            raise ValueError("Oversized BPG header integer")
        if not byte & 128:
            return value, offset
    raise ValueError("Oversized BPG header integer")


def _validate_header(data, expected_hw):
    """Bound dimensions before handing an untrusted bitstream to native code."""
    if len(data) < 9 or not data.startswith(BPG_MAGIC):
        raise ValueError("Missing or truncated BPG header")
    if (data[4] & 15) != 0:
        raise ValueError("Expected an 8-bit BPG stream")
    if data[4] >> 5 > 5 or data[5] >> 4 > 4:
        raise ValueError("Invalid BPG pixel format/color space")
    if data[5] & 1:
        raise ValueError("Animated BPG is not supported by the still-image baseline")
    width, offset = _read_ue7(data, 6)
    height, offset = _read_ue7(data, offset)
    if (height, width) != expected_hw:
        raise ValueError(f"BPG dimensions {(height, width)} do not match {expected_hw}")
    payload_size, offset = _read_ue7(data, offset)
    if data[5] & 8:
        extension_size, offset = _read_ue7(data, offset)
        offset += extension_size
    if offset >= len(data) or (payload_size and payload_size > len(data) - offset):
        raise ValueError("Truncated BPG payload/extension")


def _command_prefix(command):
    if isinstance(command, (str, os.PathLike)):
        result = [os.fspath(command)]
    else:
        result = [os.fspath(item) for item in command]
    if not result or not result[0]:
        raise ValueError("Executable command cannot be empty")
    executable = shutil.which(result[0])
    if executable is None:
        raise BPGExecutableNotFound(f"BPG executable not found: {result[0]}. "
                                    f"Install official libbpg: {LIBBPG_SOURCE_URL}")
    result[0] = str(Path(executable).resolve())
    return result


def _run(command, directory, timeout):
    # Files prevent an invalid/native decoder's diagnostics from filling RAM.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.run(command, cwd=directory, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, timeout=timeout, check=False)
        except FileNotFoundError as exc:
            raise BPGExecutableNotFound(f"BPG executable disappeared: {command[0]}") from exc
        except OSError as exc:
            raise BPGError(f"Cannot launch BPG executable {command[0]}: {exc}") from exc
        stderr.seek(0, os.SEEK_END)
        length = stderr.tell()
        stderr.seek(max(0, length - 4096))
        detail = stderr.read(4096).decode("utf-8", errors="replace").strip()
        return process.returncode, detail


class BPGCodec:
    """Bounded native codec wrapper; a one-image cache avoids duplicate q work.

    Missing/unlaunchable executables are fatal. Damaged received streams return
    DecodeResult failures; no image or score is synthesized by the codec. The
    evaluator must declare its failure convention and retain all frames. The
    paper's explicit LPIPS=1 failure marker and a separately measured gray
    reconstruction metric are permitted, provided they are clearly separated.
    Success requires exit status zero AND a valid expected-size RGB8 image;
    a nonzero exit is failure even if the process left a partial PNG behind.

    binary_checked picks the best *tested* feasible q without claiming global
    optimality for nonmonotonic sizes. exhaustive proves the smallest feasible
    q. If q51 does not fit, binary_checked falls back to exhaustive rather than
    falsely reporting that no q can fit.
    """

    def __init__(self, encoder="bpgenc", decoder="bpgdec", timeout=30.0,
                 expected_hw=(256, 256), speed=4):
        self.encoder = _command_prefix(encoder)
        self.decoder = _command_prefix(decoder)
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if len(expected_hw) != 2:
            raise ValueError("expected_hw must be (height, width)")
        self.expected_hw = tuple(_integer(v, "image dimension", 1, 16384) for v in expected_hw)
        self.speed = _integer(speed, "speed", 1, 9)
        self._cached_image = None
        self._encoded = {}
        self._cache_hits = 0

    @property
    def encoder_settings(self):
        return {"encoder": "x265", "bit_depth": 8, "color_space": "ycbcr",
                "chroma_format": "420", "speed": self.speed, "lossless": False,
                "input_mode": "RGB", "output_dtype": "uint8", "metadata": False}

    def _prepare_image(self, image):
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("image must be a PIL RGB image")
        if image.size != self.expected_hw[::-1]:
            raise ValueError(f"Input dimensions must be {self.expected_hw[::-1]}")
        key = hashlib.sha256(image.tobytes()).digest(), self.speed
        if key != self._cached_image:
            self._cached_image, self._encoded = key, {}
        return image

    def encode(self, image, q):
        """Return genuine complete .bpg bytes at one lossy quantizer value."""
        image = self._prepare_image(image)
        q = _integer(q, "q", 0, 51)
        if q in self._encoded:
            self._cache_hits += 1
            return self._encoded[q]
        with tempfile.TemporaryDirectory(prefix="fso_bpg_encode_") as directory:
            source, output = Path(directory) / "source.png", Path(directory) / "encoded.bpg"
            # A fresh RGB image deliberately strips ICC/EXIF and alpha metadata.
            Image.frombytes("RGB", image.size, image.tobytes()).save(source, format="PNG")
            command = self.encoder + ["-e", "x265", "-b", "8", "-c", "ycbcr", "-f", "420",
                                      "-m", str(self.speed), "-q", str(q), "-o", str(output), str(source)]
            try:
                returncode, detail = _run(command, directory, self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise BPGEncodeError(f"bpgenc timed out at q={q}") from exc
            if returncode != 0:
                raise BPGEncodeError(f"bpgenc exited {returncode} at q={q}: {detail}")
            if not output.is_file():
                raise BPGEncodeError(f"bpgenc produced no output at q={q}")
            data = output.read_bytes()
            try:
                _validate_header(data, self.expected_hw)
            except ValueError as exc:
                raise BPGEncodeError(f"Invalid bpgenc output at q={q}: {exc}") from exc
        self._encoded[q] = data
        return data

    def encode_to_budget(self, image, max_bytes, search_mode="binary_checked"):
        """Budget includes every BPG byte, but not caller-owned transport framing."""
        self._prepare_image(image)
        max_bytes = _integer(max_bytes, "max_bytes", 1)
        if search_mode not in ("binary_checked", "exhaustive"):
            raise ValueError("search_mode must be binary_checked or exhaustive")
        tested, cache_hits_before = {}, self._cache_hits

        def measure(q):
            if q not in tested:
                tested[q] = self.encode(image, q)
            return len(tested[q]) <= max_bytes

        mode = search_mode
        if search_mode == "exhaustive":
            for q in range(52):
                if measure(q):
                    break
        elif not measure(0):
            if not measure(51):
                mode = "exhaustive_fallback"
                for q in range(1, 51):
                    if measure(q):
                        break
            else:
                lo, hi = 0, 51
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if measure(mid):
                        hi = mid
                    else:
                        lo = mid
                best = min(q for q, data in tested.items() if len(data) <= max_bytes)
                for q in range(max(0, best - 2), min(51, best + 2) + 1):
                    measure(q)
        # Previously measured q values for this exact image are evidence too:
        # never ignore a better known feasible stream when reusing the cache.
        tested.update(self._encoded)
        feasible = [q for q, data in tested.items() if len(data) <= max_bytes]
        sizes = tuple(sorted((q, len(data)) for q, data in tested.items()))
        if not feasible:
            # The q51 failure route has measured every q before reaching here.
            raise BPGRateBudgetError(max_bytes, sizes)
        q = min(feasible)
        proven = mode != "binary_checked" or q == 0
        optimality = ("lowest feasible q over 0..51 (all lower q measured)" if proven else
                      "lowest tested feasible q; not exhaustive/global under nonmonotonic size")
        return EncodedBPG(tested[q], q, max_bytes, self.expected_hw, sizes, mode, optimality,
                          self.encoder_settings, self._cache_hits - cache_hits_before)

    def decode(self, data):
        """Decode one hard received stream; no fallback is synthesized here."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        data = bytes(data)
        try:
            _validate_header(data, self.expected_hw)
        except ValueError as exc:
            return DecodeResult("invalid_stream", error=str(exc))
        with tempfile.TemporaryDirectory(prefix="fso_bpg_decode_") as directory:
            source, output = Path(directory) / "received.bpg", Path(directory) / "decoded.png"
            source.write_bytes(data)
            try:
                returncode, detail = _run(self.decoder + ["-b", "8", "-o", str(output), str(source)],
                                          directory, self.timeout)
            except subprocess.TimeoutExpired:
                return DecodeResult("timeout", error=f"bpgdec exceeded {self.timeout:g}s")
            if returncode != 0:
                return DecodeResult("decoder_exit", error=detail or "bpgdec failed", returncode=returncode)
            if not output.is_file():
                return DecodeResult("invalid_output", error="bpgdec produced no output")
            # A valid RGB8 PNG is comfortably smaller; reject runaway output.
            maximum_png_bytes = self.expected_hw[0] * self.expected_hw[1] * 8 + 65536
            if output.stat().st_size > maximum_png_bytes:
                return DecodeResult("invalid_output", error="bpgdec output exceeds sanity bound")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(output) as decoded:
                        if decoded.format != "PNG" or decoded.size != self.expected_hw[::-1]:
                            return DecodeResult("invalid_output", error="Wrong output format/dimensions")
                        if decoded.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                            return DecodeResult("invalid_output", error="Expected an 8-bit PNG output")
                        decoded.load()
                        pixels = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError,
                    Image.DecompressionBombWarning) as exc:
                return DecodeResult("invalid_output", error=f"Unreadable decoder image: {exc}")
            if pixels.shape != self.expected_hw + (3,) or pixels.dtype != np.uint8:
                return DecodeResult("invalid_output", error="Invalid RGB uint8 reconstruction")
            return DecodeResult("ok", pixels, returncode=0)
