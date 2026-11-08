"""Read plotted markers from Chen et al.'s PDF; do not redistribute the PDF.

Usage: python tools/digitize_paper_gallery.py /path/to/paper.pdf --output config/paper_gallery_reference.json
Requires pypdf; the published reference JSON is sufficient for plotting.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

from pypdf import PdfReader
from pypdf.generic import ContentStream


PAPER_SHA256 = "bd5dd13f193b0076c0181ec4565a8dda1b24ac3494dc5b06c2062ef78440be59"
DOI = "https://doi.org/10.1109/TCOMM.2025.3562356"

# PDF-device CMYK, marker shape, paint operation. Labels were identified from
# the published legends; they cannot be inferred from the CMYK values alone.
FIG4 = {
    "fso_sc_qpsk": ((.648438, .5625, .546875, .313477), "square", b"S"),
    "fso_sc_8qam": ((.0345459, .869141, .691406, .00177002), "diamond", b"f*"),
    "fso_sc_16qam": ((.824219, .556641, 0., 0.), "triangle", b"f*"),
    "bpg_qpsk": ((.730469, .0944824, .726563, .00280762), "square", b"S"),
    "bpg_8qam": ((.446777, .580078, .00192261, 0.), "diamond", b"f*"),
    "bpg_16qam": ((.241211, .418701, .980469, .0333862), "triangle", b"f*"),
    "bpg_ldpc_qpsk": ((.685547, 0., .309082, 0.), "square", b"S"),
    "bpg_ldpc_8qam": ((.458984, .675781, .542969, .266602), "diamond", b"f*"),
    "bpg_ldpc_16qam": ((.471924, .334961, .9375, .100098), "triangle", b"f*"),
}
GRAY = ((.695313, .630859, .607422, .541016), "square", b"S")
RED = ((.209229, .966797, .892578, .116211), "circle", b"f*")
BLUE = ((.943359, .755859, .118652, .0151672), "triangle", b"f*")
GREEN = ((.773438, .339111, .798828, .20752), "triangle", b"f*")
PURPLE = ((.697266, .845703, 0., .000244141), "diamond", b"f*")
OCHRE = ((.416016, .511719, .902344, .236816), "triangle", b"f*")
FIG7 = {
    "fso_sc_dust": GRAY,
    "fso_sc_rain": RED,
    "fso_sc_snow": BLUE,
    "fso_sc_fog": GREEN,
    "bpg_ldpc_dust": PURPLE,
    "bpg_ldpc_rain": OCHRE,
    "bpg_ldpc_snow": ((.796875, .294189, .461914, .0520325), "triangle", b"f*"),
    "bpg_ldpc_fog": ((.529297, .708984, .609375, .507813), "hexagon", b"f*"),
}
SHAPES = {
    "square": (b"m", b"l", b"l", b"l", b"l", b"h"),
    "diamond": (b"m", b"l", b"l", b"l", b"l"),
    "triangle": (b"m", b"l", b"l", b"l", b"h"),
    "circle": (b"m", b"c", b"c", b"c", b"c"),
    "hexagon": (b"m", b"l", b"l", b"l", b"l", b"l", b"l"),
}


def _marker_centers(reader, page_one_based, xobject, rules):
    shapes = {(color, SHAPES[shape], op): (key, shape)
              for key, (color, shape, op) in rules.items()}
    content = reader.pages[page_one_based-1]["/Resources"]["/XObject"][xobject].get_object()
    path = []
    points = defaultdict(list)
    fill = stroke = None
    color_stack = []
    for coords, op in ContentStream(content, reader).operations:
        if op == b"q":
            color_stack.append((fill, stroke))
        elif op == b"Q":
            fill, stroke = color_stack.pop()
        elif op in (b"k", b"rg", b"g"):
            fill = tuple(float(x) for x in coords)
        elif op in (b"K", b"RG", b"G"):
            stroke = tuple(float(x) for x in coords)
        elif op in (b"m", b"l", b"c", b"h", b"re", b"v", b"y"):
            path.append((op, tuple(float(x) for x in coords)))
        elif op in (b"f", b"f*", b"S", b"s", b"n", b"B", b"B*", b"b", b"b*"):
            match = shapes.get((stroke if op == b"S" else fill, tuple(tag for tag, _ in path), op))
            if match:
                key, shape = match
                raw = [p for _, v in path for p in zip(v[::2], v[1::2])]
                xs, ys = zip(*raw)
                if shape == "triangle":
                    # The triangle centroid agrees with the associated line-node
                    # position. A bounding-box midpoint would shift LPIPS.
                    xs, ys = zip(*raw[:3])
                    cx, cy = mean(xs), mean(ys)
                else:
                    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
                points[key].append((cx, cy))
            path = []
    return points


def _panel(centers, *, x0, x_last, y0, y1, axis, x_values, **conditions):
    """Map nominal x labels and LPIPS ticks; reject missing/duplicate points."""
    step = (x_last-x0)/(len(x_values)-1)
    output = {"conditions": conditions, "series": {}}
    for name, series in centers.items():
        values = []
        for index, value in enumerate(x_values):
            x_tick = x0+index*step
            candidates = [(x, y) for x, y in series if abs(x-x_tick) < 9 and y0-3 < y < y1+3]
            if len(candidates) != 1:
                raise ValueError((name, conditions, value, x_tick, candidates))
            _, y = candidates[0]
            lpips = round(min(1., max(0., (y-y0)/(y1-y0))), 3)
            values.append({axis: value, "lpips": lpips})
        output["series"][name] = values
    return output


def digitize(pdf):
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != PAPER_SHA256:
        raise ValueError("PDF differs from the edition whose drawing coordinates were checked")
    reader = PdfReader(pdf)
    # Each color paints the nine data markers for each panel contiguously;
    # the remaining marker is a legend sample. The x/y axes below are the
    # printed major-tick coordinates in the raw Form stream (not pixels).
    c4 = _marker_centers(reader, 7, "/Im6", FIG4)
    c7 = _marker_centers(reader, 10, "/Im10", FIG7)
    c8 = _marker_centers(reader, 11, "/Im11", {
        "fso_sc": GRAY, "baseline": RED, "2rx_egc": BLUE,
        "4rx_egc": GREEN, "2rx_mrc": PURPLE, "4rx_mrc": OCHRE,
    })
    if not all(len(v) == 28 for v in c4.values()) or set(c4) != set(FIG4):
        raise ValueError("Unexpected Fig. 4 marker counts")
    if not all(len(v) == 10 for v in c7.values()) or set(c7) != set(FIG7):
        raise ValueError("Unexpected Fig. 7 marker counts")
    if not all(len(v) == 40 for v in c8.values()) or len(c8) != 6:
        raise ValueError("Unexpected Fig. 8 marker counts")
    result = {
        "metadata": {
            "source": "Chen et al., IEEE Transactions on Communications, 73(10), 2025",
            "doi": DOI,
            "pdf_sha256": PAPER_SHA256,
            "method": "Checked PDF Form vector marker shape/color against legends; map marker centers to printed axes, round to 3 decimals.",
            "scope": "Figure 4(a-c), printed page 9174; Figure 7, p. 9177; Figure 8(a-d), p. 9178.",
            "limitations": [
                "Approximate plotted LPIPS, not raw author measurements or uncertainty estimates.",
                "Matching evaluation images and LPIPS backbone could not be verified.",
                "Paper convention: LPIPS 1 may be a failed BPG reconstruction.",
                "Fig. 8(d) follows the plotted legend and body (with LDPC); its caption contradicts them.",
            ],
        },
        "figure4": {},
        "figure8": {},
    }
    for letter, power, x0, x_last, y0, y1 in (
        ("a", 30, 327.035, 1525.47, 496.059, 1592.77),
        ("b", 33, 2048.23, 3246.67, 498.238, 1594.93),
        ("c", 36, 3776.94, 4975.39, 496.059, 1592.77),
    ):
        result["figure4"][letter] = _panel(
            c4, x0=x0, x_last=x_last, y0=y0, y1=y1, axis="zenith_deg",
            x_values=range(0, 81, 10), power_dbm=power, pointing_urad=20)
    result["figure7"] = _panel(
        {key: vals[:9] for key, vals in c7.items()},
        x0=310.69, x_last=2192.20, y0=267.199, y1=1889.43, axis="power_dbm",
        x_values=range(28, 37), modulation="QPSK", zenith_deg=0, pointing_urad=15)
    for letter, ratio, with_ldpc, x0, x_last, y0, y1 in (
        ("a", .95, False, 376.77, 2285.20, 2363.34, 4011.33),
        ("b", .95, True, 2843.94, 4752.39, 2364.83, 4012.82),
        ("c", .75, True, 376.441, 2284.87, 279.168, 1927.17),
        ("d", .596, True, 2851.11, 4759.55, 271.449, 1919.44),
    ):
        baseline = "bpg_ldpc" if with_ldpc else "bpg"
        result["figure8"][letter] = _panel(
            {("fso_sc_1rx" if name == "fso_sc" else baseline+"_1rx" if name == "baseline"
              else baseline+"_"+name): vals for name, vals in c8.items()},
            x0=x0, x_last=x_last, y0=y0, y1=y1, axis="power_dbm",
            x_values=range(28, 37), modulation="16QAM", zenith_deg=0, pointing_urad=19,
            scattered_power_ratio=ratio, bpg_ldpc=with_ldpc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = json.dumps(digitize(args.pdf), indent=2, allow_nan=False)
    # One point per line; preserving explicit axis values makes review easier.
    import re
    output = re.sub(r'\{\n +("(?:power_dbm|zenith_deg)": [0-9]+),\n +("lpips": [0-9.]+)\n +\}',
                    r'{\1, \2}', output)
    args.output.write_text(output+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
