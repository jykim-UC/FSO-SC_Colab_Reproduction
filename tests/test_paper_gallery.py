"""Verify the published gallery's point mapping without Matplotlib or private inputs."""

from collections import Counter
import csv
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fso_sc_link_budget_propagation_v29 import audit_paper_table_iii_v29
from plot_paper_gallery import DEFAULT_CHANNEL, case_key


class PaperGalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def read_json(path):
            return json.loads((ROOT / path).read_text(encoding="utf-8"))

        cls.aggregate = read_json("results/paper_sweep_v27_aggregate.json")
        cls.reference = read_json("config/paper_gallery_reference.json")
        cls.raster = read_json("config/paper_gallery_fig6_raster_reference.json")
        cls.ceilings = read_json("config/paper_gallery_fig6d_anchors.json")
        with (ROOT / "results/paper_gallery_points.csv").open(newline="", encoding="utf-8") as handle:
            cls.points = list(csv.DictReader(handle))

    def test_each_gallery_row_matches_a_measured_case(self):
        measured = {
            case_key(row["method"], row["receivers"], row["combining"], row["channel"]): row
            for row in self.aggregate["points"] if row["channel"] is not None
        }
        observed = set()
        seen = set()
        for point in self.points:
            channel = {k: point[k] if isinstance(v, str) else float(point[k]) for k, v in DEFAULT_CHANNEL.items()}
            key = case_key(point["method"], int(point["receivers"]), point["combining"], channel)
            identity = point["figure"], point["panel"], key
            self.assertNotIn(identity, seen)
            seen.add(identity)
            observed.add(key)
            self.assertEqual(float(point["x_value"]), channel[point["x_axis"]])
            row = measured[key]
            self.assertEqual(float(point["v27_lpips_alex"]), row["lpips_alex"])
            self.assertEqual([float(point["v27_ci95_lo"]), float(point["v27_ci95_hi"])], row["lpips_alex_ci95"])
        self.assertEqual(len(self.points), 783)
        self.assertEqual(len(measured), 732)
        self.assertEqual(observed, set(measured))
        self.assertEqual(Counter(p["figure"] for p in self.points),
                         {"fig4": 243, "fig6": 252, "fig7": 72, "fig8": 216})

    def test_vector_marker_readings_have_exactly_one_matching_row(self):
        counts = Counter()
        for point in self.points:
            figure, panel, method = (point[k] for k in ("figure", "panel", "method"))
            if figure == "fig6":
                continue
            if figure == "fig4":
                series = self.reference["figure4"][panel]["series"][f"{method}_{point['modulation'].lower()}"]
                axis = "zenith_deg"
            elif figure == "fig7":
                series = self.reference["figure7"]["series"][f"{method}_{point['weather']}"]
                axis = "power_dbm"
            else:
                branches = f"{point['receivers']}rx"
                if point["receivers"] != "1":
                    branches += f"_{point['combining'].lower()}"
                series = self.reference["figure8"][panel]["series"][f"{method}_{branches}"]
                axis = "power_dbm"
            reference, = [item["lpips"] for item in series if item[axis] == float(point["x_value"])]
            self.assertEqual(float(point["paper_lpips_approx"]), reference)
            self.assertEqual(point["paper_reference_kind"], "vector_marker_approx")
            counts[figure] += 1
        self.assertEqual(counts, {"fig4": 243, "fig7": 72, "fig8": 216})

    def test_raster_readings_leave_other_figure_6_points_unfilled(self):
        matched = Counter()
        for point in (p for p in self.points if p["figure"] == "fig6"):
            panel, method, x_value = point["panel"], point["method"], float(point["x_value"])
            if panel in "abc":
                group = f"z{float(point['zenith_angle_deg']):g}"
                axis = "pointing_urad"
            else:
                group = f"bgl{float(point['background_light_radiance_w_m2_sr_um']):g}"
                axis = "power_dbm"
            reference = self.raster["figure6"][panel]["series"].get(f"{method}_{group}", [])
            raster = [r for r in reference if r[axis] == x_value]
            ceilings = [r for r in self.ceilings["points"] if r[axis] == x_value] if (
                (panel, method, group) == ("d", "bpg_ldpc", "bgl1")) else []
            if raster:
                reference_point, = raster
                self.assertFalse(ceilings)
                self.assertLessEqual(reference_point["marker_ray_residual_px"], 4.5)
                self.assertEqual(point["paper_reference_kind"], "raster_approx")
            elif ceilings:
                reference_point, = ceilings
                self.assertEqual(point["paper_reference_kind"], "raster_ceiling_approx")
            else:
                self.assertEqual(point["paper_lpips_approx"], "")
                self.assertEqual(point["paper_reference_kind"], "")
                continue
            self.assertEqual(float(point["paper_lpips_approx"]), reference_point["lpips"])
            matched[point["paper_reference_kind"]] += 1
        self.assertEqual(matched, {"raster_approx": 66, "raster_ceiling_approx": 3})

    def test_table_iii_is_independent_and_keeps_lab_edfa_separate(self):
        audit = audit_paper_table_iii_v29()
        paper = audit["paper_table_iii"]
        calculated = audit["independent_friis_diffraction_only"]
        self.assertAlmostEqual(paper["total_attenuation_db"], 67.38, places=9)
        self.assertAlmostEqual(calculated["total_attenuation_db"], 67.313565, places=6)
        self.assertAlmostEqual(calculated["net_attenuation_including_lab_edfa_db"],
                               calculated["total_attenuation_db"] - 20.0, places=9)
