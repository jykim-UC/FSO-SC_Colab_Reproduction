"""Check the public v35 crossover and report-only correction without private frames."""

import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Fig8bReleaseTests(unittest.TestCase):
    def test_frozen_scope_and_mean_crossover(self):
        with (ROOT/'results/fig8b_lowpower_v35_aggregate.json').open(encoding='utf-8') as handle:
            result = json.load(handle)
        self.assertEqual((result['images'], result['physical_image_method_evaluations'],
                          result['clean_image_method_evaluations'], result['training_updates_during_evaluation']),
                         (32, 768, 64, 0))
        self.assertEqual(result['published_overlap_dbm'], [28])
        self.assertEqual(result['power_grid_dbm'], list(range(18, 29, 2)))
        self.assertEqual(len(result['points']), 24)
        paired = result['paired_points']
        self.assertEqual([r['one_rx_failed_four_rx_decoded_images'] for r in paired], [31, 27, 10, 0, 0, 0])
        self.assertEqual([r['four_rx_failed_one_rx_decoded_images'] for r in paired], [0]*6)
        self.assertFalse(result['report_correction']['evaluation_rerun'])
        self.assertFalse(result['report_correction']['lpips_ber_failure_rates_ci_csv_and_plots_changed'])
        delta = [r['four_rx_minus_fso']['lpips_alex'] for r in paired]
        self.assertLess(delta[0]['ci95'][1], 0)
        self.assertLess(delta[1]['ci95'][0], 0)
        self.assertGreater(delta[1]['ci95'][1], 0)
        self.assertGreater(delta[2]['ci95'][0], 0)
        self.assertEqual(result['descriptive_contrasts']['four_rx_minus_fso']['lpips_alex']
                         ['observed_sign_change_brackets_dbm'], [[20, 22]])
        for name, sha in result['aggregate_plot_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/'results/figures'/name).read_bytes()).hexdigest(), sha)
        name, = result['aggregate_csv_sha256']
        with (ROOT/'results'/name).open(encoding='utf-8', newline='') as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 24)


if __name__ == '__main__':
    unittest.main()
