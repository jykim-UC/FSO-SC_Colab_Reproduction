"""Synthetic source-only checks for the optional 16QAM receiver."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from fso_sc_signal_dsp import SignalDSPConfig
from fso_sc_signal_dsp_v3 import recover_qam_carrier_v3
from fso_sc_signal_dsp_v34 import recover_qam_carrier_v34


class ReceiverVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        gen = torch.Generator().manual_seed(3417)
        axis = torch.tensor([-3., -1., 1., 3.])/10**.5
        x, y = axis[torch.randint(4, (2, 3584), generator=gen)]
        cls.symbols = torch.complex(x, y).unsqueeze(0)
        cls.gain = torch.full((1, 1), .002)
        cls.t = torch.linspace(0, 1, 3584)

    def test_new_phase_estimation_on_fixed_synthetic_frame(self):
        incoming = self.symbols * self.gain * torch.exp(1j*(.22+.17*self.t))
        config = replace(SignalDSPConfig(), cfo_enabled=False)
        output, diag = recover_qam_carrier_v34(incoming, "16QAM", 28e9, config, self.gain)
        self.assertLess(float((output-self.symbols).abs().max()), .025)
        self.assertFalse(diag["uses_true_phase_or_frequency"])
        self.assertFalse(diag["local_phase_unwrap_used"])

    def test_coarse_frequency_stage_is_unchanged(self):
        time = torch.arange(3584)
        incoming = self.symbols * self.gain * torch.exp(1j*2*torch.pi*(420e6/28e9)*time)
        config = replace(SignalDSPConfig(), cpr_enabled=False)
        current, a = recover_qam_carrier_v3(incoming, "16QAM", 28e9, config, self.gain)
        candidate, b = recover_qam_carrier_v34(incoming, "16QAM", 28e9, config, self.gain)
        self.assertTrue(torch.equal(current, candidate))
        self.assertEqual(a["estimated_cfo_hz"], b["estimated_cfo_hz"])


if __name__ == "__main__":
    unittest.main()
