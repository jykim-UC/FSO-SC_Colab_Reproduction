# Paper Tables I–III: parameter and link-budget checks

[한국어](PAPER_TABLES.ko.md) · [Paper figure results](PAPER_FIGURES.md) · [Implementation assumptions](ASSUMPTIONS.md)

Source: [Chen et al. (2025)](https://doi.org/10.1109/TCOMM.2025.3562356), pp. 9174, 9176, and 9178. Tables I–II are simulation inputs, not performance measurements. Table III is an equivalent-lab optical power budget, not the measured LPIPS in Figs. 10–12.

## Table I: downlink simulation parameters

The v27 grids apply these settings panel by panel rather than running every possible combination. The fixed physical defaults are in [`PaperFSOSystemConfig`](../tools/paper_fso_channel_v6.py); the executed conditions are in the [v27 aggregate](../results/paper_sweep_v27_aggregate.json).

| Quantity | Paper Table I | v27 reproduction |
| --- | --- | --- |
| Launch optical power | 28–36 dBm | Same published grid per panel; wider v31 sweep is separate |
| Wavelength | 1550 nm | 1550 nm |
| Satellite and receiver altitudes | 620 km and 0 km | 620 km and 0 km |
| Zenith angle | 0–80° | Same panel grids |
| Pointing error | 18–22 µrad | Same in Fig. 6(a–c); Fig. 7 separately specifies 15 µrad |
| Málaga LOS mean power Ω and scatter parameter b₀ | 0.5 and 0.25 | 0.5 and 0.25; β and phase are inferred, not reported settings |
| Málaga coupled-scatter fraction ρ | 0.95, 0.75, 0.596 | Same values in Fig. 8 |
| Background light radiance | 10⁻⁴–1 W/(m²·sr·µm) | 10⁻⁴, 10⁻², and 1 in Fig. 6(d–f); same input units, not a verified receiver noise conversion |
| Receiver aperture diameter | 31.8 cm | 31.8 cm |
| Local-oscillator linewidth | 1000 Hz | 1000 Hz |
| Local-oscillator frequency offset | f<sub>DS</sub>/10 (table displays MHz) | 0.1 × Doppler shift, computed and applied in Hz |
| Amplifier gain and noise figure | 30 dB and 4 dB | 30 dB and 4 dB in the coherent simulation profile; not the Table III lab EDFA |
| Symbol rate | 28 GBaud | 28 GBaud; symbol-rate simulation, not a sampled optical waveform |
| Photodiode responsivity | 0.85 A/W | 0.85 A/W; the shot-noise conversion is our modeling choice |
| Modulation | QPSK, 8QAM, 16QAM | Same labels; rectangular Gray 8QAM geometry is our choice |

Other published inputs used by the simulation include a 0.05 m transmitter beam radius and the nominal 20 km atmospheric path (8 km for fog). The paper gives `FOV=0.1` without a unit; the simulator interprets it as radians and assumes a 1 nm optical filter. Matching input values is not evidence of matching the undisclosed VPI receiver.

## Table II: weather attenuation

The implementation uses the published attenuation coefficients directly; visibility and precipitation rates are reference labels, not recomputed weather inputs. Path loss here is **only** the weather term, not the total link loss.

| Weather | Paper visibility / precipitation | Paper and code (dB/km) | Modeled path (km) | Modeled weather loss (dB) |
| --- | ---: | ---: | ---: | ---: |
| Clear | 30 km | 0.147 | 20 | 2.940 |
| Dust | 5 km | 1.043 | 20 | 20.860 |
| Fog | 0.5 km | 33.979 | 8 | 271.832 |
| Rain | 5 mm/h | 1.403 | 20 | 28.060 |
| Snow | 0.5 mm/h | 2.145 | 20 | 42.900 |

## Table III: equivalent-lab power budget

Independent v29 Friis/Gaussian calculation at 620 km, 1550 nm, clear air, 0° zenith, 20 µrad pointing, and a 31.8 cm receiver aperture. Signs below are positive for gain and negative for loss, as in the paper. The totals are derived from the listed terms; the 20 dB lab EDFA is carried over from the paper, **not independently estimated or applied to the 28 GBaud sweeps**.

| Budget term | Paper (dB) | Independent v29 (dB) |
| --- | ---: | ---: |
| Transmitter telescope gain | +109.15 | +109.146663 |
| Pointing loss | −35.68 | −35.679737 |
| Free-space loss | −254.03 | −254.025397 |
| Atmospheric loss | −3.00 | −2.940000 |
| Lab EDFA gain | +20.00 | +20.000000 (input) |
| Receiver telescope gain | +116.18 | +116.184906 |
| Net gain before lab EDFA (derived) | −67.38 | −67.313565 |
| Net gain with lab EDFA (derived) | −47.38 | −47.313565 |

Absolute difference in the pre-EDFA budget: **0.066435 dB**. The atmospheric term explains 0.060 dB of that difference (`0.147 × 20 = 2.94` dB versus the paper’s 3 dB entry). Using the independent full Eqs. (5)–(12) spreading calculation instead gives **67.312209 dB** of pre-EDFA loss. This is an arithmetic cross-check of average propagation, not a validation of stochastic fading, DSP, or the hardware experiment.

Rerun the [independent v29 calculation](../tools/fso_sc_link_budget_propagation_v29.py) from the repository root; no model, Python dependencies, or Drive files are required:

```sh
python -c "from tools.fso_sc_link_budget_propagation_v29 import audit_paper_table_iii_v29 as audit; import json; print(json.dumps(audit(), indent=2))"
```
