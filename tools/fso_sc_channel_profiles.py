"""Explicit noise interpretations; embeddable after the legacy channel namespace.

No training/model changes and no default profile. These are sensitivity profiles,
not settings selected to match the published LPIPS curve or a full VPI replica.
"""

import math


_PROFILE_FLAGS = {
    "paper_eq3_4_background_only": (True, False, False),
    "coherent_receiver_extension_v6": (True, True, True),
    "background_plus_ase_ablation": (True, True, False),
}
PROFILE_NAMES = tuple(_PROFILE_FLAGS)
_NOISE_FLAGS = (
    "include_background_noise", "include_ase_noise", "include_shot_noise"
)


def _profile_flags(name):
    if name not in _PROFILE_FLAGS:
        raise ValueError(f"Unknown explicit channel profile {name!r}; choose {PROFILE_NAMES}")
    return dict(zip(_NOISE_FLAGS, _PROFILE_FLAGS[name]))


def make_channel_profile(name, config_class, **overrides):
    """Construct the caller's config class; noise flags cannot contradict its label."""
    flags = _profile_flags(name)
    protected = set(overrides).intersection(_NOISE_FLAGS)
    if protected:
        raise ValueError(f"Profile noise flags cannot be overridden: {sorted(protected)}")
    return config_class(**flags, **overrides)


def profile_metadata(name):
    """Return fresh JSON-safe provenance, including what this profile cannot claim."""
    flags = _profile_flags(name)
    basis = {
        "paper_eq3_4_background_only": (
            "Chen et al. Eq. (3) defines n(t) as background-light noise; Eq. (4) "
            "gives its collected optical power. No receiver shot or ASE term is "
            "printed in those equations."
        ),
        "coherent_receiver_extension_v6": (
            "Unchanged v6 interpretation: background plus propagated transmitter "
            "EDFA ASE and input-referred single-polarization heterodyne shot noise."
        ),
        "background_plus_ase_ablation": (
            "Diagnostic ablation only: retain v6 background and transmitter ASE "
            "but omit its receiver shot term. Not an independently published model."
        ),
    }
    return {
        "profile": name,
        "noise_flags": flags,
        "basis": basis[name],
        "diagnostic_only": name == "background_plus_ase_ablation",
        "full_vpi_reproduction": False,
        "physical_superiority_claim": False,
        "curve_fitted": False,
        "limitations": [
            "Eq. (4) optical background power to complex matched-filter variance "
            "is an implementation approximation, not a disclosed receiver transfer.",
            "FOV unit, optical filter width, Malaga phase/beta, fading coherence "
            "and proprietary coherent receiver/DSP details remain assumptions.",
            "Background-only omission of quantum shot noise is an equation-level "
            "idealization, not a physically complete coherent receiver.",
            "ASE bandwidth/post-EDFA launch reference and shot bandwidth/polarization "
            "are inferred; neither establishes the undisclosed VPI configuration.",
        ],
        "provenance": {
            "paper": "https://ieeexplore.ieee.org/document/10969992/",
            "published_channel": "Chen et al., Sec. II-B, Eqs. (3)-(4); Table I",
            "v6_shot_reference": "https://doi.org/10.1103/PhysRevA.77.032116",
            "selection_rule": "Choose explicitly before examining evaluation outcomes.",
        },
    }


def _positive_power_dbm(power_w):
    return 10.0 * math.log10(power_w) + 30.0 if 0.0 < power_w < math.inf else None


def summarize_channel_budget(config, condition, budget_function):
    """Scalar no-fade, single-branch Es/N0; excludes fading and DSP errors.

    Powers are in W; the active noise terms denote E|n|^2 at the receiver.
    ASE is propagated through the same power attenuation as the signal. Raw
    formula outputs remain visible even for disabled noise terms. No Torch is
    imported; pass the existing compute_paper_link_budget function explicitly.
    """
    flags = {key: bool(getattr(config, key)) for key in _NOISE_FLAGS}
    matches = [name for name in PROFILE_NAMES if _profile_flags(name) == flags]
    if len(matches) != 1:
        raise ValueError("Config noise flags do not match an explicit channel profile")
    budget = budget_function(config, **dict(condition))
    attenuation = float(budget["total_attenuation_db"])
    launch = float(budget["launch_power_w"])
    if not math.isfinite(attenuation) or not math.isfinite(launch) or launch < 0.0:
        raise ValueError("Attenuation and nonnegative launch power must be finite")
    transmission = 10.0 ** (-attenuation / 10.0)
    signal = launch * transmission
    computed = {
        "background": float(budget["background_noise_w"]),
        "transmitter_ase": float(budget["ase_noise_w"]),
        "receiver_shot": float(budget["receiver_shot_noise_w"]),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in computed.values()):
        raise ValueError("Computed noise powers must be finite and nonnegative")
    active = {
        "background": computed["background"] if flags[_NOISE_FLAGS[0]] else 0.0,
        "transmitter_ase": computed["transmitter_ase"] * transmission
        if flags[_NOISE_FLAGS[1]] else 0.0,
        "receiver_shot": computed["receiver_shot"] if flags[_NOISE_FLAGS[2]] else 0.0,
    }
    active["total"] = sum(active.values())
    snr = signal / active["total"] if active["total"] > 0.0 else None
    snr = snr if snr is None or math.isfinite(snr) else None
    snr_db = 10.0 * math.log10(snr) if snr is not None and snr > 0.0 else None
    return {
        "profile": matches[0],
        "condition": dict(condition),
        "noise_flags": flags,
        "total_attenuation_db": attenuation,
        "weather_attenuation_db": float(budget["atmospheric_db"]),
        "signal_power_no_fading_w": signal,
        "signal_power_no_fading_dbm": _positive_power_dbm(signal),
        "computed_noise_variance_w": computed,
        "computed_noise_power_dbm": {
            key: _positive_power_dbm(value) for key, value in computed.items()
        },
        "active_received_noise_variance_w": active,
        "active_received_noise_power_dbm": {
            key: _positive_power_dbm(value) for key, value in active.items()
        },
        "no_fading_esn0_linear": snr,
        "no_fading_esn0_db": snr_db,
        "snr_status": "finite" if snr_db is not None else "zero_signal_or_noise",
        "snr_scope": "single branch; unit-energy symbols; no fading or DSP errors",
    }


def qpsk_awgn_ber_from_esn0_db(esn0_db):
    """Gray QPSK coherent AWGN BER: Q(sqrt(Es/N0)); not a fading/DSP prediction."""
    esn0_db = float(esn0_db)
    if not math.isfinite(esn0_db):
        raise ValueError("Es/N0 must be finite dB")
    if esn0_db > 300.0:
        return 0.0
    return 0.5 * math.erfc(math.sqrt(10.0 ** (esn0_db / 10.0) / 2.0))
