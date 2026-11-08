"""Independent scalar propagation check; does not alter the production channel.

Chen et al. (2025), https://doi.org/10.1109/TCOMM.2025.3562356:
Eqs. (5)-(12), Tables I-II and Sections II-B/IV-D. The finite aperture
and altitude-shell sensitivities are not part of the primary paper calculation.
Ground-altitude Gauss-Legendre integration is independent of v6's trapezoids
in fractional slant distance. Table III is a separate laboratory anchor.
"""

import math
from collections.abc import Mapping


PROPAGATION_REFERENCE_V29 = "https://doi.org/10.1109/TCOMM.2025.3562356"
_WEATHER_V29 = {"clear": 0.147, "dust": 1.043, "fog": 33.979, "rain": 1.403, "snow": 2.145}
_GL8_V29 = (
    (0.1834346424956498, 0.3626837833783620),
    (0.5255324099163290, 0.3137066458778873),
    (0.7966664774136267, 0.2223810344533745),
    (0.9602898564975363, 0.1012285362903763),
)


def _finite_v29(value, name, minimum=0.0, strict=False):
    x = float(value)
    if not math.isfinite(x) or x < minimum or (strict and x == minimum):
        raise ValueError(f"Invalid {name}: {value!r}")
    return x


def _gl8_v29(fn, left, right):
    center, half = (left + right) / 2.0, (right - left) / 2.0
    return half * math.fsum(
        weight * (fn(center - half * node) + fn(center + half * node))
        for node, weight in _GL8_V29
    )


def _distance_and_derivative_v29(r0, dh, cos_z):
    """Ray/spherical shell intersection; rationalized positive root."""
    radius_squared_difference = dh * (2 * r0 + dh)
    projection = r0 * cos_z
    root = math.sqrt(radius_squared_difference + projection**2)
    return radius_squared_difference / (root + projection), (r0 + dh) / root


def _altitude_moment_v29(r0, ground, height, cos_z, slant, wind, ground_cn2):
    # Eq. (7) under x=altitude-ground: dxi/dx=(r0+x)/(L sqrt(...)).
    def integrand(dh):
        h = ground + dh
        l, dl_dh = _distance_and_derivative_v29(r0, dh, cos_z)
        cn2 = (
            0.00594 * (wind / 27.0)**2 * (1e-5 * h)**10 * math.exp(-h / 1000.0)
            + 2.7e-16 * math.exp(-h / 1500.0)
            + ground_cn2 * math.exp(-h / 100.0)
        )
        return cn2 * (l / slant)**(5.0 / 3.0) * dl_dh / slant

    points = [0.0]
    points += [x for x in (1., 10., 50., 100., 250., 500., 1000., 2000., 4000.,
                            8000., 16000., 32000., 64000., 128000.) if x < height]
    points.append(height)
    panels = tuple(zip(points, points[1:]))
    coarse = math.fsum(_gl8_v29(integrand, a, b) for a, b in panels)
    for subdivisions in (2, 4, 8, 16, 32, 64):
        fine = math.fsum(
            _gl8_v29(integrand, a + j * (b - a) / subdivisions, a + (j + 1) * (b - a) / subdivisions)
            for a, b in panels for j in range(subdivisions)
        )
        residual = abs(fine - coarse) / fine if fine else 0.0
        if residual <= 1e-7:
            return fine, residual
        coarse = fine
    raise ArithmeticError("Altitude integration did not pass internal refinement check")


def _aperture_loss_v29(diameter, waist, offset):
    """Numerical disk integral for shifted normalized Gaussian irradiance.

    The y-chord integral is analytic (erf); x=a sin(t) smooths the endpoints.
    This tests Eq. (5)+(12)'s small-aperture approximation, not a channel change.
    See also Farid & Hranilovic: https://doi.org/10.1109/JLT.2007.899174.
    """
    radius = diameter / 2.0
    scale = math.sqrt(2.0) / waist
    min_displacement = max(offset - radius, 0.0)

    def integrand(t):
        dx = abs(radius * math.sin(t) - offset)
        chord_half = radius * math.cos(t)
        log_relative = -scale**2 * max(0.0, (dx - min_displacement) * (dx + min_displacement))
        return (scale / math.sqrt(math.pi) * chord_half
                * math.exp(log_relative) * math.erf(scale * chord_half))

    step = math.pi / 32.0
    relative = math.fsum(
        _gl8_v29(integrand, -math.pi / 2.0 + j * step, -math.pi / 2.0 + (j + 1) * step)
        for j in range(32)
    )
    if relative <= 0.0:
        return None
    return -10.0 / math.log(10.0) * (math.log(relative) - (scale * min_displacement)**2)


def audit_propagation_v29(config, condition, legacy_budget=None):
    """Evaluate paper propagation with fixed published weather coefficients.

    config is a PaperFSOSystemConfig-like object or dict. condition includes
    launch_power_dbm, zenith_angle_deg, pointing_error_urad and weather.
    legacy_budget, when given, is the existing v6 budget for that exact condition.
    Inputs/outputs are powers; the matching field amplitude is sqrt(transmission).
    No receiver noise, gain or random fading is folded into received_signal_w.
    """
    get = config.__getitem__ if isinstance(config, Mapping) else lambda key: getattr(config, key)
    earth, ground, satellite = (_finite_v29(get(name), name) for name in
                                 ("earth_radius_m", "receiver_altitude_m", "satellite_altitude_m"))
    if earth <= 0 or satellite <= ground:
        raise ValueError("Need a positive Earth radius and satellite above receiver")
    wavelength, beam, diameter = (_finite_v29(get(name), name, strict=True) for name in
                                   ("wavelength_m", "beam_radius_m", "receiver_aperture_m"))
    wind, ground_cn2 = (_finite_v29(get(name), name) for name in
                         ("rms_wind_speed_m_s", "ground_cn2_m_neg_two_thirds"))
    angle = _finite_v29(condition["zenith_angle_deg"], "zenith angle")
    pointing = _finite_v29(condition["pointing_error_urad"], "pointing error")
    if angle >= 90.0:
        raise ValueError("Zenith angle must be below 90 degrees")
    launch_dbm = float(condition["launch_power_dbm"])
    if not math.isfinite(launch_dbm):
        raise ValueError("Launch power must be finite")
    weather = condition["weather"]
    if weather not in _WEATHER_V29:
        raise ValueError(f"No Table II coefficient for weather {weather!r}")
    path_key = "fog_range_m" if weather == "fog" else "atmospheric_range_m"
    weather_path = _finite_v29(get(path_key), path_key)
    cos_z = math.cos(math.radians(angle))
    r0, height = earth + ground, satellite - ground
    slant, _ = _distance_and_derivative_v29(r0, height, cos_z)

    # Collimated Gaussian equivalents of Eqs. (7), (10) and (11).
    divergence = wavelength / (math.pi * beam)
    diffraction = math.hypot(beam, slant * divergence)
    theta1 = (beam / diffraction)**2
    lambda1 = slant * wavelength / (math.pi * diffraction**2)
    moment, residual = _altitude_moment_v29(r0, ground, height, cos_z, slant, wind, ground_cn2)
    spread = (4.35 * moment * lambda1**(5.0 / 6.0)
              * (2.0 * math.pi / wavelength)**(7.0 / 6.0) * slant**(11.0 / 6.0))
    long_term = diffraction * math.sqrt(1.0 + spread)
    offset = slant * pointing * 1e-6

    def losses(waist):
        geo = 10 * math.log10(2.0 * (waist / diameter)**2)
        pt = (20.0 / math.log(10.0)) * (offset / waist)**2
        return geo, pt

    geometric, pointing_db = losses(long_term)
    atmosphere = _WEATHER_V29[weather] * weather_path / 1000.0
    total = math.fsum((geometric, pointing_db, atmosphere))
    launch_w = 10**((launch_dbm - 30.0) / 10.0)
    transmission = 10**(-total / 10.0)
    independent = {
        "slant_range_m": slant, "theta1": theta1, "lambda1": lambda1,
        "diffraction_radius_m": diffraction, "long_term_radius_m": long_term,
        "hv_mu_2d_m_neg_two_thirds": moment, "hv_quadrature_refinement_relative": residual,
        "long_term_spreading_factor": spread, "pointing_offset_m": offset,
        "geometric_scintillation_db": geometric, "pointing_db": pointing_db,
        "weather_coefficient_db_per_km": _WEATHER_V29[weather],
        "weather_path_m": weather_path, "atmospheric_db": atmosphere,
        "total_attenuation_db": total, "launch_power_w": launch_w,
        "power_transmission": transmission, "field_transmission": math.sqrt(transmission),
        "received_signal_w": launch_w * transmission, "received_signal_dbm": launch_dbm - total,
    }
    shell_length = (_distance_and_derivative_v29(r0, min(height, weather_path), cos_z)[0]
                    if weather_path else 0.0)
    exact_combined_db = _aperture_loss_v29(diameter, long_term, offset)
    no_spread_geo, no_spread_pointing = losses(diffraction)
    nominal_atmospheric_path = _finite_v29(get("atmospheric_range_m"), "atmospheric_range_m")
    fixed_atmosphere = {label: _WEATHER_V29[label] * nominal_atmospheric_path / 1000.0
                        for label in ("clear", "snow")}
    sensitivities = {
        "paper_diffraction_only_approximation": {
            "total_attenuation_db": no_spread_geo + no_spread_pointing + atmosphere,
            "delta_db": (no_spread_geo + no_spread_pointing + atmosphere) - total,
        },
        "exact_disk_aperture_same_beam": {
            "geometry_plus_pointing_db": exact_combined_db,
            "delta_db": (None if exact_combined_db is None
                         else exact_combined_db - (geometric + pointing_db)),
        },
        "weather_layer_as_vertical_shell_not_primary": {
            "weather_slant_path_m": shell_length,
            "atmospheric_db": _WEATHER_V29[weather] * shell_length / 1000.0,
            "delta_db": _WEATHER_V29[weather] * (shell_length - weather_path) / 1000.0,
        },
        "weather_at_fixed_geometry": {
            label: {
                "atmospheric_db": atm,
                "total_attenuation_db": geometric + pointing_db + atm,
                "received_signal_dbm": launch_dbm - (geometric + pointing_db + atm),
            } for label, atm in fixed_atmosphere.items()
        },
    }
    sensitivities["weather_at_fixed_geometry"]["snow_minus_clear_db"] = (
        fixed_atmosphere["snow"] - fixed_atmosphere["clear"]
    )
    common = ("slant_range_m", "theta1", "lambda1", "diffraction_radius_m",
              "long_term_radius_m", "geometric_scintillation_db", "pointing_db",
              "atmospheric_db", "total_attenuation_db")
    legacy = ({key: _finite_v29(legacy_budget[key], key) for key in common}
              if legacy_budget is not None else None)
    if legacy is not None:
        for key, expected in (("launch_power_dbm", launch_dbm), ("zenith_angle_deg", angle),
                              ("pointing_error_urad", pointing), ("weather", weather)):
            if key in legacy_budget and legacy_budget[key] != expected:
                raise ValueError(f"Legacy budget condition mismatch: {key}")
        legacy["power_transmission"] = 10**(-legacy["total_attenuation_db"] / 10.0)
        legacy["received_signal_w"] = launch_w * legacy["power_transmission"]
        legacy["received_signal_dbm"] = launch_dbm - legacy["total_attenuation_db"]

    return {
        "paper_reference": PROPAGATION_REFERENCE_V29,
        "conventions": {
            "beam": "Collimated Gaussian; W0 and WLT are radii, DR is aperture diameter.",
            "attenuation": "Power dB; fixed slant atmospheric path 20 km (8 km fog), not secant-scaled.",
            "reference_plane": "Launched signal in free space to collected optical signal before any receiver gain/noise.",
            "scope": "Deterministic mean-path ledger only. Does not validate stochastic Malaga or recover author simulator.",
            "integration": "Ground-altitude change of variable, subdivided 8-point Gauss-Legendre quadrature.",
        },
        "independent": independent, "legacy": legacy, "sensitivities": sensitivities,
        "differences": ({key: independent[key] - legacy[key] for key in legacy} if legacy else None),
    }


def audit_paper_table_iii_v29(config=None):
    """Compare independently computed lab attenuation against Table III.

    Uses the published lab geometry, regardless of simulation config. The
    20 dB lab EDFA is reported separately; never applied to Fig. 4/7.
    """
    # Frozen paper values for this published comparator, not experimental knobs.
    paper = dict(earth_radius_m=6371e3, receiver_altitude_m=0.0, satellite_altitude_m=620e3,
                 wavelength_m=1550e-9, beam_radius_m=0.05, receiver_aperture_m=0.318,
                 rms_wind_speed_m_s=21., ground_cn2_m_neg_two_thirds=1.7e-14,
                 atmospheric_range_m=20e3, fog_range_m=8e3)
    condition = dict(launch_power_dbm=0.0, zenith_angle_deg=0.0, pointing_error_urad=20.0, weather="clear")
    audit = audit_propagation_v29(paper, condition)
    primary = audit["independent"]
    published = {
        "transmitter_telescope_gain_db": 109.15, "receiver_telescope_gain_db": 116.18,
        "free_space_loss_db": 254.03, "pointing_db": 35.68,
        "atmospheric_db": 3.0, "lab_edfa_gain_db": 20.0,
    }
    published["geometric_scintillation_db"] = (
        published["free_space_loss_db"] - published["transmitter_telescope_gain_db"]
        - published["receiver_telescope_gain_db"]
    )
    published["total_attenuation_db"] = sum(
        published[key] for key in ("geometric_scintillation_db", "pointing_db", "atmospheric_db")
    )
    published["net_attenuation_including_lab_edfa_db"] = (
        published["total_attenuation_db"] - published["lab_edfa_gain_db"]
    )

    # Independent Friis gains: ideal aperture and far-field Gaussian on-axis
    # directivity; this recovers the *geometry* ledger by another route.
    wavelength, w0, diameter = (paper[key] for key in ("wavelength_m", "beam_radius_m", "receiver_aperture_m"))
    fspl = 20.0 * math.log10(4.0 * math.pi * primary["slant_range_m"] / wavelength)
    tx_gain = 10.0 * math.log10(8.0 * (math.pi * w0 / wavelength)**2)
    rx_gain = 20.0 * math.log10(math.pi * diameter / wavelength)
    geo_friis = fspl - tx_gain - rx_gain
    diffraction_total = audit["sensitivities"]["paper_diffraction_only_approximation"]["total_attenuation_db"]
    diffraction_pointing = diffraction_total - primary["atmospheric_db"] - 10.0 * math.log10(
        2.0 * (primary["diffraction_radius_m"] / diameter)**2
    )
    friis_total = geo_friis + diffraction_pointing + primary["atmospheric_db"]
    calculated = {
        "transmitter_telescope_gain_db": tx_gain, "receiver_telescope_gain_db": rx_gain,
        "free_space_loss_db": fspl, "geometric_scintillation_db": geo_friis,
        "pointing_db": diffraction_pointing, "atmospheric_db": primary["atmospheric_db"],
        "total_attenuation_db": friis_total, "lab_edfa_gain_db": 20.0,
        "net_attenuation_including_lab_edfa_db": friis_total - 20.0,
    }
    return {
        "paper_reference": PROPAGATION_REFERENCE_V29,
        "scope": "Table III clear 620 km equivalent experiment only, not 28 GBd Fig. 4/7 simulation.",
        "paper_table_iii": published, "independent_friis_diffraction_only": calculated,
        "independent_full_eq5_to12": {key: primary[key] for key in (
            "geometric_scintillation_db", "pointing_db", "atmospheric_db", "total_attenuation_db")},
        "friis_minus_paper": {key: calculated[key] - published[key] for key in calculated},
    }
