"""Paper-parameter symbol-rate FSO channel with explicit heterodyne shot floor.

The vacuum/shot term is a physically motivated missing receiver detail, not an
explicit Chen et al. parameter or an LPIPS-curve calibration. One complex
heterodyne mode/symbol has E|n|^2=h*nu/eta, so the optical-field (sqrt W)
convention gives N_complex=(q_e/R)*R_s W; I and Q each have half that variance.
Reference: A. Martinez, Physical Review A 77,032116 (2008),
doi:10.1103/PhysRevA.77.032116, https://ir.cwi.nl/pub/13524/13524D.pdf .
"""

from dataclasses import asdict, dataclass, replace
import math
import warnings

import numpy as np
import torch


PAPER_FSO_CHANNEL_VERSION = "chen2025_eq3_18_heterodyne_shot_v6"
WEATHER_ATTENUATION_DB_PER_KM = {
    "clear": 0.147,
    "dust": 1.043,
    "fog": 33.979,
    "rain": 1.403,
    "snow": 2.145,
}

# The paper specifies FOV=0.1 but does not print a unit, and it does not give
# the optical filter width or the proprietary VPI receiver/DSP implementation.
# These assumptions are explicit and stored with the final model metadata.
PAPER_UNDISCLOSED_RECEIVER_ASSUMPTIONS = {
    "fov_interpretation": "0.1 rad",
    "optical_filter_bandwidth": "1 nm",
    "background_noise_conversion": (
        "Eq.-(4) optical background power is used as unit-energy complex "
        "matched-filter noise variance; proprietary coherent receiver is undisclosed"
    ),
    "ase_model": (
        "Table-I launch power is post-transmitter-EDFA; standard two-polarization "
        "EDFA ASE is propagated through the optical path"
    ),
    "ase_noise_bandwidth": "28 GHz symbol-rate equivalent",
    "diversity_ase_correlation": "one common transmitter-ASE waveform across Rx branches",
    "photodiode_responsivity": (
        "Table-I R=0.85 A/W determines eta=R*h*nu/q_e; "
        "one-complex-mode/symbol heterodyne shot floor q_e*Rs/R is input-referred; "
        "strong LO and negligible receiver thermal/RIN/dark current are assumptions"
    ),
    "receiver_shot_noise": (
        "complex heterodyne mode variance h*nu/eta per symbol; "
        "N_complex=q_e*Rs/R W, with half per real quadrature; "
        "physically motivated undisclosed implementation detail, not curve-fitted"
    ),
    "receiver_shot_noise_reference": "https://doi.org/10.1103/PhysRevA.77.032116",
    "pulse_shaping": "symbol-rate matched-filter equivalent; RRC rolloff undisclosed",
    "carrier_recovery": (
        "raw channel returns true complex gain for an explicitly ideal/oracle baseline; "
        "the separate SignalDSP wrapper uses only block magnitude and estimates phase/frequency"
    ),
    "rytov_quadrature": "xi=u^2 ground-clustered trapezoid with convergence self-test",
    "malaga_phase_difference": "pi/2 chosen phase assumption; paper undisclosed",
    "malaga_beta_fit": (
        "paper omits beta and phase; nearest Eq.-(15) natural beta is used with "
        "an explicit feasibility/residual diagnostic (strict policy is available)"
    ),
    "fading_coherence": "one independent block fade per image and receiver",
    "8qam_geometry": "rectangular Gray 8-QAM; paper only states 8-QAM",
}


def paper_generator_device_matches(generator, tensor_device):
    generator_device = generator.device
    if generator_device.type != tensor_device.type:
        return False
    if tensor_device.type != "cuda":
        return generator_device == tensor_device
    generator_index = (
        torch.cuda.current_device()
        if generator_device.index is None
        else generator_device.index
    )
    tensor_index = (
        torch.cuda.current_device() if tensor_device.index is None else tensor_device.index
    )
    return generator_index == tensor_index


@dataclass(frozen=True)
class PaperFSOSystemConfig:
    # Paper Table I / Eqs. (4)-(18).
    wavelength_m: float = 1550e-9
    satellite_altitude_m: float = 620e3
    receiver_altitude_m: float = 0.0
    earth_radius_m: float = 6371e3
    earth_gravitational_parameter: float = 3.986004418e14
    light_speed_m_s: float = 299792458.0
    planck_constant_j_s: float = 6.62607015e-34
    electron_charge_c: float = 1.602176634e-19
    beam_radius_m: float = 0.05
    receiver_aperture_m: float = 0.318
    rms_wind_speed_m_s: float = 21.0
    ground_cn2_m_neg_two_thirds: float = 1.7e-14
    atmospheric_range_m: float = 20e3
    fog_range_m: float = 8e3
    field_of_view_rad: float = 0.1
    optical_filter_bandwidth_um: float = 1e-3  # 1 nm; not disclosed.
    symbol_rate_baud: float = 28e9
    amplifier_gain_db: float = 30.0
    amplifier_noise_figure_db: float = 4.0
    lo_linewidth_hz: float = 1000.0
    photodiode_responsivity_a_w: float = 0.85
    doppler_residual_ratio: float = 0.1
    malaga_omega: float = 0.5
    malaga_b0: float = 0.25
    malaga_phase_difference_rad: float = math.pi / 2
    malaga_beta: object = None
    max_derived_beta: int = 1000
    beta_fit_policy: str = "nearest_warn"
    beta_fit_relative_tolerance: float = 1e-3
    integration_points: int = 4097
    integration_grid_power: float = 2.0
    num_receivers: int = 1
    combining: str = "MRC"
    include_turbulence: bool = True
    include_background_noise: bool = True
    include_ase_noise: bool = True
    include_shot_noise: bool = True
    heterodyne_shot_noise_convention: float = 1.0
    include_doppler: bool = True
    include_phase_noise: bool = True

    # Convenience/reference condition; this combination is not a paper-wide default.
    launch_power_dbm: float = 33.0
    zenith_angle_deg: float = 30.0
    pointing_error_urad: float = 20.0
    scattered_power_ratio: float = 0.95
    background_light_radiance_w_m2_sr_um: float = 1e-4
    weather: str = "clear"

    def __post_init__(self):
        if self.integration_points < 257 or self.integration_points % 2 == 0:
            raise ValueError("integration_points must be an odd integer >= 257")
        if self.integration_grid_power < 1.0:
            raise ValueError("integration_grid_power must be at least 1")
        if self.photodiode_responsivity_a_w <= 0:
            raise ValueError("Photodiode responsivity must be positive")
        quantum_efficiency = (
            self.photodiode_responsivity_a_w * self.planck_constant_j_s
            * self.light_speed_m_s / self.wavelength_m / self.electron_charge_c
        )
        if not 0 < quantum_efficiency <= 1:
            raise ValueError("Responsivity implies unphysical photodiode quantum efficiency")
        if self.heterodyne_shot_noise_convention <= 0:
            raise ValueError("Heterodyne complex-noise convention factor must be positive")
        if self.num_receivers < 1:
            raise ValueError("num_receivers must be positive")
        if self.combining.upper() not in {"MRC", "EGC"}:
            raise ValueError("combining must be MRC or EGC")
        if not math.isclose(self.malaga_omega + 2 * self.malaga_b0, 1.0):
            raise ValueError("The paper normalizes Malaga power as Omega + 2*b0 = 1")
        if self.beta_fit_policy not in {"nearest_warn", "strict"}:
            raise ValueError("beta_fit_policy must be nearest_warn or strict")
        if self.beta_fit_relative_tolerance < 0:
            raise ValueError("beta_fit_relative_tolerance must be nonnegative")


def _trapezoid(values, grid):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, grid)
    return np.trapz(values, grid)


def hufnagel_valley_cn2(altitude_m, config):
    h = np.asarray(altitude_m, dtype=np.float64)
    v = config.rms_wind_speed_m_s
    a = config.ground_cn2_m_neg_two_thirds
    return (
        0.00594 * (v / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
        + 2.7e-16 * np.exp(-h / 1500.0)
        + a * np.exp(-h / 100.0)
    )


def satellite_slant_range_m(zenith_angle_deg, config):
    phi = math.radians(float(zenith_angle_deg))
    if not 0.0 <= phi < math.pi / 2:
        raise ValueError("zenith_angle_deg must be in [0, 90)")
    ground_radius = config.earth_radius_m + config.receiver_altitude_m
    satellite_radius = config.earth_radius_m + config.satellite_altitude_m
    return (
        math.sqrt(
            satellite_radius**2
            - ground_radius**2 * math.sin(phi) ** 2
        )
        - ground_radius * math.cos(phi)
    )


def path_altitude_m(path_fraction, slant_range_m, zenith_angle_deg, config):
    xi = np.asarray(path_fraction, dtype=np.float64)
    phi = math.radians(float(zenith_angle_deg))
    ground_radius = config.earth_radius_m + config.receiver_altitude_m
    distance = xi * slant_range_m
    radius = np.sqrt(
        ground_radius**2
        + distance**2
        + 2.0 * ground_radius * distance * math.cos(phi)
    )
    return radius - config.earth_radius_m


def shadowed_rician_second_moment(beta, omega_prime, gamma):
    mean_y = omega_prime + gamma
    beta_independent = omega_prime**2 + 4 * omega_prime * gamma + 2 * gamma**2
    return (
        beta_independent + omega_prime**2 / float(beta)
    ) / mean_y**2


def fit_malaga_beta_eq15(target_second_moment, omega_prime, gamma, max_beta):
    """Nearest natural beta plus an honest Eq.-(15) feasibility diagnostic."""
    mean_y = omega_prime + gamma
    beta_independent = omega_prime**2 + 4 * omega_prime * gamma + 2 * gamma**2
    denominator = target_second_moment * mean_y**2 - beta_independent
    continuous_beta = (
        omega_prime**2 / denominator if denominator > 0 else float("inf")
    )
    continuum_feasible = 1.0 <= continuous_beta <= float(max_beta)
    if math.isfinite(continuous_beta):
        candidates = {
            1,
            int(max_beta),
            min(max(int(math.floor(continuous_beta)), 1), int(max_beta)),
            min(max(int(math.ceil(continuous_beta)), 1), int(max_beta)),
        }
    else:
        candidates = {int(max_beta)}
    beta = min(
        candidates,
        key=lambda value: abs(
            shadowed_rician_second_moment(value, omega_prime, gamma)
            - target_second_moment
        ),
    )
    fitted = shadowed_rician_second_moment(beta, omega_prime, gamma)
    relative_residual = abs(fitted - target_second_moment) / max(
        abs(target_second_moment), 1e-30
    )
    return {
        "beta": int(beta),
        "continuous_beta": continuous_beta,
        "continuum_feasible_within_search": continuum_feasible,
        "fitted_second_moment": fitted,
        "relative_residual": relative_residual,
        "minimum_second_moment": beta_independent / mean_y**2,
        "maximum_natural_second_moment": shadowed_rician_second_moment(
            1, omega_prime, gamma
        ),
    }


def compute_paper_link_budget(
    config,
    launch_power_dbm=None,
    zenith_angle_deg=None,
    pointing_error_urad=None,
    scattered_power_ratio=None,
    background_light_radiance_w_m2_sr_um=None,
    weather=None,
):
    """Implement paper Eqs. (4)-(18) and Table I as scalar diagnostics."""
    launch_power_dbm = (
        config.launch_power_dbm if launch_power_dbm is None else float(launch_power_dbm)
    )
    zenith_angle_deg = (
        config.zenith_angle_deg if zenith_angle_deg is None else float(zenith_angle_deg)
    )
    pointing_error_urad = (
        config.pointing_error_urad
        if pointing_error_urad is None
        else float(pointing_error_urad)
    )
    rho = (
        config.scattered_power_ratio
        if scattered_power_ratio is None
        else float(scattered_power_ratio)
    )
    radiance = (
        config.background_light_radiance_w_m2_sr_um
        if background_light_radiance_w_m2_sr_um is None
        else float(background_light_radiance_w_m2_sr_um)
    )
    weather = (config.weather if weather is None else weather).lower()
    if weather not in WEATHER_ATTENUATION_DB_PER_KM:
        raise ValueError(f"Unsupported weather: {weather}")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("scattered_power_ratio must be in [0, 1]")
    if radiance < 0.0:
        raise ValueError("background light radiance cannot be negative")

    wavelength = config.wavelength_m
    wavenumber = 2.0 * math.pi / wavelength
    slant_range = satellite_slant_range_m(zenith_angle_deg, config)

    theta0 = 1.0  # Collimated input beam: F0 -> infinity.
    lambda0 = 2.0 * slant_range / (wavenumber * config.beam_radius_m**2)
    denominator = theta0**2 + lambda0**2
    theta1 = theta0 / denominator
    lambda1 = lambda0 / denominator
    diffraction_radius = config.beam_radius_m / math.sqrt(theta1**2 + lambda1**2)

    # The Hufnagel-Valley ground term decays on a ~100-m altitude scale. A
    # uniform 4097-point slant grid undersamples it, so cluster samples near xi=0.
    xi = np.linspace(
        0.0, 1.0, config.integration_points, dtype=np.float64
    ) ** config.integration_grid_power
    altitude = path_altitude_m(xi, slant_range, zenith_angle_deg, config)
    cn2 = hufnagel_valley_cn2(altitude, config)
    mu_2d = float(_trapezoid(cn2 * xi ** (5.0 / 3.0), xi))
    spreading_factor = (
        4.35
        * mu_2d
        * lambda1 ** (5.0 / 6.0)
        * wavenumber ** (7.0 / 6.0)
        * slant_range ** (11.0 / 6.0)
    )
    long_term_radius = diffraction_radius * math.sqrt(1.0 + spreading_factor)

    geometric_scintillation_db = -10.0 * math.log10(
        config.receiver_aperture_m**2 / (2.0 * long_term_radius**2)
    )
    pointing_offset_m = slant_range * pointing_error_urad * 1e-6
    pointing_db = (10.0 / math.log(10.0)) * (
        2.0 * pointing_offset_m**2 / long_term_radius**2
    )
    weather_range_m = config.fog_range_m if weather == "fog" else config.atmospheric_range_m
    atmospheric_db = WEATHER_ATTENUATION_DB_PER_KM[weather] * weather_range_m / 1000.0
    total_attenuation_db = geometric_scintillation_db + pointing_db + atmospheric_db

    complex_kernel = (
        xi ** (5.0 / 6.0)
        * (
            lambda1 * xi
            + 1j * (1.0 - (1.0 - theta1) * xi)
        ) ** (5.0 / 6.0)
        - lambda1 ** (5.0 / 6.0) * xi ** (5.0 / 3.0)
    )
    rytov_variance = float(
        8.7
        * wavenumber ** (7.0 / 6.0)
        * slant_range ** (11.0 / 6.0)
        * np.real(_trapezoid(cn2 * complex_kernel, xi))
    )
    rytov_variance = max(rytov_variance, 0.0)
    sigma_lnx2 = (
        0.49
        * rytov_variance
        / (
            1.0
            + (1.0 + theta1) * 0.56 * rytov_variance ** (6.0 / 5.0)
        ) ** (7.0 / 6.0)
    )
    sigma_lny2 = (
        0.51
        * rytov_variance
        / (1.0 + 0.69 * rytov_variance ** (6.0 / 5.0)) ** (5.0 / 6.0)
    )
    alpha = 1.0 / math.expm1(max(sigma_lnx2, 1e-12))

    omega = config.malaga_omega
    b0 = config.malaga_b0
    phase_difference = config.malaga_phase_difference_rad
    omega_prime = (
        omega
        + 2.0 * b0 * rho
        + 2.0 * math.sqrt(2.0 * b0 * omega * rho) * math.cos(phase_difference)
    )
    gamma = 2.0 * b0 * (1.0 - rho)
    target_y_second_moment = math.exp(sigma_lny2)
    if config.malaga_beta is None:
        beta_fit = fit_malaga_beta_eq15(
            target_y_second_moment,
            omega_prime,
            gamma,
            config.max_derived_beta,
        )
        beta = beta_fit["beta"]
        beta_fit_feasible = (
            beta_fit["continuum_feasible_within_search"]
            and beta_fit["relative_residual"]
            <= config.beta_fit_relative_tolerance
        )
        beta_source = "eq15_nearest_natural"
        if config.beta_fit_policy == "strict" and not beta_fit_feasible:
            raise ValueError(
                "Paper-undisclosed Malaga beta/phase cannot satisfy Eq. (15) "
                f"within tolerance: target={target_y_second_moment:.8g}, "
                f"nearest={beta_fit['fitted_second_moment']:.8g}, "
                f"relative residual={beta_fit['relative_residual']:.3%}"
            )
    else:
        beta = int(config.malaga_beta)
        if beta < 1:
            raise ValueError("malaga_beta must be a natural number")
        fitted = shadowed_rician_second_moment(beta, omega_prime, gamma)
        beta_fit = {
            "beta": beta,
            "continuous_beta": None,
            "continuum_feasible_within_search": None,
            "fitted_second_moment": fitted,
            "relative_residual": abs(fitted - target_y_second_moment)
            / max(abs(target_y_second_moment), 1e-30),
            "minimum_second_moment": None,
            "maximum_natural_second_moment": None,
        }
        beta_fit_feasible = None
        beta_source = "explicit_user_assumption"
    fitted_y_second_moment = beta_fit["fitted_second_moment"]

    earth_radius = config.earth_radius_m
    orbit_radius = earth_radius + config.satellite_altitude_m
    orbital_angular_velocity = math.sqrt(
        config.earth_gravitational_parameter / orbit_radius**3
    )
    phi = math.radians(zenith_angle_deg)
    phi_max = math.pi / 2.0
    eta = math.cos(
        math.acos(earth_radius / orbit_radius * math.cos(phi_max)) - phi_max
    )
    optical_frequency_hz = config.light_speed_m_s / wavelength
    doppler_denominator = math.sqrt(
        earth_radius**2
        + orbit_radius**2
        - 2.0 * earth_radius * orbit_radius * math.cos(phi) * eta
    )
    doppler_hz = -(
        optical_frequency_hz
        / config.light_speed_m_s
        * orbital_angular_velocity
        * earth_radius
        * orbit_radius
        * math.sin(phi)
        * eta
        / doppler_denominator
    )
    residual_doppler_hz = config.doppler_residual_ratio * doppler_hz

    background_noise_w = (
        radiance
        * math.pi**2
        * config.field_of_view_rad**2
        * config.receiver_aperture_m**2
        / 16.0
        * config.optical_filter_bandwidth_um
    )
    amplifier_gain_linear = 10.0 ** (config.amplifier_gain_db / 10.0)
    noise_factor = 10.0 ** (config.amplifier_noise_figure_db / 10.0)
    spontaneous_emission_factor = (
        noise_factor - 1.0 / amplifier_gain_linear
    ) / (2.0 * (1.0 - 1.0 / amplifier_gain_linear))
    ase_noise_w = (
        2.0
        * spontaneous_emission_factor
        * config.planck_constant_j_s
        * optical_frequency_hz
        * (amplifier_gain_linear - 1.0)
        * config.symbol_rate_baud
    )

    # One detected complex temporal mode per Nyquist symbol. Equivalent
    # input field noise is independent between receiver branches, not common
    # transmitter ASE. No unknown LO power or arbitrary thermal floor is fitted.
    quantum_efficiency = (
        config.photodiode_responsivity_a_w * config.planck_constant_j_s
        * optical_frequency_hz / config.electron_charge_c
    )
    receiver_shot_noise_w = (
        config.heterodyne_shot_noise_convention * config.electron_charge_c
        / config.photodiode_responsivity_a_w * config.symbol_rate_baud
    )

    return {
        "channel_version": PAPER_FSO_CHANNEL_VERSION,
        "launch_power_dbm": launch_power_dbm,
        "launch_power_w": 10.0 ** ((launch_power_dbm - 30.0) / 10.0),
        "zenith_angle_deg": zenith_angle_deg,
        "pointing_error_urad": pointing_error_urad,
        "scattered_power_ratio": rho,
        "background_light_radiance_w_m2_sr_um": radiance,
        "weather": weather,
        "slant_range_m": slant_range,
        "theta1": theta1,
        "lambda1": lambda1,
        "diffraction_radius_m": diffraction_radius,
        "long_term_radius_m": long_term_radius,
        "geometric_scintillation_db": geometric_scintillation_db,
        "pointing_db": pointing_db,
        "atmospheric_db": atmospheric_db,
        "total_attenuation_db": total_attenuation_db,
        "rytov_variance": rytov_variance,
        "sigma_lnx2": sigma_lnx2,
        "sigma_lny2": sigma_lny2,
        "malaga_alpha": alpha,
        "malaga_beta": beta,
        "malaga_beta_source": beta_source,
        "malaga_beta_fit_feasible": beta_fit_feasible,
        "malaga_beta_continuous": beta_fit["continuous_beta"],
        "malaga_beta_fit_relative_residual": beta_fit["relative_residual"],
        "malaga_beta_minimum_second_moment": beta_fit["minimum_second_moment"],
        "malaga_beta_maximum_natural_second_moment": beta_fit[
            "maximum_natural_second_moment"
        ],
        "malaga_omega_prime": omega_prime,
        "malaga_gamma": gamma,
        "malaga_mean_y": omega_prime + gamma,
        "target_y_second_moment": target_y_second_moment,
        "fitted_y_second_moment": fitted_y_second_moment,
        "doppler_hz": doppler_hz,
        "residual_doppler_hz": residual_doppler_hz,
        "background_noise_w": background_noise_w,
        "amplifier_gain_linear": amplifier_gain_linear,
        "ase_noise_w": ase_noise_w,
        "photodiode_quantum_efficiency": quantum_efficiency,
        "receiver_shot_noise_w": receiver_shot_noise_w,
        "complex_shot_noise_convention": config.heterodyne_shot_noise_convention,
    }


def _real_dtype(complex_dtype):
    if complex_dtype == torch.complex128:
        return torch.float64
    return torch.float32


def _complex_normal(shape, device, complex_dtype, generator=None):
    real_dtype = _real_dtype(complex_dtype)
    real = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    imag = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    return torch.complex(real, imag) / math.sqrt(2.0)


def _gamma_unit_mean(shape_parameter, shape, device, dtype, generator=None):
    concentration = torch.full(
        shape,
        float(shape_parameter),
        device=device,
        dtype=dtype,
    )
    # PyTorch distributions do not expose a generator argument. This native
    # primitive does and is available in the paper's PyTorch 2.0 runtime.
    return torch._standard_gamma(concentration, generator=generator) / float(
        shape_parameter
    )


class PaperMalagaFSOChannel:
    """Paper Eq. (3) channel with Eqs. (4)-(18) and ideal coherent DSP.

    This is a symbol-rate matched-filter equivalent. Block Malaga fading is
    sampled independently per image/receiver. Doppler and LO phase are applied
    before ideal CFR/CPR/equalization through the returned complex channel gain.
    """

    def __init__(self, config=None):
        self.config = config or PaperFSOSystemConfig()
        self._budget_cache = {}
        self._reproducibility_config_cache = None
        self._warned_infeasible_beta_fit = False

    def reproducibility_config(self):
        if self._reproducibility_config_cache is None:
            reference = compute_paper_link_budget(self.config)
            self._reproducibility_config_cache = {
                "channel_version": PAPER_FSO_CHANNEL_VERSION,
                "system": asdict(self.config),
                "undisclosed_assumptions": PAPER_UNDISCLOSED_RECEIVER_ASSUMPTIONS,
                "reference_beta_diagnostic": {
                    "beta": reference["malaga_beta"],
                    "source": reference["malaga_beta_source"],
                    "fit_feasible": reference["malaga_beta_fit_feasible"],
                    "target_second_moment": round(
                        reference["target_y_second_moment"], 12
                    ),
                    "fitted_second_moment": round(
                        reference["fitted_y_second_moment"], 12
                    ),
                    "relative_residual": round(
                        reference["malaga_beta_fit_relative_residual"], 12
                    ),
                },
            }
        return self._reproducibility_config_cache

    def _budget(self, **condition):
        key = tuple(sorted(condition.items()))
        if key not in self._budget_cache:
            self._budget_cache[key] = compute_paper_link_budget(
                self.config, **condition
            )
        budget = self._budget_cache[key]
        if (
            self.config.malaga_beta is None
            and self.config.beta_fit_policy == "nearest_warn"
            and not budget["malaga_beta_fit_feasible"]
            and not self._warned_infeasible_beta_fit
        ):
            warnings.warn(
                "The paper omits Malaga beta and phase, and Eq. (15) has no "
                "fit within the configured range/tolerance for this condition; "
                f"using beta={budget['malaga_beta']} with relative moment "
                f"residual={budget['malaga_beta_fit_relative_residual']:.3%}. "
                "Set an explicit malaga_beta or beta_fit_policy='strict'.",
                RuntimeWarning,
            )
            self._warned_infeasible_beta_fit = True
        return budget

    def sample_malaga_power(self, shape, budget, device, dtype, generator=None):
        if not self.config.include_turbulence:
            return torch.ones(shape, device=device, dtype=dtype)
        large_scale = _gamma_unit_mean(
            budget["malaga_alpha"], shape, device, dtype, generator
        )
        shadowing = _gamma_unit_mean(
            budget["malaga_beta"], shape, device, dtype, generator
        )
        rho = budget["scattered_power_ratio"]
        coherent_phasor = (
            math.sqrt(self.config.malaga_omega)
            + math.sqrt(2.0 * self.config.malaga_b0 * rho)
            * complex(
                math.cos(-self.config.malaga_phase_difference_rad),
                math.sin(-self.config.malaga_phase_difference_rad),
            )
        )
        diffuse = _complex_normal(
            shape,
            device,
            torch.complex128 if dtype == torch.float64 else torch.complex64,
            generator,
        ) * math.sqrt(budget["malaga_gamma"])
        small_field = shadowing.sqrt().to(diffuse.dtype) * coherent_phasor + diffuse
        small_power = small_field.abs().square().to(dtype)
        return large_scale * small_power / budget["malaga_mean_y"]

    @torch.no_grad()
    def __call__(
        self,
        symbols,
        launch_power_dbm=None,
        zenith_angle_deg=None,
        pointing_error_urad=None,
        scattered_power_ratio=None,
        background_light_radiance_w_m2_sr_um=None,
        weather=None,
        generator=None,
    ):
        if symbols.ndim != 2 or not torch.is_complex(symbols):
            raise ValueError("symbols must be complex baseband with shape [batch, symbols]")
        if generator is not None and not paper_generator_device_matches(
            generator, symbols.device
        ):
            raise ValueError("generator and symbols must use the same device")

        condition = {
            "launch_power_dbm": (
                self.config.launch_power_dbm
                if launch_power_dbm is None
                else float(launch_power_dbm)
            ),
            "zenith_angle_deg": (
                self.config.zenith_angle_deg
                if zenith_angle_deg is None
                else float(zenith_angle_deg)
            ),
            "pointing_error_urad": (
                self.config.pointing_error_urad
                if pointing_error_urad is None
                else float(pointing_error_urad)
            ),
            "scattered_power_ratio": (
                self.config.scattered_power_ratio
                if scattered_power_ratio is None
                else float(scattered_power_ratio)
            ),
            "background_light_radiance_w_m2_sr_um": (
                self.config.background_light_radiance_w_m2_sr_um
                if background_light_radiance_w_m2_sr_um is None
                else float(background_light_radiance_w_m2_sr_um)
            ),
            "weather": self.config.weather if weather is None else str(weather),
        }
        budget = self._budget(**condition)
        batch, symbol_count = symbols.shape
        receivers = self.config.num_receivers
        real_dtype = _real_dtype(symbols.dtype)
        fading_shape = (batch, receivers, 1)
        scintillation_power = self.sample_malaga_power(
            fading_shape,
            budget,
            symbols.device,
            real_dtype,
            generator,
        )

        attenuation_amplitude = 10.0 ** (-budget["total_attenuation_db"] / 20.0)
        # Table I reports transmitted optical power after the transmitter AMP.
        # Therefore its 30-dB gain must not be applied to the signal a second time.
        propagation_amplitude = attenuation_amplitude * scintillation_power.sqrt()

        time_index = torch.arange(
            symbol_count, device=symbols.device, dtype=real_dtype
        )
        phase = torch.zeros(
            (batch, receivers, symbol_count),
            device=symbols.device,
            dtype=real_dtype,
        )
        if self.config.include_doppler:
            phase = phase + (
                2.0
                * math.pi
                * budget["residual_doppler_hz"]
                / self.config.symbol_rate_baud
                * time_index
            ).view(1, 1, -1)
        if self.config.include_phase_noise and self.config.lo_linewidth_hz > 0:
            increment_std = math.sqrt(
                2.0
                * math.pi
                * self.config.lo_linewidth_hz
                / self.config.symbol_rate_baud
            )
            increments = torch.randn(
                phase.shape,
                device=symbols.device,
                dtype=real_dtype,
                generator=generator,
            ) * increment_std
            phase = phase + increments.cumsum(dim=-1)

        propagation_gain = propagation_amplitude.to(symbols.dtype) * torch.exp(
            1j * phase
        ).to(symbols.dtype)
        channel_gain = math.sqrt(budget["launch_power_w"]) * propagation_gain
        received = channel_gain * symbols[:, None, :]

        if self.config.include_background_noise and budget["background_noise_w"] > 0:
            received = received + _complex_normal(
                received.shape, symbols.device, symbols.dtype, generator
            ) * math.sqrt(budget["background_noise_w"])
        if self.config.include_shot_noise and budget["receiver_shot_noise_w"] > 0:
            received = received + _complex_normal(
                received.shape, symbols.device, symbols.dtype, generator
            ) * math.sqrt(budget["receiver_shot_noise_w"])
        if self.config.include_ase_noise and budget["ase_noise_w"] > 0:
            transmitter_ase = _complex_normal(
                (batch, 1, symbol_count),
                symbols.device,
                symbols.dtype,
                generator,
            ) * math.sqrt(budget["ase_noise_w"])
            received = received + propagation_gain * transmitter_ase

        # The transmitter-EDFA ASE is one common source waveform, so diversity
        # cannot average it away. Receiver background noise is branch-independent.
        equivalent_noise_variance = torch.zeros(
            (batch, 1), device=symbols.device, dtype=real_dtype
        )
        if self.config.include_ase_noise:
            equivalent_noise_variance = equivalent_noise_variance + (
                budget["ase_noise_w"] / budget["launch_power_w"]
            )
        receiver_independent_noise_w = (
            budget["background_noise_w"] if self.config.include_background_noise else 0.0
        ) + (
            budget["receiver_shot_noise_w"] if self.config.include_shot_noise else 0.0
        )
        if receiver_independent_noise_w > 0:
            if receivers == 1 or self.config.combining.upper() == "MRC":
                combined_signal_power = channel_gain[..., :1].abs().square().sum(dim=1)
                equivalent_noise_variance = equivalent_noise_variance + (
                    receiver_independent_noise_w
                    / combined_signal_power.clamp_min(torch.finfo(real_dtype).tiny)
                )
            else:
                combined_signal_amplitude = channel_gain[..., :1].abs().sum(dim=1)
                equivalent_noise_variance = equivalent_noise_variance + (
                    receivers
                    * receiver_independent_noise_w
                    / combined_signal_amplitude.square().clamp_min(
                        torch.finfo(real_dtype).tiny
                    )
                )
        if bool((equivalent_noise_variance > 0).any()):
            instantaneous_snr = equivalent_noise_variance.clamp_min(
                torch.finfo(real_dtype).tiny
            ).reciprocal()
            snr_linear_sum = float(instantaneous_snr.sum().item())
            snr_sample_count = int(instantaneous_snr.numel())
            mean_snr_linear = snr_linear_sum / max(snr_sample_count, 1)
            mean_snr_db = 10.0 * math.log10(max(mean_snr_linear, 1e-30))
        else:
            snr_linear_sum = float("inf")
            snr_sample_count = batch
            mean_snr_db = float("inf")
        diagnostics = {
            "channel_version": PAPER_FSO_CHANNEL_VERSION,
            "mean_snr_db": mean_snr_db,
            "snr_linear_sum": snr_linear_sum,
            "snr_sample_count": snr_sample_count,
            "num_receivers": receivers,
            "combining": self.config.combining.upper(),
            "budget": budget,
        }

        if receivers == 1:
            return FSOChannelOutput(
                symbols=received[:, 0, :],
                channel_gain=channel_gain[:, 0, :],
                is_equalized=False,
                diagnostics=diagnostics,
            )

        if self.config.combining.upper() == "MRC":
            denominator = (
                channel_gain.abs().square().sum(dim=1)
            ).clamp_min(torch.finfo(real_dtype).tiny)
            combined = (channel_gain.conj() * received).sum(dim=1) / denominator
        else:
            magnitudes = channel_gain.abs().clamp_min(1e-30)
            combined = (
                channel_gain.conj() / magnitudes * received
            ).sum(dim=1) / magnitudes.sum(dim=1).clamp_min(1e-30)
        return FSOChannelOutput(
            symbols=combined,
            is_equalized=True,
            diagnostics=diagnostics,
        )


@dataclass
class PaperFSOIndexChannel:
    receiver: object
    modulations: tuple = ("QPSK", "8QAM", "16QAM")
    bits_per_index: int = BITS_PER_INDEX

    @property
    def receiver_config(self):
        return self.receiver.reproducibility_config()

    def reproducibility_config(self):
        return {
            "qam_mapping_version": QAM_MAPPING_VERSION,
            "modulations": tuple(name.upper() for name in self.modulations),
            "bits_per_index": self.bits_per_index,
            "receiver": self.receiver_config,
        }

    def __call__(
        self,
        indices,
        modulation="QPSK",
        return_statistics=False,
        **channel_kwargs,
    ):
        modulation = modulation.upper()
        if modulation not in tuple(name.upper() for name in self.modulations):
            raise ValueError(f"Unsupported paper modulation: {modulation}")
        link = QAMIndexLink(
            self.receiver,
            modulation=modulation,
            bits_per_index=self.bits_per_index,
            receiver_config=self.receiver_config,
        )
        return link(
            indices,
            return_statistics=return_statistics,
            **channel_kwargs,
        )


def paper_fig4_conditions():
    return [
        {
            "label": f"{modulation}, Pt={power} dBm, zenith={zenith} deg",
            "channel_kwargs": {
                "modulation": modulation,
                "launch_power_dbm": float(power),
                "zenith_angle_deg": float(zenith),
                "pointing_error_urad": 20.0,
                "scattered_power_ratio": 0.95,
                "background_light_radiance_w_m2_sr_um": 1e-4,
                "weather": "clear",
            },
        }
        for power in (30, 33, 36)
        for modulation in ("QPSK", "8QAM", "16QAM")
        for zenith in range(0, 81, 10)
    ]


def paper_fig6_pointing_conditions():
    return [
        {
            "label": (
                f"{modulation}, Pt=36 dBm, zenith={zenith} deg, "
                f"pointing={pointing} urad"
            ),
            "channel_kwargs": {
                "modulation": modulation,
                "launch_power_dbm": 36.0,
                "zenith_angle_deg": float(zenith),
                "pointing_error_urad": float(pointing),
                "scattered_power_ratio": 0.95,
                "background_light_radiance_w_m2_sr_um": 1e-4,
                "weather": "clear",
            },
        }
        for modulation in ("QPSK", "8QAM", "16QAM")
        for zenith in (0, 30, 60)
        for pointing in range(18, 23)
    ]


def paper_fig6_background_conditions():
    return [
        {
            "label": (
                f"{modulation}, Pt={power} dBm, "
                f"BGL={radiance:g} W/m2/sr/um"
            ),
            "channel_kwargs": {
                "modulation": modulation,
                "launch_power_dbm": float(power),
                "zenith_angle_deg": 0.0,
                "pointing_error_urad": 19.0,
                "scattered_power_ratio": 0.95,
                "background_light_radiance_w_m2_sr_um": float(radiance),
                "weather": "clear",
            },
        }
        for modulation in ("QPSK", "8QAM", "16QAM")
        for radiance in (1e-4, 1e-2, 1.0)
        for power in range(28, 37)
    ]


def paper_fig7_weather_conditions():
    return [
        {
            "label": f"QPSK, Pt={power} dBm, weather={weather}",
            "channel_kwargs": {
                "modulation": "QPSK",
                "launch_power_dbm": float(power),
                "zenith_angle_deg": 0.0,
                "pointing_error_urad": 15.0,
                "scattered_power_ratio": 0.95,
                "background_light_radiance_w_m2_sr_um": 1e-4,
                "weather": weather,
            },
        }
        for weather in ("clear", "dust", "rain", "snow", "fog")
        for power in range(28, 37)
    ]


def paper_fig8_turbulence_conditions():
    """Return the common FSO-SC rho/Pt sweep, not the BPG codec pipelines."""
    return [
        {
            "label": f"16QAM, Pt={power} dBm, rho={rho}",
            "channel_kwargs": {
                "modulation": "16QAM",
                "launch_power_dbm": float(power),
                "zenith_angle_deg": 0.0,
                "pointing_error_urad": 19.0,
                "scattered_power_ratio": float(rho),
                "background_light_radiance_w_m2_sr_um": 1e-4,
                "weather": "clear",
            },
        }
        for rho in (0.95, 0.75, 0.596)
        for power in range(28, 37)
    ]


def paper_fig8_diversity_index_channels(base_config=None):
    """Expose the paper's 2/4-Rx EGC/MRC comparisons.

    The paper does not disclose inter-branch turbulence correlation, so the
    receiver implementation records its independent-block-fade assumption.
    """
    base_config = PaperFSOSystemConfig() if base_config is None else base_config
    channels = {
        "1Rx": PaperFSOIndexChannel(
            PaperMalagaFSOChannel(replace(base_config, num_receivers=1))
        )
    }
    for receivers in (2, 4):
        for combining in ("EGC", "MRC"):
            label = f"{receivers}Rx-{combining}"
            receiver_config = replace(
                base_config,
                num_receivers=receivers,
                combining=combining,
            )
            channels[label] = PaperFSOIndexChannel(
                PaperMalagaFSOChannel(receiver_config)
            )
    return channels


def paper_fso_equation_self_test():
    config = PaperFSOSystemConfig()
    reference = compute_paper_link_budget(
        config,
        launch_power_dbm=33.0,
        zenith_angle_deg=0.0,
        pointing_error_urad=20.0,
        weather="clear",
    )
    assert abs(reference["slant_range_m"] - 620e3) < 1e-6
    assert abs(reference["geometric_scintillation_db"] - 28.69) < 0.05
    assert abs(reference["pointing_db"] - 35.68) < 0.05
    assert abs(reference["atmospheric_db"] - 2.94) < 1e-9
    # The paper omits beta and the coherent phase. With the documented pi/2
    # phase, its Eq. (15) has no feasible beta here; this must stay explicit.
    assert reference["malaga_beta_fit_feasible"] is False
    assert reference["malaga_beta_fit_relative_residual"] > 0.0
    try:
        compute_paper_link_budget(
            PaperFSOSystemConfig(beta_fit_policy="strict"),
            zenith_angle_deg=0.0,
            scattered_power_ratio=0.95,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Strict beta fitting must reject the underdetermined fit")
    expected_ranges_km = {
        0: 620.000,
        20: 655.958,
        30: 705.628,
        60: 1107.733,
        80: 1977.252,
    }
    for zenith, expected_km in expected_ranges_km.items():
        actual_km = satellite_slant_range_m(zenith, config) / 1000.0
        assert abs(actual_km - expected_km) < 0.001
    expected_weather_db = {
        "clear": 2.94,
        "dust": 20.86,
        "rain": 28.06,
        "snow": 42.90,
        "fog": 271.832,
    }
    for weather, expected_db in expected_weather_db.items():
        weather_budget = compute_paper_link_budget(config, weather=weather)
        assert abs(weather_budget["atmospheric_db"] - expected_db) < 0.001
    peak = compute_paper_link_budget(config, zenith_angle_deg=20.0)
    high = compute_paper_link_budget(config, zenith_angle_deg=80.0)
    assert abs(peak["doppler_hz"]) > abs(high["doppler_hz"])

    convergence_reference = compute_paper_link_budget(
        PaperFSOSystemConfig(integration_points=32769),
        zenith_angle_deg=0.0,
        pointing_error_urad=20.0,
    )
    assert abs(
        reference["rytov_variance"] / convergence_reference["rytov_variance"] - 1.0
    ) < 5e-3
    assert abs(
        reference["malaga_alpha"] / convergence_reference["malaga_alpha"] - 1.0
    ) < 5e-3

    receiver = PaperMalagaFSOChannel(config)
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    sample_a = receiver.sample_malaga_power(
        (4096, 1, 1), reference, torch.device("cpu"), torch.float32, generator_a
    )
    sample_b = receiver.sample_malaga_power(
        (4096, 1, 1), reference, torch.device("cpu"), torch.float32, generator_b
    )
    assert torch.equal(sample_a, sample_b)
    assert bool((sample_a >= 0).all()) and bool(torch.isfinite(sample_a).all())
    assert 0.9 < float(sample_a.mean()) < 1.1
    print(
        "Paper FSO Eq. (3)-(18) self-test passed: "
        f"asg={reference['geometric_scintillation_db']:.2f} dB, "
        f"ape={reference['pointing_db']:.2f} dB, "
        f"alpha={reference['malaga_alpha']:.2f}, beta={reference['malaga_beta']} "
        f"(Eq15 residual={reference['malaga_beta_fit_relative_residual']:.2%})"
    )


paper_fso_equation_self_test()
