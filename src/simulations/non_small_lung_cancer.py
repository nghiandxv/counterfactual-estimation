from abc import abstractmethod
from copy import deepcopy
from typing import Any

import numpy as np
from enum import IntEnum

from dataclasses import field
from scipy.stats import truncnorm
from scipy.special import expit as sigmoid
from src.utils.misc import dataclass, new_rng, Seed, tqdm


def calc_spherical_volume(diameter):
    diameter = np.asarray(diameter)
    radius = diameter / 2.0
    return 4.0 / 3.0 * np.pi * (radius**3.0)


def calc_spherical_diameter(volume):
    volume = np.asarray(volume)
    radius = (volume * 3.0 / 4.0 / np.pi) ** (1.0 / 3.0)
    return 2.0 * radius


DEATH_CONDITION_DIAMETER = 13.0
DEATH_CONDITION_VOLUME = float(calc_spherical_volume(DEATH_CONDITION_DIAMETER))

TumorStages = IntEnum('TumorStages', ['I', 'II', 'IIIA', 'IIIB', 'IV'], start=0)
PatientGroups = IntEnum('PatientGroups', ['I', 'II', 'III'], start=0)
PatientStatuses = IntEnum('PatientStatuses', ['TUMOR', 'RECOVERED', 'DEAD'], start=0)


@dataclass
class TruncNormalDist:
    mu: float
    sigma: float
    lower: float
    upper: float

    def sample(self, size: int, seed: Seed = 0):
        rng = new_rng(seed)
        samples = truncnorm.rvs(self.lower, self.upper, size=size, random_state=rng)
        return samples * self.sigma + self.mu


@dataclass
class TruncLogNormalDist(TruncNormalDist):
    @property
    def standard_lower(self):
        return (np.log(self.lower) - self.mu) / self.sigma

    @property
    def standard_upper(self):
        return (np.log(self.upper) - self.mu) / self.sigma

    def sample(self, size: int, seed: Seed = 0):
        rng = new_rng(seed)
        samples = truncnorm.rvs(
            self.standard_lower, self.standard_upper, size=size, random_state=rng
        )
        return np.exp(samples * self.sigma + self.mu)


@dataclass
class TumorDist(TruncLogNormalDist):
    rate: int | float


@dataclass
class ParamDist(TruncNormalDist):
    lower: float = 0
    upper: float = np.inf


@dataclass
class SimulationParams:
    rho_dist: ParamDist
    alpha_dist: ParamDist
    beta_c_dist: ParamDist

    alpha_beta_ratio: float
    alpha_rho_corr: float

    k: float

    @property
    def alpha_rho_mean(self):
        return np.asarray([self.alpha_dist.mu, self.rho_dist.mu])

    @property
    def alpha_rho_cov(self):
        return np.asarray(
            [
                [
                    self.alpha_dist.sigma**2,
                    self.alpha_rho_corr * self.alpha_dist.sigma * self.rho_dist.sigma,
                ],
                [
                    self.alpha_rho_corr * self.alpha_dist.sigma * self.rho_dist.sigma,
                    self.rho_dist.sigma**2,
                ],
            ]
        )

    def sample_alpha_rho(self, size: int, seed: Seed = 0):
        rng = new_rng(seed)
        alphas = np.full(size, fill_value=np.nan)
        rhos = np.full(size, fill_value=np.nan)

        idx = 0
        while idx <= size - 1:  # rejection sampling
            alpha, rho = rng.multivariate_normal(
                mean=self.alpha_rho_mean, cov=self.alpha_rho_cov
            )
            if alpha < self.alpha_dist.lower or alpha > self.alpha_dist.upper:
                continue
            if rho < self.rho_dist.lower or rho > self.rho_dist.upper:
                continue
            alphas[idx], rhos[idx] = alpha, rho
            idx += 1
        return alphas, rhos


@dataclass
class PatientParams:
    group: PatientGroups

    initial_tumor_stage: TumorStages
    initial_tumor_volume: float
    tumor_rho: float
    tumor_k: float

    radio_alpha: float
    radio_beta: float

    chemo_beta_c: float


DEFAULT_TUMOR_STAGE_TO_DIST = {
    TumorStages.I: TumorDist(mu=1.72, sigma=4.70, lower=0.3, upper=5.0, rate=1432),
    TumorStages.II: TumorDist(
        mu=1.96, sigma=1.63, lower=0.3, upper=DEATH_CONDITION_DIAMETER, rate=128
    ),
    TumorStages.IIIA: TumorDist(
        mu=1.91, sigma=9.40, lower=0.3, upper=DEATH_CONDITION_DIAMETER, rate=1306
    ),
    TumorStages.IIIB: TumorDist(
        mu=2.76, sigma=6.87, lower=0.3, upper=DEATH_CONDITION_DIAMETER, rate=7248
    ),
    TumorStages.IV: TumorDist(
        mu=3.86, sigma=8.82, lower=0.3, upper=DEATH_CONDITION_DIAMETER, rate=12840
    ),
}

DEFAULT_SIMULATION_PARAMS = SimulationParams(
    rho_dist=ParamDist(mu=7.0 * 1e-5, sigma=7.23 * 1e-3),
    alpha_dist=ParamDist(mu=0.0398, sigma=0.168),
    beta_c_dist=ParamDist(mu=0.028, sigma=0.0007),
    alpha_beta_ratio=10,
    alpha_rho_corr=0.87,
    k=calc_spherical_volume(30),
)


def generate_patient_profiles(
    num_patients: int,
    tumor_stage_to_dist: dict[str, TumorDist] = DEFAULT_TUMOR_STAGE_TO_DIST,
    simulation_params: SimulationParams = DEFAULT_SIMULATION_PARAMS,
    seed: Seed = 0,
):
    rng = new_rng(seed)
    groups = rng.choice(list(PatientGroups), num_patients)

    total_tumor_rates = sum(dist.rate for dist in tumor_stage_to_dist.values())
    tumor_proportions = {
        stage: dist.rate / total_tumor_rates
        for stage, dist in tumor_stage_to_dist.items()
    }

    initial_tumor_stages = rng.choice(
        list(tumor_proportions.keys()),
        size=num_patients,
        p=list(tumor_proportions.values()),
    )

    initial_tumor_diameters = []
    for stage, dist in tumor_stage_to_dist.items():
        num_samples = np.sum(initial_tumor_stages == stage)
        initial_tumor_diameters.extend(dist.sample(size=num_samples, seed=rng))

    initial_tumor_volumes = calc_spherical_volume(np.asarray(initial_tumor_diameters))

    alphas, rhos = simulation_params.sample_alpha_rho(size=num_patients, seed=rng)
    beta = alphas / simulation_params.alpha_beta_ratio
    beta_cs = simulation_params.beta_c_dist.sample(size=num_patients, seed=rng)

    return [
        PatientParams(
            group=params[0],
            initial_tumor_stage=params[1],
            initial_tumor_volume=params[2],
            tumor_rho=params[3],
            tumor_k=params[4],
            radio_alpha=params[5],
            radio_beta=params[6],
            chemo_beta_c=params[7],
        )
        for params in zip(
            groups,
            initial_tumor_stages,
            initial_tumor_volumes,
            rhos,
            np.full(num_patients, simulation_params.k),
            alphas,
            beta,
            beta_cs,
        )
    ]


def add_heterogeneous_effect(
    patient_params: PatientParams,
    simulation_params: SimulationParams = DEFAULT_SIMULATION_PARAMS,
):
    match patient_params.group:
        case PatientGroups.I:
            alpha_adjustments = 0.1 * simulation_params.alpha_dist.mu
            patient_params.radio_alpha += alpha_adjustments
            patient_params.radio_beta += (
                alpha_adjustments * simulation_params.alpha_beta_ratio
            )
        case PatientGroups.III:
            beta_c_adjustments = 0.1 * simulation_params.beta_c_dist.mu
            patient_params.chemo_beta_c += beta_c_adjustments
    return patient_params


@dataclass
class TumorModel:
    @dataclass
    class State:
        volumes: list[float] = field(default_factory=list)
        chemo_concentrations: list[float] = field(default_factory=list)
        patient_statuses: list[PatientStatuses] = field(default_factory=list)

    tumor_rho: float
    tumor_k: float
    radio_alpha: float
    radio_beta: float
    chemo_beta_c: float
    initial_volume: float

    seed: Seed = 0

    chemo_half_life: float = field(default=1.0, init=False)
    noise_sigma: float = field(default=0.01, init=False)

    min_volume: float = field(default=0.0, init=False)
    max_volume: float = field(default=DEATH_CONDITION_VOLUME, init=False)
    tumor_cell_density: float = field(default=5.8 * 1e8, init=False)

    state: State = field(default_factory=State, init=False)

    def __post_init__(self):
        self.rng = new_rng(self.seed)
        self.state.volumes.append(self.initial_volume)
        self.state.chemo_concentrations.append(0)
        self.state.patient_statuses.append(PatientStatuses.TUMOR)

    @property
    def last_volume(self):
        return self.state.volumes[-1]

    @property
    def volumes(self):
        return self.state.volumes

    @property
    def last_chemo_concentration(self):
        return self.state.chemo_concentrations[-1]

    @property
    def last_patient_status(self):
        return self.state.patient_statuses[-1]

    def apply_chemo(self, dose: float):
        return self.chemo_beta_c * dose * self.last_volume

    def apply_radio(self, dose: float):
        return (self.radio_alpha * dose + self.radio_beta * dose**2) * self.last_volume

    def grow(self):
        return (
            1 + self.tumor_rho * np.log(self.tumor_k / self.last_volume)
        ) * self.last_volume

    def decay_residual_chemo(self):
        return np.exp(-np.log(2) / self.chemo_half_life) * self.last_chemo_concentration

    def is_controlled(self, volume: float):
        return np.exp(-volume * self.tumor_cell_density) > self.rng.random()

    def step(self, chemo_dose: float, radio_dose: float):
        chemo_concentration = self.decay_residual_chemo() + chemo_dose

        match self.last_patient_status:
            case PatientStatuses.TUMOR:
                volume = self.grow()
                volume -= self.apply_chemo(chemo_concentration)
                volume -= self.apply_radio(radio_dose)
                volume += self.rng.normal(scale=self.noise_sigma)
                volume = np.clip(volume, self.min_volume, self.max_volume)

                if self.is_controlled(volume) or volume == self.min_volume:
                    patient_status = PatientStatuses.RECOVERED
                elif volume == self.max_volume:
                    patient_status = PatientStatuses.DEAD
                else:
                    patient_status = PatientStatuses.TUMOR

            case PatientStatuses.DEAD:
                volume = self.max_volume
                patient_status = PatientStatuses.DEAD
            case PatientStatuses.RECOVERED:
                volume = self.min_volume
                patient_status = PatientStatuses.RECOVERED

        self.state.volumes.append(float(volume))
        self.state.chemo_concentrations.append(float(chemo_concentration))
        self.state.patient_statuses.append(float(patient_status))


@dataclass
class BaseTreatmentPlan:
    @dataclass
    class State:
        chemo_doses: list[float] = field(default_factory=list)
        radio_doses: list[float] = field(default_factory=list)

        @property
        def chemo_usages(self):
            return [dose > 0 for dose in self.chemo_doses]

        @property
        def radio_usages(self):
            return [dose > 0 for dose in self.radio_doses]

    state: State = field(default_factory=State, init=False)

    @property
    def num_steps(self):
        assert len(self.state.chemo_doses) == len(self.state.radio_doses)
        return len(self.state.chemo_doses)

    @property
    def chemo_dose(self):
        if not self.state.chemo_doses:
            raise ValueError(
                'No chemo doses have been planned yet. '
                'Please run the `step` method first.'
            )
        return self.state.chemo_doses[-1]

    @property
    def radio_dose(self):
        if not self.state.radio_doses:
            raise ValueError(
                'No radio doses have been planned yet. '
                'Please run the `step` method first.'
            )
        return self.state.radio_doses[-1]

    @abstractmethod
    def step(self, tumor_model: TumorModel):
        pass

    def reset(self):
        self.state.chemo_doses.clear()
        self.state.radio_doses.clear()


@dataclass
class DefaultTreatmentPlan(BaseTreatmentPlan):
    @dataclass
    class State(BaseTreatmentPlan.State):
        chemo_probs: list[float] = field(default_factory=list)
        radio_probs: list[float] = field(default_factory=list)

    state: State = field(default_factory=State, init=False)

    chemo_gamma: float
    radio_gamma: float
    chemo_dose_per_usage: float = 5.0
    radio_dose_per_usage: float = 2.0

    chemo_theta: float = DEATH_CONDITION_DIAMETER / 2.0
    radio_theta: float = DEATH_CONDITION_DIAMETER / 2.0

    window_size: int = 15
    seed: Seed = 0

    def __post_init__(self):
        self.rng = new_rng(self.seed)

    def reset(self):
        super().reset()
        self.state.chemo_probs.clear()
        self.state.radio_probs.clear()

    def calc_avg_tumor_diameter(self, tumor_model: TumorModel):
        tumow_volume_window = tumor_model.volumes[-self.window_size :]
        return np.mean(list(map(calc_spherical_diameter, tumow_volume_window)))

    def calc_treatment(
        self,
        avg_tumor_diameter: float,
        gamma: float,
        theta: float,
        dose_per_usage: float,
    ):
        prob = sigmoid(gamma / DEATH_CONDITION_DIAMETER * (avg_tumor_diameter - theta))
        usage = prob > self.rng.random()
        dose = usage * dose_per_usage
        return dose, prob

    def calc_chemo_treatment(self, avg_tumor_diameter: float):
        return self.calc_treatment(
            avg_tumor_diameter,
            self.chemo_gamma,
            self.chemo_theta,
            self.chemo_dose_per_usage,
        )

    def calc_radio_treatment(self, avg_tumor_diameter: float):
        return self.calc_treatment(
            avg_tumor_diameter,
            self.radio_gamma,
            self.radio_theta,
            self.radio_dose_per_usage,
        )

    def step(self, tumor_model: TumorModel):
        if tumor_model.last_patient_status == PatientStatuses.TUMOR:
            avg_tumor_diameter = self.calc_avg_tumor_diameter(tumor_model)

            chemo_dose, chemo_prob = self.calc_chemo_treatment(avg_tumor_diameter)
            radio_dose, radio_prob = self.calc_radio_treatment(avg_tumor_diameter)
        else:
            chemo_dose, chemo_prob = 0, 0
            radio_dose, radio_prob = 0, 0

        self.state.chemo_doses.append(float(chemo_dose))
        self.state.chemo_probs.append(float(chemo_prob))
        self.state.radio_doses.append(float(radio_dose))
        self.state.radio_probs.append(float(radio_prob))


@dataclass
class DoNothingPlan(BaseTreatmentPlan):
    def step(self, tumor_model: TumorModel):
        self.state.chemo_doses.append(0)
        self.state.radio_doses.append(0)


@dataclass
class PredefinedTreatmentPlan(BaseTreatmentPlan):
    chemo_usages: list[bool]
    radio_usages: list[bool]

    chemo_dose_per_usage: float = 5.0
    radio_dose_per_usage: float = 2.0

    def step(self, tumor_model: TumorModel):
        step_idx = self.num_steps
        self.state.chemo_doses.append(
            float(self.chemo_usages[step_idx] * self.chemo_dose_per_usage)
        )
        self.state.radio_doses.append(
            float(self.radio_usages[step_idx] * self.radio_dose_per_usage)
        )

    def reset(self):
        super().reset()
        self.step_idx = 0

    @classmethod
    def generate_finished(
        cls,
        chemo_usages: list[bool],
        radio_usages: list[bool],
        chemo_dose_per_usage: float = 5,
        radio_dose_per_usage: float = 2,
    ):
        plan = cls(
            chemo_usages=chemo_usages,
            radio_usages=radio_usages,
            chemo_dose_per_usage=chemo_dose_per_usage,
            radio_dose_per_usage=radio_dose_per_usage,
        )
        plan.state.chemo_doses = [
            chemo_dose * chemo_dose_per_usage for chemo_dose in chemo_usages
        ]
        plan.state.radio_doses = [
            radio_dose * radio_dose_per_usage for radio_dose in radio_usages
        ]
        return plan


@dataclass
class SimulationResult:
    tumor_model: TumorModel
    treatment_plan: BaseTreatmentPlan


def run_simulation(
    num_patients: int,
    num_time_steps: int,
    treatment_plan: BaseTreatmentPlan,
    seed: Any = 0,
) -> tuple[list[PatientParams], list[SimulationResult]]:
    patient_profiles = generate_patient_profiles(num_patients, seed=seed)
    patient_profiles = list(map(add_heterogeneous_effect, patient_profiles))

    results = []
    for patient_params in tqdm(patient_profiles, desc='Simulating patients'):
        tumor_model = TumorModel(
            tumor_rho=patient_params.tumor_rho,
            tumor_k=patient_params.tumor_k,
            radio_alpha=patient_params.radio_alpha,
            radio_beta=patient_params.radio_beta,
            chemo_beta_c=patient_params.chemo_beta_c,
            initial_volume=patient_params.initial_tumor_volume,
        )
        treatment_plan.reset()
        for _ in range(1, num_time_steps):
            treatment_plan.step(tumor_model)
            tumor_model.step(
                chemo_dose=treatment_plan.chemo_dose,
                radio_dose=treatment_plan.radio_dose,
            )
        results.append(
            SimulationResult(
                tumor_model=deepcopy(tumor_model),
                treatment_plan=deepcopy(treatment_plan),
            )
        )

    return patient_profiles, results


@dataclass
class SimulationResultWithCounterfactuals(SimulationResult):
    counterfactuals_over_time: list[list[SimulationResult]]


def run_simulation_with_counterfactuals(
    num_patients: int,
    num_time_steps: int,
    treatment_plan: DefaultTreatmentPlan,
    seed: Any = 0,
) -> tuple[list[PatientParams], list[SimulationResultWithCounterfactuals]]:
    patient_profiles = generate_patient_profiles(num_patients, seed=seed)
    patient_profiles = list(map(add_heterogeneous_effect, patient_profiles))

    results: tuple[TumorModel, BaseTreatmentPlan] = []

    chemo_dose_per_usage = treatment_plan.chemo_dose_per_usage
    radio_dose_per_usage = treatment_plan.radio_dose_per_usage

    for patient_params in tqdm(patient_profiles, desc='Simulating patients'):
        tumor_model = TumorModel(
            tumor_rho=patient_params.tumor_rho,
            tumor_k=patient_params.tumor_k,
            radio_alpha=patient_params.radio_alpha,
            radio_beta=patient_params.radio_beta,
            chemo_beta_c=patient_params.chemo_beta_c,
            initial_volume=patient_params.initial_tumor_volume,
        )
        treatment_plan.reset()

        counterfactuals_over_time = []

        for _ in range(1, num_time_steps):
            reference_tumor_model = deepcopy(tumor_model)
            reference_treatment_plan = deepcopy(treatment_plan)

            treatment_plan.step(tumor_model)

            factual_chemo_dose = treatment_plan.chemo_dose
            factual_radio_dose = treatment_plan.radio_dose

            tumor_model.step(
                chemo_dose=factual_chemo_dose,
                radio_dose=factual_radio_dose,
            )

            counterfactuals = []

            for chemo_dose, radio_dose in [
                (0, 0),
                (0, radio_dose_per_usage),
                (chemo_dose_per_usage, 0),
                (chemo_dose_per_usage, radio_dose_per_usage),
            ]:
                if (
                    chemo_dose == factual_chemo_dose
                    and radio_dose == factual_radio_dose
                ):
                    continue

                counterfactual_tumor_model = deepcopy(reference_tumor_model)
                counterfactual_tumor_model.step(
                    chemo_dose=chemo_dose,
                    radio_dose=radio_dose,
                )
                predefined_treatment_plan = PredefinedTreatmentPlan.generate_finished(
                    chemo_usages=reference_treatment_plan.state.chemo_usages
                    + [chemo_dose > 0],
                    radio_usages=reference_treatment_plan.state.radio_usages
                    + [radio_dose > 0],
                    chemo_dose_per_usage=chemo_dose_per_usage,
                    radio_dose_per_usage=radio_dose_per_usage,
                )

                counterfactuals.append(
                    SimulationResult(
                        tumor_model=counterfactual_tumor_model,
                        treatment_plan=predefined_treatment_plan,
                    )
                )

            counterfactuals_over_time.append(counterfactuals)

        results.append(
            SimulationResultWithCounterfactuals(
                tumor_model=deepcopy(tumor_model),
                treatment_plan=deepcopy(treatment_plan),
                counterfactuals_over_time=counterfactuals_over_time,
            )
        )

    return patient_profiles, results


def plot_results(results: list[SimulationResult]):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()

    for idx, result in enumerate(results):
        tumor_model, treatment_plan = result.tumor_model, result.treatment_plan
        tumor_volumes = np.asarray(tumor_model.state.volumes)
        chemo_usages = np.asarray(treatment_plan.state.chemo_usages)
        radio_usages = np.asarray(treatment_plan.state.radio_usages)

        ax.plot(tumor_volumes)

        arange = np.arange(len(chemo_usages))
        chemo_label = 'Chemo' if idx == 0 else None
        radio_label = 'Radio' if idx == 0 else None
        ax.scatter(
            arange[chemo_usages],
            tumor_volumes[:-1][chemo_usages],
            color='red',
            label=chemo_label,
            marker='+',
        )
        ax.scatter(
            arange[radio_usages],
            tumor_volumes[:-1][radio_usages],
            color='blue',
            label=radio_label,
            marker='x',
        )

    ax.set_title('Tumor Volume')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Volume')
    ax.legend()
    plt.show()
