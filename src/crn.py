from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.utils.data as data_utils
import einops as ei
import numpy as np
import jaxtyping as jty
from skorch import NeuralNet
from src.utils.tensor import expect_shape
from src.simulations.non_small_lung_cancer import (
    DefaultTreatmentPlan,
    PatientParams,
    SimulationResult,
    run_simulation,
    PatientStatuses,
)
from src.utils.misc import Seed, get_device, tqdm
from typing import TypedDict
from beartype import beartype

from src.modules.gradient_reversal import GradientReversal


class CRNModule(nn.Module):
    @beartype
    class Output(TypedDict):
        used_treatment_logits_series: jty.Float[torch.Tensor, 'b t n']
        estimated_outcomes_series: jty.Float[torch.Tensor, 'b t o']
        next_representation_series: jty.Float[torch.Tensor, 'b t h']

    @beartype
    class ForcastOutput(TypedDict):
        outcome: jty.Float[torch.Tensor, 'b o']
        representation: jty.Float[torch.Tensor, 'b h']

    def __init__(
        self,
        num_covariates: int,
        num_outcomes: int,
        num_hidden_units: int,
        num_treatments: int,
        dropout: float = 0.5,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.num_covariates = num_covariates
        self.num_outcomes = num_outcomes
        self.num_hidden_units = num_hidden_units
        self.num_treatments = num_treatments
        self.alpha = alpha

        self.lstm = nn.LSTM(
            input_size=self.num_covariates + 1,
            hidden_size=num_hidden_units,
            dropout=dropout,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.representor = nn.Sequential(
            nn.ELU(),
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.ELU(),
        )

        self.treatment_classifier = nn.Sequential(
            GradientReversal(self.alpha),
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            nn.ELU(),
            nn.Linear(self.num_hidden_units, self.num_treatments),
        )
        self.outcome_regressor = nn.Sequential(
            nn.Linear(self.num_hidden_units + 1, self.num_hidden_units),
            nn.ELU(),
            nn.Linear(self.num_hidden_units, self.num_outcomes),
        )

    def make_init_hidden_state(
        self,
        init_representation: jty.Float[torch.Tensor, 'b h'] | None = None,
    ):
        if init_representation is None:
            return None

        b, _ = init_representation.shape
        device = get_device(init_representation)
        return (
            ei.rearrange(init_representation, 'b h -> 1 b h').to(device),
            torch.zeros(1, b, self.num_hidden_units).to(device),
        )

    def forward(
        self,
        covariates_series: jty.Float[torch.Tensor, 'b t c'],
        used_treatment_series: jty.Integer[torch.Tensor, 'b t'],
        next_treatment: jty.Integer[torch.Tensor, 'b'],
        init_representation: jty.Float[torch.Tensor, 'b h'] | None = None,
    ):
        b, t, _ = covariates_series.shape
        c, h = self.num_covariates, self.num_hidden_units

        expect_shape(covariates_series, 'b t c', b=b, t=t, c=c)
        expect_shape(used_treatment_series, 'b t', b=b, t=t)
        if next_treatment is not None:
            expect_shape(next_treatment, 'b', b=b)
        if init_representation is not None:
            expect_shape(init_representation, 'b h', b=b, h=h)

        used_treatment_series = used_treatment_series.float()
        next_treatment = next_treatment.float()

        lstm_inputs = ei.pack(
            [covariates_series, used_treatment_series], pattern='b t *'
        )[0]

        init_hidden_state = self.make_init_hidden_state(init_representation)
        next_hidden_state_series, _ = self.lstm(lstm_inputs, init_hidden_state)
        next_representation_series = self.representor(next_hidden_state_series)

        used_treatment_logits_series = self.treatment_classifier(
            next_representation_series
        )

        next_treatment_series = ei.pack(
            [used_treatment_series[:, 1:], next_treatment], pattern='b *'
        )[0]

        regressor_inputs = ei.pack(
            [next_representation_series, next_treatment_series], pattern='b t *'
        )[0]
        next_outcome_series = self.outcome_regressor(regressor_inputs)

        return self.Output(
            used_treatment_logits_series=used_treatment_logits_series,
            estimated_outcomes_series=next_outcome_series,
            next_representation_series=next_representation_series,
        )

    # @torch.no_grad()
    # def generate_next(
    #     self,
    #     covariates_series: jty.Float[torch.Tensor, 'b t c'],
    #     used_treatment_series: jty.Float[torch.Tensor, 'b t'],
    #     next_treatment: jty.Float[torch.Tensor, 'b'],
    #     init_representation: jty.Float[torch.Tensor, 'b h'],
    # ):
    #     output = self(
    #         covariates_series=covariates_series,
    #         used_treatment_series=used_treatment_series,
    #         next_treatment=next_treatment,
    #         init_representation=init_representation,
    #     )
    #     return self.ForcastOutput(
    #         outcome=ei.rearrange(
    #             output.estimated_outcome_series[:, -1, :], 'b 1 o -> b -> o'
    #         ),
    #         representation=ei.rearrange(
    #             output.next_representaion_series[:, -1, :], 'b 1 h -> b h'
    #         ),
    #     )


class DefaultTumorSimulationDataset(data_utils.Dataset):
    def __init__(
        self,
        num_patients: int,
        num_time_steps: int,
        chemo_gamma: float = 5.0,
        radio_gamma: float = 5.0,
        seed: Seed = 0,
    ):
        self.num_patients = num_patients
        self.num_time_steps = num_time_steps
        self.treatment_plan = DefaultTreatmentPlan(
            chemo_gamma=chemo_gamma, radio_gamma=radio_gamma
        )
        self.patient_profiles, self.results = run_simulation(
            num_patients=num_patients,
            num_time_steps=num_time_steps,
            treatment_plan=self.treatment_plan,
            seed=seed,
        )

    def __len__(self):
        return self.num_patients

    def __getitem__(self, idx):
        return self.patient_profiles[idx], self.results[idx]

    def plot(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

        for idx, (_, result) in enumerate(self):
            tumor_model, treatment_plan = result.tumor_model, result.treatment_plan
            tumor_volumes = np.asarray(tumor_model.state.volumes)
            chemo_usages = np.asarray([False] + list(treatment_plan.state.chemo_usages))
            radio_usages = np.asarray([False] + list(treatment_plan.state.radio_usages))

            ax.plot(tumor_volumes)

            arange = np.arange(len(tumor_volumes))
            chemo_label = 'Chemo' if idx == 0 else None
            radio_label = 'Radio' if idx == 0 else None
            ax.scatter(
                arange[chemo_usages],
                tumor_volumes[chemo_usages],
                color='red',
                label=chemo_label,
                marker='+',
            )
            ax.scatter(
                arange[radio_usages],
                tumor_volumes[radio_usages],
                color='blue',
                label=radio_label,
                marker='x',
            )

        ax.set_title('Tumor Volume')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Volume')
        ax.legend()
        plt.show()


class BaseCRNDataset(data_utils.Dataset):
    @beartype
    class Input(TypedDict):
        covariates_series: jty.Float[torch.Tensor, 't c']
        used_treatment_series: jty.Integer[torch.Tensor, 't']
        next_treatment: jty.Integer[torch.Tensor, '']

    @beartype
    class Target(TypedDict):
        used_treatment_series: jty.Integer[torch.Tensor, 't']
        outcomes_series: jty.Float[torch.Tensor, 't o']
        mask_series: jty.Bool[torch.Tensor, 't']

    def __init__(self, dataset: DefaultTumorSimulationDataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.num_patients


class CRNLoss(nn.Module):
    def forward(self, y_pred, y_true):
        used_treatment_logits_series = y_pred['used_treatment_logits_series']
        estimated_outcomes_series = y_pred['estimated_outcomes_series']

        treatment_label_series = y_true['used_treatment_series']
        outcomes_series = y_true['outcomes_series']
        mask_series = y_true['mask_series']

        flattened_logits = ei.rearrange(
            used_treatment_logits_series, pattern='b t n -> (b t) n'
        )
        flattened_label = ei.rearrange(treatment_label_series, pattern='b t -> (b t)')
        flattened_is_valid = ei.rearrange(mask_series, pattern='b t -> (b t)')
        treatment_loss = fn.cross_entropy(
            flattened_logits[flattened_is_valid],
            flattened_label[flattened_is_valid],
        )

        outcome_loss = fn.mse_loss(
            estimated_outcomes_series[mask_series], outcomes_series[mask_series]
        )
        return outcome_loss, treatment_loss


class CRNTrainer(NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        outcome_loss, treatment_loss = super().get_loss(y_pred, y_true, X, training)

        num_train_batches = sum(
            len(epoch_history['batches']) for epoch_history in self.history
        )

        epoch_ratio = torch.tensor(len(self.history) / self.max_epochs).float()

        lambda_ = 2 / (1 + torch.exp(-10 * epoch_ratio)) - 1
        loss = outcome_loss - lambda_ * treatment_loss

        prefix = 'train' if training else 'valid'

        self.history.record_batch(prefix + '_treatment_loss', treatment_loss.item())
        self.history.record_batch(prefix + '_outcome_loss', outcome_loss.item())
        if training:
            self.history.record_batch(prefix + '_lambda', lambda_.item())
        self.history.record_batch(prefix + '_steps', num_train_batches)

        return loss


def process_crn_simulation_result(
    result: SimulationResult, patient_params: PatientParams
):
    tumor_model, treatment_plan = result.tumor_model, result.treatment_plan

    volume_series = np.asarray(tumor_model.state.volumes)
    patient_group_series = np.full_like(volume_series, patient_params.group)
    patient_status_series = np.asarray(tumor_model.state.patient_statuses)

    next_chemo_usage_series = np.asarray(treatment_plan.state.chemo_usages)
    next_radio_usage_series = np.asarray(treatment_plan.state.radio_usages)

    usage_series = ei.pack(
        [
            ei.pack([np.array(0), next_chemo_usage_series], '*')[0],
            ei.pack([np.array(0), next_radio_usage_series], '*')[0],
        ],
        pattern='num_time_steps *',
    )[0]

    treatment_series = np.select(
        condlist=[
            ei.reduce(usage_series == [0, 0], 't d -> t', np.all),
            ei.reduce(usage_series == [1, 0], 't d -> t', np.all),
            ei.reduce(usage_series == [0, 1], 't d -> t', np.all),
            ei.reduce(usage_series == [1, 1], 't d -> t', np.all),
        ],
        choicelist=[0, 1, 2, 3],
        default=np.nan,
    )

    assert np.all(~np.isnan(treatment_series))
    assert len(volume_series) == len(patient_status_series)

    mask_series = patient_status_series == PatientStatuses.TUMOR
    masked_volume_series = np.where(mask_series, volume_series, 0)
    masked_patient_group_series = np.where(mask_series, patient_group_series, 0)
    masked_treatment_series = np.where(mask_series, treatment_series, 0)

    return (
        masked_volume_series,
        masked_patient_group_series,
        masked_treatment_series,
        mask_series,
    )


def process_crn_inputs_and_labels(
    full_volume_series: list[jty.Float[np.ndarray, 't']],
    full_patient_group_series: list[jty.Float[np.ndarray, 't']],
    full_treatment_series: list[jty.Float[np.ndarray, 't']],
    full_mask_series: list[jty.Bool[np.ndarray, 't']],
):
    all_patient_group_series = ei.pack(full_patient_group_series, '* x')[0]
    all_volume_series = ei.pack(full_volume_series, '* x')[0]
    all_treatment_series = ei.pack(full_treatment_series, '* x')[0]
    all_mask_series = ei.pack(full_mask_series, '* x')[0]

    all_covariates_series = ei.pack(
        [all_volume_series[:, :-1], all_patient_group_series[:, :-1]], 'b t *'
    )[0].astype(np.float32)

    all_outcomes_series = ei.rearrange(
        all_volume_series[:, 1:], pattern='b t -> b t 1'
    ).astype(np.float32)

    all_used_treatment_series = all_treatment_series[:, :-1].astype(np.int64)
    all_next_treatment = all_treatment_series[:, -1].astype(np.int64)
    all_mask_series = all_mask_series[:, :-1]

    return (
        all_covariates_series,
        all_outcomes_series,
        all_used_treatment_series,
        all_next_treatment,
        all_mask_series,
    )


# class CRNEncoderDataset(BaseCRNDataset):
class CRNEncoderDataset(data_utils.Dataset):
    @beartype
    class Input(TypedDict):
        covariates_series: jty.Float[torch.Tensor, 't c']
        used_treatment_series: jty.Integer[torch.Tensor, 't']
        next_treatment: jty.Integer[torch.Tensor, '']

    @beartype
    class Target(TypedDict):
        used_treatment_series: jty.Integer[torch.Tensor, 't']
        outcomes_series: jty.Float[torch.Tensor, 't o']
        mask_series: jty.Bool[torch.Tensor, 't']

    SeriesStats = namedtuple('SeriesStats', ['mean', 'std'])

    def __init__(
        self, patient_profiles: list[PatientParams], results: list[SimulationResult]
    ):
        self.patient_profiles = patient_profiles
        self.results = results

        full_patient_group_series = []
        full_volume_series = []
        full_treatment_series = []
        full_mask_series = []

        for patient_params, result in tqdm(
            zip(patient_profiles, results),
            desc='Processing',
            total=len(patient_profiles),
        ):
            (
                masked_volume_series,
                masked_patient_group_series,
                masked_treatment_series,
                mask_series,
            ) = process_crn_simulation_result(result, patient_params)

            full_patient_group_series.append(masked_patient_group_series)
            full_volume_series.append(masked_volume_series)
            full_treatment_series.append(masked_treatment_series)
            full_mask_series.append(mask_series)

        (
            self.all_covariates_series,
            self.all_outcomes_series,
            self.all_used_treatment_series,
            self.all_next_treatment,
            self.all_mask_series,
        ) = process_crn_inputs_and_labels(
            full_volume_series,
            full_patient_group_series,
            full_treatment_series,
            full_mask_series,
        )

        def calc_stats(all_series):
            masked_series = np.ma.array(
                all_series,
                mask=~ei.repeat(
                    self.all_mask_series, 'b t -> b t x', x=all_series.shape[-1]
                ),
            )
            return self.SeriesStats(
                mean=ei.reduce(masked_series, 'b t x -> x', np.ma.mean).astype(
                    np.float32
                ),
                std=ei.reduce(masked_series, 'b t x -> x', np.ma.std).astype(
                    np.float32
                ),
            )

        self.covariates_stats = calc_stats(self.all_covariates_series)
        self.outcomes_stats = calc_stats(self.all_outcomes_series)

    def scale(self, series, stats):
        return (series - stats.mean) / stats.std

    def unscale(self, series, stats):
        return series * stats.std + stats.mean

    def __getitem__(self, idx: int):
        return self.Input(
            covariates_series=self.scale(
                self.all_covariates_series[idx], self.covariates_stats
            ),
            used_treatment_series=self.all_used_treatment_series[idx],
            next_treatment=self.all_next_treatment[idx],
        ), self.Target(
            used_treatment_series=self.all_used_treatment_series[idx],
            outcomes_series=self.scale(
                self.all_outcomes_series[idx], self.outcomes_stats
            ),
            mask_series=self.all_mask_series[idx],
        )

    def __len__(self):
        return len(self.all_covariates_series)


class CRNDecoderDataset(CRNEncoderDataset):
    @beartype
    class Input(CRNEncoderDataset.Input):
        init_representation: jty.Float[torch.Tensor, 'b h']

    def __init__(
        self,
        dataset: DefaultTumorSimulationDataset,
        max_time_steps: int,
        encoder: CRNModule,
    ):
        super().__init__(dataset)

        self.max_time_steps = max_time_steps
        self.encoder = encoder

        full_covariates_series, full_outcomes_series = [], []
        full_used_treatment_series, full_next_treatment = [], []
        full_init_representation = []
        full_mask_series = []

        device = get_device(self.encoder)

        def as_input(input):
            return ei.rearrange(torch.as_tensor(input), '... -> 1 ...').to(device)

        for (
            covariates_series,
            outcomes_series,
            used_treatment_series,
            next_treatment,
            mask_series,
        ) in tqdm(
            zip(
                self.all_covariates_series,
                self.all_outcomes_series,
                self.all_used_treatment_series,
                self.all_next_treatment,
                self.all_mask_series,
            ),
            desc='Processing',
            total=len(self.all_covariates_series),
        ):
            num_actual_time_steps = np.sum(mask_series)
            covariates_series = covariates_series[:num_actual_time_steps]
            outcomes_series = outcomes_series[:num_actual_time_steps]
            used_treatment_series = used_treatment_series[:num_actual_time_steps]
            mask_series = mask_series[:num_actual_time_steps]

            considered_time_steps = num_actual_time_steps - self.max_time_steps
            for t in range(considered_time_steps):
                end_t = t + self.max_time_steps

                full_covariates_series.append(covariates_series[t:end_t])
                full_outcomes_series.append(outcomes_series[t:end_t])
                full_used_treatment_series.append(used_treatment_series[t:end_t])
                if t == considered_time_steps - 1:
                    full_next_treatment.append(next_treatment)
                else:
                    full_next_treatment.append(used_treatment_series[end_t])
                full_mask_series.append(mask_series[t:end_t])

                self.encoder.eval()
                with torch.no_grad():
                    output = self.encoder(
                        covariates_series=as_input(full_covariates_series[-1]),
                        used_treatment_series=as_input(full_used_treatment_series[-1]),
                        next_treatment=as_input(full_next_treatment[-1]),
                    )
                    full_init_representation.append(
                        output['next_representation_series'][:, -1, :]
                    )

        self.all_covariates_series = ei.pack(full_covariates_series, '* t c')[0]
        self.all_outcomes_series = ei.pack(full_outcomes_series, '* t o')[0]
        self.all_used_treatment_series = ei.pack(full_used_treatment_series, '* t')[0]
        self.all_next_treatment = np.asarray(full_next_treatment)
        self.all_init_representation = ei.pack(full_init_representation, '* h')[0]
        self.all_mask_series = ei.pack(full_mask_series, '* t')[0]

    def __getitem__(self, idx: int):
        return self.Input(
            covariates_series=self.scale(
                self.all_covariates_series[idx], self.covariates_stats
            ),
            used_treatment_series=self.all_used_treatment_series[idx],
            next_treatment=self.all_next_treatment[idx],
            init_representation=self.all_init_representation[idx],
        ), self.Target(
            used_treatment_series=self.all_used_treatment_series[idx],
            outcomes_series=self.scale(
                self.all_outcomes_series[idx], self.outcomes_stats
            ),
            mask_series=self.all_mask_series[idx],
        )
