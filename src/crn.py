from typing import NamedTuple

import einops as ei
import jaxtyping as jty
import lightning as ltn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.utils.data as data_utils
from beartype import beartype
from loguru import logger
from torchmetrics.functional import mean_absolute_error, mean_squared_error

from src.simulations import non_small_lung_cancer as nslc
from src.utils.misc import tqdm_print, zip_tqdm
from src.utils.modules.gradient_reversal import GradientReversal
from src.utils.modules.misc import TakeElement, UnpackInput
from src.utils.tensor import expect, get_device, to_device, to_tensor

logger.remove()
logger.add(lambda msg: tqdm_print(msg), colorize=True)


@beartype
def make_treatments_seq(
    next_treatments_seqs: jty.Int[torch.Tensor, '*b t a'],
    init_treatments: jty.Int[torch.Tensor, '*b a'] | None = None,
):
    *b, _, a = next_treatments_seqs.shape
    if len(b) > 1:
        raise ValueError('At most 1 batch dimension is allowed.')

    if init_treatments is None:
        device = get_device(next_treatments_seqs)
        init_treatments = torch.zeros((*b, 1, a), device=device, dtype=int)
    else:
        init_treatments = ei.rearrange(init_treatments, 'b a -> b 1 a')

    pattern = '* a' if len(b) == 0 else 'b * a'
    return ei.pack([init_treatments, next_treatments_seqs[:, :-1, :]], pattern)[0]


@beartype
def make_mask(max_time_step: int, end_time_steps: jty.Int[torch.Tensor, 'b']):
    time_steps = torch.arange(max_time_step, device=get_device(end_time_steps))
    mask_seqs = time_steps < ei.repeat(end_time_steps, 'b -> b t', t=max_time_step)
    return mask_seqs


class CRNModule(nn.Module):
    @beartype
    class Output(NamedTuple):
        treatment_logits_seqs: jty.Float[torch.Tensor, '*b t n']
        next_outcomes_seqs: jty.Float[torch.Tensor, '*b t o']
        next_representation_seqs: jty.Float[torch.Tensor, '*b t h']

    @beartype
    def __init__(
        self,
        num_outcomes: int,
        num_treatments: int,
        num_covariates: int = 0,
        num_static_covariates: int = 0,
        num_hidden_units: int = 10,
        dropout_prob: float = 0.2,
        num_hidden_layers: int = 3,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.num_outcomes, self.num_treatments = num_outcomes, num_treatments
        self.num_covariates, self.num_static_covariates = num_covariates, num_static_covariates

        self.num_hidden_units = num_hidden_units

        self.num_hidden_layers = num_hidden_layers
        self.dropout_prob = dropout_prob
        self.alpha = alpha

        num_total_covariates = self.num_covariates + self.num_static_covariates

        self.lstm = nn.LSTM(
            input_size=self.num_outcomes + self.num_treatments + num_total_covariates,
            hidden_size=self.num_hidden_units,
            dropout=self.dropout_prob,
            num_layers=self.num_hidden_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.projector = nn.Linear(num_hidden_units, num_hidden_units)
        self.representor = nn.Sequential(
            UnpackInput(self.lstm), TakeElement(0), nn.ELU(), self.projector, nn.ELU()
        )

        self.treatment_classifier = nn.Sequential(
            GradientReversal(self.alpha),
            nn.Linear(self.num_hidden_units, self.num_hidden_units),
            nn.ELU(),
            nn.Linear(self.num_hidden_units, self.num_treatments),
        )
        self.outcome_regressor = nn.Sequential(
            nn.Linear(self.num_hidden_units + self.num_treatments, self.num_hidden_units),
            nn.ELU(),
            nn.Linear(self.num_hidden_units, self.num_outcomes),
        )

    def make_init_hidden_state(
        self,
        init_representation: jty.Float[torch.Tensor, 'b h'] | None = None,
    ):
        if init_representation is None:
            return None

        b, h = init_representation.shape
        l = self.num_hidden_layers  # noqa: E741
        device = get_device(init_representation)
        return (
            ei.repeat(init_representation, 'b h -> l b h', l=l).to(device).contiguous(),
            torch.zeros(self.num_hidden_layers, b, h).to(device),
        )

    def forward(
        self,
        outcomes_seqs: jty.Float[torch.Tensor, 'b t o'],
        next_treatments_seqs: jty.Int[torch.Tensor, 'b t a'],
        # covariates_seqs: jty.Float[torch.Tensor, 'b t c'] | None = None,
        static_covariates: jty.Float[torch.Tensor, 'b s'] | None = None,
        init_treatments: jty.Float[torch.Tensor, 'b a'] | None = None,
        init_representations: jty.Float[torch.Tensor, 'b h'] | None = None,
    ):
        b, t, _ = outcomes_seqs.shape

        o, a = self.num_outcomes, self.num_treatments
        s = self.num_static_covariates
        # c, s = self.num_covariates, self.num_static_covariates
        h = self.num_hidden_units

        expect(outcomes_seqs, 'b t o', b=b, t=t, o=o)
        expect(next_treatments_seqs, 'b t a', b=b, t=t, a=a)
        # if covariates_seqs is not None:
        #     expect(covariates_seqs, 'b t c', b=b, t=t, c=c)
        if static_covariates is not None:
            expect(static_covariates, 'b s', b=b, s=s)
        if init_treatments is not None:
            expect(init_treatments, 'b a', b=b, a=a)
        if init_representations is not None:
            expect(init_representations, 'b h', b=b, h=h)

        treatments_seq = make_treatments_seq(next_treatments_seqs, init_treatments)
        lstm_inputs = ei.pack([outcomes_seqs, treatments_seq], 'b t *')[0]
        # if covariates_seqs is not None:
        #     lstm_inputs = ei.pack([lstm_inputs, covariates_seqs], 'b t *')[0]
        if static_covariates is not None:
            static_covariate_seq = ei.repeat(static_covariates, 'b s -> b t s', t=t)
            lstm_inputs = ei.pack([lstm_inputs, static_covariate_seq], 'b t *')[0]

        init_hidden_states = self.make_init_hidden_state(init_representations)
        next_representation_seqs = self.representor((lstm_inputs, init_hidden_states))
        treatment_logits_seqs = self.treatment_classifier(next_representation_seqs)

        regressor_inputs = ei.pack([next_representation_seqs, next_treatments_seqs], 'b t *')[0]
        next_outcome_seqs = self.outcome_regressor(regressor_inputs)

        return self.Output(
            treatment_logits_seqs=treatment_logits_seqs,
            next_outcomes_seqs=next_outcome_seqs,
            next_representation_seqs=next_representation_seqs,
        )


class CRNBaseDataset(data_utils.Dataset):
    @beartype
    class EncoderBatch(NamedTuple):
        outcome_seqs: jty.Float[torch.Tensor, '*b t o']
        next_outcome_seqs: jty.Float[torch.Tensor, '*b t o']
        next_treatment_seqs: jty.Int[torch.Tensor, '*b t a']
        static_covariates: jty.Float[torch.Tensor, '*b s']
        end_time_steps: jty.Int[torch.Tensor, '*b']

        @property
        def treatment_seqs(self):
            return make_treatments_seq(self.next_treatment_seqs, None)

    @beartype
    class DecoderBatch(NamedTuple):
        outcome_seqs: jty.Float[torch.Tensor, '*b t o']
        next_outcome_seqs: jty.Float[torch.Tensor, '*b t o']
        next_treatment_seqs: jty.Int[torch.Tensor, '*b t a']
        static_covariates: jty.Float[torch.Tensor, '*b s']
        end_time_steps: jty.Int[torch.Tensor, '*b']
        init_treatments: jty.Int[torch.Tensor, '*b a']
        init_representations: jty.Float[torch.Tensor, '*b h']

        @property
        def treatment_seqs(self):
            return make_treatments_seq(self.next_treatment_seqs, self.init_treatments)

    Batch = EncoderBatch | DecoderBatch


@beartype
class SeqStats(NamedTuple):
    mean: jty.Float[torch.Tensor, '*']
    std: jty.Float[torch.Tensor, '*']


DEFAULT_STATS = SeqStats(mean=torch.tensor(0.0), std=torch.tensor(1.0))


class CRNLightningModule(ltn.LightningModule):
    def __init__(
        self,
        outcomes_stats: SeqStats = DEFAULT_STATS,
        static_covariates_stats: SeqStats = DEFAULT_STATS,
        max_outcome_value: float = 1.0,
        **crn_kwargs,
    ):
        super().__init__()
        self.outcomes_stats = outcomes_stats
        self.static_covariates_stats = static_covariates_stats
        self.max_outcome_value = max_outcome_value
        self.save_hyperparameters(crn_kwargs)
        self.module = CRNModule(**crn_kwargs)

    def forward(self, batch: CRNBaseDataset.Batch):
        return self.module(
            outcomes_seqs=self.scale(batch.outcome_seqs, self.outcomes_stats),
            next_treatments_seqs=batch.next_treatment_seqs,
            static_covariates=self.scale(batch.static_covariates, self.static_covariates_stats),
            init_treatments=getattr(batch, 'init_treatments', None),
            init_representations=getattr(batch, 'init_representations', None),
        )

    def scale(self, seq, stats: SeqStats):
        device = get_device(seq)
        return (seq - to_device(stats.mean, device=device)) / to_device(stats.std, device=device)

    def unscale(self, seq, stats: SeqStats):
        device = get_device(seq)
        return seq * to_device(stats.std, device=device) + to_device(stats.mean, device=device)

    def compute_loss(self, output: CRNModule.Output, batch: CRNBaseDataset.Batch, prefix: str):
        max_time_step = batch.outcome_seqs.shape[1]
        mask_seqs = make_mask(max_time_step=max_time_step, end_time_steps=batch.end_time_steps)

        next_outcome_seqs = output.next_outcomes_seqs[mask_seqs]
        target_next_outcome_seqs = self.scale(
            batch.next_outcome_seqs[mask_seqs], self.outcomes_stats
        )

        treatment_logits_seqs = output.treatment_logits_seqs[mask_seqs]
        target_treatment_seqs = batch.treatment_seqs[mask_seqs]

        outcome_loss = fn.mse_loss(next_outcome_seqs, target_next_outcome_seqs)
        treatment_loss = fn.binary_cross_entropy_with_logits(
            treatment_logits_seqs, target_treatment_seqs.float()
        )

        epoch_ratio = torch.tensor(self.current_epoch / self.trainer.max_epochs).float()
        lambda_ = 2 / (1 + torch.exp(-10 * epoch_ratio)) - 1

        loss = outcome_loss - lambda_ * treatment_loss

        with torch.no_grad():
            unscaled_next_outcome_seqs = self.unscale(next_outcome_seqs, self.outcomes_stats)
            unscaled_target_next_outcome_seqs = self.unscale(
                target_next_outcome_seqs, self.outcomes_stats
            )
            rmse = mean_squared_error(
                unscaled_next_outcome_seqs, unscaled_target_next_outcome_seqs, squared=False
            )
            nrmse = rmse / self.max_outcome_value * 100

            mae = mean_absolute_error(unscaled_next_outcome_seqs, unscaled_target_next_outcome_seqs)
            nmae = mae / self.max_outcome_value * 100

        self.log(f'{prefix}/loss', loss)
        self.log(f'{prefix}/loss/outcome', outcome_loss)
        self.log(f'{prefix}/loss/treatment', treatment_loss)
        self.log(f'{prefix}/metric/nrmse', nrmse)
        self.log(f'{prefix}/metric/nmae', nmae)
        self.log(f'{prefix}/other/lambda', lambda_)

        return loss

    def training_step(self, batch: CRNBaseDataset.Batch, batch_idx: int):
        output = self(batch)
        return self.compute_loss(output, batch, prefix='train')

    def validation_step(self, batch: CRNBaseDataset.Batch, batch_idx: int):
        output = self(batch)
        return self.compute_loss(output, batch, prefix='val')

    def test_step(self, batch: CRNBaseDataset.Batch, batch_idx: int):
        output = self(batch)
        return self.compute_loss(output, batch, prefix='test')

    def regress_outcome(self, batch: CRNBaseDataset.Batch):
        output = self(batch)
        return self.unscale(output.next_outcomes_seqs, self.outcomes_stats)

    def configure_optimizers(self):
        return optim.Adam(self.module.parameters(), lr=1e-4)

    def make_metrics_string(self, metrics: dict[str, torch.Tensor]):
        return ' | '.join(
            f'{key}={value:6.2f}' for key, value in metrics.items() if 'other' not in key
        )

    def on_validation_epoch_end(self):
        metrics_str = self.make_metrics_string(self.trainer.callback_metrics)
        epoch_str = str(self.current_epoch).zfill(len(str(self.trainer.max_epochs)))
        logger.info(f'E{epoch_str} - {metrics_str}')


def extract_result(result: nslc.SimulationResult, patient_params: nslc.PatientParams):
    tumor_model, treatment_plan = result.tumor_model, result.treatment_plan

    patient_type = patient_params.patient_type
    volume_seq = np.asarray(tumor_model.state.volumes)
    patient_status_seq = np.asarray(tumor_model.state.patient_statuses)

    next_chemo_usage_seq = np.asarray(treatment_plan.state.chemo_usages)
    next_radio_usage_seq = np.asarray(treatment_plan.state.radio_usages)

    next_usages_seq = ei.pack(
        [
            ei.pack([np.array(0), next_chemo_usage_seq], '*')[0],
            ei.pack([np.array(0), next_radio_usage_seq], '*')[0],
        ],
        pattern='t *',
    )[0]

    end_time_steps = np.flatnonzero(patient_status_seq != nslc.PatientStatuses.TUMOR)
    end_time_step = end_time_steps[0] if len(end_time_steps) != 0 else len(volume_seq)

    return volume_seq, next_usages_seq, patient_type, end_time_step


def process_data(
    volume_seqs: list[jty.Float[np.ndarray, 't']],
    usages_seqs: list[jty.Float[np.ndarray, 't a']],
    patient_types: list[nslc.PatientTypes],
    end_time_steps: list[int],
):
    outcomes_seqs = ei.rearrange(ei.pack(volume_seqs, '* t')[0], 'b t -> b t 1')
    treatments_seqs = ei.pack(usages_seqs, '* t a')[0]
    static_covariates_grp = ei.rearrange(np.asarray(patient_types), 'b -> b 1')

    return to_tensor(
        outcomes_seqs.astype(np.float32),
        treatments_seqs.astype(np.int64),
        static_covariates_grp.astype(np.float32),
        np.asarray(end_time_steps).astype(np.int64),
    )


def calc_all_stats(
    outcomes_seqs: jty.Float[torch.Tensor, '*b t o'],
    static_covariates_grp: jty.Float[torch.Tensor, '*b s'],
    end_time_steps: jty.Int[torch.Tensor, '*b'],
):
    def calc_stats(data):
        return SeqStats(
            mean=to_tensor(ei.reduce(ei.asnumpy(data), '... x -> x', reduction=np.nanmean)),
            std=to_tensor(ei.reduce(ei.asnumpy(data), '... x -> x', reduction=np.nanstd)),
        )

    mask_seqs = make_mask(outcomes_seqs.shape[1], end_time_steps=end_time_steps)
    masked_outcomes_seqs = torch.where(
        ei.rearrange(mask_seqs, 'b t -> b t 1'), outcomes_seqs, torch.nan
    )
    outcomes_stats = calc_stats(masked_outcomes_seqs)
    static_covariate_stats = calc_stats(static_covariates_grp)

    return outcomes_stats, static_covariate_stats


class CRNEncoderDataset(CRNBaseDataset):
    def __init__(
        self, patient_profiles: list[nslc.PatientParams], results: list[nslc.SimulationResult]
    ):
        self.patient_profiles, self.results = patient_profiles, results

        volume_seqs, next_usages_seqs, patient_types, end_time_steps = [], [], [], []
        for patient_params, result in zip_tqdm(patient_profiles, results, desc='Processing'):
            volume_seq, next_usages_seq, patient_type, end_time_step = extract_result(
                result, patient_params
            )

            volume_seqs.append(volume_seq)
            next_usages_seqs.append(next_usages_seq)
            patient_types.append(patient_type)
            end_time_steps.append(end_time_step)

        (
            self.outcomes_seqs,
            self.next_treatments_seqs,
            self.static_covariates_grp,
            self.end_time_steps,
        ) = process_data(volume_seqs, next_usages_seqs, patient_types, end_time_steps)

    @property
    def num_outcomes(self):
        return self.outcomes_seqs.shape[2]

    @property
    def num_treatments(self):
        return self.next_treatments_seqs.shape[2]

    @property
    def num_static_covariates(self):
        return self.static_covariates_grp.shape[1]

    def __getitem__(self, idx: int):
        return self.EncoderBatch(
            outcome_seqs=self.outcomes_seqs[idx][:-1],
            next_outcome_seqs=self.outcomes_seqs[idx][1:],
            next_treatment_seqs=self.next_treatments_seqs[idx][:-1],
            static_covariates=self.static_covariates_grp[idx],
            end_time_steps=self.end_time_steps[idx],
        )

    def __len__(self):
        return len(self.outcomes_seqs)


class CRNDecoderDataset(CRNEncoderDataset):
    def __init__(
        self,
        patient_profiles: list[nslc.PatientParams],
        results: list[nslc.SimulationResult],
        max_time_steps: int,
        encoder: CRNModule,
    ):
        super().__init__(patient_profiles, results)

        self.max_time_steps = max_time_steps

        outcomes_seqs, next_treatments_seqs, static_covariates_grp = [], [], []
        init_treatments_grp, init_representation_grp = [], []

        device = get_device(encoder)
        encoder.eval()

        def as_input(data):
            return to_device(ei.rearrange(data, '... -> 1 ...'), device=device)

        for outcomes_seq, next_treatments_seq, static_covariates, end_time_step in zip_tqdm(
            self.outcomes_seqs,
            self.next_treatments_seqs,
            self.static_covariates_grp,
            self.end_time_steps,
            desc='Processing',
        ):
            outcomes_seq = outcomes_seq[:end_time_step]
            next_treatments_seq = next_treatments_seq[:end_time_step]

            for t in range(end_time_step - self.max_time_steps):
                end_t = t + self.max_time_steps

                outcomes_seqs.append(outcomes_seq[t:end_t])
                next_treatments_seqs.append(next_treatments_seq[t:end_t])
                static_covariates_grp.append(static_covariates)

                if t == 0:
                    num_treatments = next_treatments_seq.shape[1]
                    init_treatments_grp.append(torch.zeros(num_treatments).long())
                else:
                    init_treatments_grp.append(next_treatments_seq[t - 1])

            init_representation_grp.append(torch.zeros(encoder.num_hidden_units).float())
            if end_time_step > 1:
                with torch.no_grad():
                    output = encoder(
                        outcomes_seqs=as_input(outcomes_seq[: end_time_step - 1]),
                        next_treatments_seqs=as_input(next_treatments_seq[: end_time_step - 1]),
                        static_covariates=as_input(static_covariates),
                    )
                init_representation_grp.extend(output.next_representation_seqs[0].cpu())

        self.outcomes_seqs = ei.pack(outcomes_seqs, '* t o')[0]
        self.next_treatments_seqs = ei.pack(next_treatments_seqs, '* t a')[0]
        self.static_covariates_grp = ei.pack(static_covariates_grp, '* s')[0]
        self.init_treatments_grp = ei.pack(init_treatments_grp, '* a')[0]
        self.init_representation_grp = ei.pack(init_representation_grp, '* h')[0]

    def __getitem__(self, idx: int):
        return self.DecoderBatch(
            outcome_seqs=self.outcomes_seqs[idx],
            next_outcome_seqs=self.outcomes_seqs[idx],
            next_treatment_seqs=self.next_treatments_seqs[idx],
            static_covariates=self.static_covariates_grp[idx],
            end_time_steps=torch.tensor(self.max_time_steps),
            init_treatments=self.init_treatments_grp[idx],
            init_representations=self.init_representation_grp[idx],
        )
