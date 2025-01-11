import os
from datetime import datetime

import einops as ei
import lightning as ltn
import numpy as np
import torch
import torch.utils.data as data_utils
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.crn import (
    CRNBaseDataset,
    # CRNDecoderDataset,
    CRNEncoderDataset,
    CRNLightningModule,
    calc_all_stats,
    extract_result,
    make_mask,
    process_data,
)
from src.simulations.non_small_lung_cancer import (
    DEATH_CONDITION_VOLUME,
    DefaultTreatmentPlan,
    run_simulation,
    run_simulation_with_counterfactuals,
)
from src.utils.misc import zip_tqdm

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

seed_everything(0, workers=True)

num_patients = 1000
num_time_steps = 60
chemo_gamma = 0.0
radio_gamma = 0.0
max_time_steps = 5

num_epochs = 50
batch_size = 128

encoder_batch_size = batch_size
decoder_batch_size = batch_size * 16

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f'{timestamp}_c{chemo_gamma}_r{radio_gamma}'
encoder_run_name = run_name
decoder_run_name = f'{run_name}_decoder'

treatment_plan = DefaultTreatmentPlan(
    chemo_gamma=chemo_gamma,
    radio_gamma=radio_gamma,
)

train_patient_profiles, train_results = run_simulation(
    num_patients=num_patients * 10,
    num_time_steps=num_time_steps,
    treatment_plan=treatment_plan,
    seed=1,
)

validation_patient_profiles, validation_results = run_simulation(
    num_patients=num_patients,
    num_time_steps=num_time_steps,
    treatment_plan=treatment_plan,
    seed=10,
)

test_patient_profiles, test_results = run_simulation_with_counterfactuals(
    num_patients=num_patients,
    num_time_steps=num_time_steps,
    treatment_plan=treatment_plan,
    seed=100,
)

train_encoder_dataset = CRNEncoderDataset(
    patient_profiles=train_patient_profiles,
    results=train_results,
)
outcomes_stats, static_covariates_stats = calc_all_stats(
    train_encoder_dataset.outcomes_seqs,
    train_encoder_dataset.static_covariates_grp,
    train_encoder_dataset.end_time_steps,
)
validation_encoder_dataset = CRNEncoderDataset(
    patient_profiles=validation_patient_profiles, results=validation_results
)
test_encoder_dataset = CRNEncoderDataset(
    patient_profiles=test_patient_profiles, results=test_results
)

encoder_trainer = ltn.Trainer(
    max_epochs=num_epochs,
    deterministic=True,
    logger=[
        CSVLogger(save_dir='logs/crn', name=encoder_run_name),
        WandbLogger(
            project='crn',
            save_dir='logs/crn',
            name=encoder_run_name,
        ),
    ],
)
encoder_trainer.fit(
    model=CRNLightningModule(
        outcomes_stats=outcomes_stats,
        static_covariates_stats=static_covariates_stats,
        max_outcome_value=DEATH_CONDITION_VOLUME,
        num_outcomes=train_encoder_dataset.num_outcomes,
        num_treatments=train_encoder_dataset.num_treatments,
        num_static_covariates=train_encoder_dataset.num_static_covariates,
    ),
    train_dataloaders=data_utils.DataLoader(
        train_encoder_dataset, batch_size=encoder_batch_size, shuffle=True
    ),
    val_dataloaders=data_utils.DataLoader(
        validation_encoder_dataset, batch_size=encoder_batch_size, shuffle=False
    ),
)
encoder_trainer.test(
    dataloaders=data_utils.DataLoader(
        test_encoder_dataset, batch_size=encoder_batch_size, shuffle=False
    )
)

rmses, maes = [], []
for patient_params, result in zip_tqdm(test_patient_profiles, test_results):
    for counterfactuals in result.counterfactuals_over_time:
        volume_seqs, next_usages_seqs, patient_types, end_time_steps = [], [], [], []
        for counterfactual in counterfactuals:
            volume_seq, next_usages_seq, patient_type, end_time_step = extract_result(
                counterfactual, patient_params
            )

            volume_seqs.append(volume_seq)
            next_usages_seqs.append(next_usages_seq)
            patient_types.append(patient_type)
            end_time_steps.append(end_time_step)

        (
            outcomes_seqs,
            next_treatments_seqs,
            static_covariates_grp,
            end_time_steps,
        ) = process_data(volume_seqs, next_usages_seqs, patient_types, end_time_steps)

        encoder_trainer.lightning_module.eval()
        with torch.no_grad():
            batch = CRNBaseDataset.EncoderBatch(
                outcome_seqs=outcomes_seqs[:, :-1, :],
                next_outcome_seqs=outcomes_seqs[:, 1:, :],
                next_treatment_seqs=next_treatments_seqs[:, :-1, :],
                static_covariates=static_covariates_grp,
                end_time_steps=end_time_steps,
            )
            next_outcome_seqs = encoder_trainer.lightning_module.regress_outcome(batch)
            mask_seqs = make_mask(batch.outcome_seqs.shape[1], end_time_steps=end_time_steps)

            np_next_outcome_seqs = ei.asnumpy(next_outcome_seqs[mask_seqs])
            np_batch_next_outcome_seqs = ei.asnumpy(batch.next_outcome_seqs[mask_seqs])
            rmse = root_mean_squared_error(np_next_outcome_seqs, np_batch_next_outcome_seqs)
            rmses.append(rmse / DEATH_CONDITION_VOLUME * 100)

            mae = mean_absolute_error(np_next_outcome_seqs, np_batch_next_outcome_seqs)
            maes.append(mae / DEATH_CONDITION_VOLUME * 100)

rmse, mae = np.mean(rmses), np.mean(maes)
print(f'RMSE: {rmse:.4f}%, MAE: {mae:.4f}%')
for logger in encoder_trainer.loggers:
    logger.log_metrics(
        {
            'counterfactual/nrmse': rmse,
            'counterfactual/nmae': mae,
        }
    )

# seed_everything(0, workers=True)

# encoder_checkpoint_map = {
#     (0, 0): 'logs/crn/20250101_210920_c0.0_r0.0/version_0/checkpoints',
#     (2, 2): 'logs/crn/20250101_211847_c2.0_r2.0/version_0/checkpoints',
#     (4, 4): 'logs/crn/20250101_212755_c4.0_r4.0/version_0/checkpoints',
#     (6, 6): 'logs/crn/20250101_212814_c6.0_r6.0/version_0/checkpoints',
#     (8, 8): 'logs/crn/20250101_213944_c8.0_r8.0/version_0/checkpoints',
#     (10, 10): 'logs/crn/20250101_213956_c10.0_r10.0/version_0/checkpoints'
# }

# def get_checkpoint(chemo_gamma, radio_gamma):
#     checkpoint_dir = encoder_checkpoint_map[(chemo_gamma, radio_gamma)]
#     checkpoint_file = os.listdir(checkpoint_dir)[0]
#     return f'{checkpoint_dir}/{checkpoint_file}'

# lightning_module = CRNLightningModule.load_from_checkpoint(
#     get_checkpoint(chemo_gamma, radio_gamma)
# )

# train_decoder_dataset = CRNDecoderDataset(
#     patient_profiles=train_patient_profiles,
#     results=train_results,
#     max_time_steps=max_time_steps,
#     encoder=lightning_module.module,
# )
# validation_decoder_dataset = CRNDecoderDataset(
#     patient_profiles=validation_patient_profiles,
#     results=validation_results,
#     max_time_steps=max_time_steps,
#     encoder=lightning_module.module,
# )
# test_decoder_dataset = CRNDecoderDataset(
#     patient_profiles=test_patient_profiles,
#     results=test_results,
#     max_time_steps=max_time_steps,
#     encoder=lightning_module.module,
# )

# decoder_trainer = ltn.Trainer(
#     max_epochs=num_epochs,
#     deterministic=True,
#     logger=[
#         CSVLogger(save_dir='logs/crn', name=decoder_run_name),
#         WandbLogger(
#             project='crn', save_dir='logs/crn',name=decoder_run_name,
#         ),
#     ],
# )

# decoder_trainer.fit(
#     model=CRNLightningModule(
#         outcomes_stats=outcomes_stats,
#         static_covariates_stats=static_covariates_stats,
#         max_outcome_value=DEATH_CONDITION_VOLUME,
#         num_outcomes=train_decoder_dataset.num_outcomes,
#         num_treatments=train_decoder_dataset.num_treatments,
#         num_static_covariates=train_decoder_dataset.num_static_covariates,
#     ),
#     train_dataloaders=data_utils.DataLoader(train_decoder_dataset, batch_size=decoder_batch_size, shuffle=True),
#     val_dataloaders=data_utils.DataLoader(validation_decoder_dataset, batch_size=decoder_batch_size, shuffle=False),
# )
# decoder_trainer.test(
#     dataloaders=data_utils.DataLoader(test_decoder_dataset, batch_size=decoder_batch_size, shuffle=False)
# )
