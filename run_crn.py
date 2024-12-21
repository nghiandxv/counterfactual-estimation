import numpy as np
from src.crn import (
    CRNEncoderDataset,
    CRNTrainer,
    CRNModule,
    CRNLoss,
    process_crn_simulation_result,
    process_crn_inputs_and_labels,
)
from src.simulations.non_small_lung_cancer import (
    run_simulation_with_counterfactuals,
    run_simulation,
    DefaultTreatmentPlan,
)
import torch.optim as optim
from skorch.callbacks import PassthroughScoring
import torch
import einops as ei
from src.utils.misc import tqdm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

num_patients = 1000
num_time_steps = 60
chemo_gamma = 8.0
radio_gamma = 6.0


treatment_plan = DefaultTreatmentPlan(
    chemo_gamma=chemo_gamma,
    radio_gamma=radio_gamma,
)

patient_profiles, results = run_simulation(
    num_patients=num_patients,
    num_time_steps=num_time_steps,
    treatment_plan=treatment_plan,
    seed=0,
)

callbacks = [
    PassthroughScoring(
        name='train_treatment_loss', on_train=True, lower_is_better=False
    ),
    PassthroughScoring(
        name='valid_treatment_loss', on_train=False, lower_is_better=False
    ),
    PassthroughScoring(name='train_outcome_loss', on_train=True),
    PassthroughScoring(name='valid_outcome_loss', on_train=False),
    PassthroughScoring(name='train_lambda', on_train=True),
]

encoder_dataset = CRNEncoderDataset(patient_profiles=patient_profiles, results=results)

encoder_trainer = CRNTrainer(
    module=CRNModule,
    module__num_covariates=2,
    module__num_outcomes=1,
    module__num_hidden_units=20,
    module__num_treatments=4,
    module__dropout=0.1,
    module__alpha=1.0,
    criterion=CRNLoss,
    optimizer=optim.Adam,
    lr=1e-4,
    max_epochs=100,
    batch_size=64,
    device='cuda',
    callbacks=callbacks,
)

encoder_trainer.fit(encoder_dataset)

patient_profiles, results = run_simulation_with_counterfactuals(
    num_patients=num_patients,
    num_time_steps=num_time_steps,
    treatment_plan=treatment_plan,
    seed=100,
)


rmses = []
maes = []
for patient_params, result in tqdm(zip(patient_profiles, results), total=len(results)):
    for courterfactuals in result.counterfactuals_over_time:
        full_patient_group_series = []
        full_volume_series = []
        full_treatment_series = []
        full_mask_series = []
        for counterfactual in courterfactuals:
            (
                masked_volume_series,
                masked_patient_group_series,
                masked_treatment_series,
                mask_series,
            ) = process_crn_simulation_result(
                result=counterfactual,
                patient_params=patient_params,
            )
            full_volume_series.append(masked_volume_series)
            full_patient_group_series.append(masked_patient_group_series)
            full_treatment_series.append(masked_treatment_series)
            full_mask_series.append(mask_series)

        (
            all_covariates_series,
            all_outcomes_series,
            all_used_treatment_series,
            all_next_treatment,
            all_mask_series,
        ) = process_crn_inputs_and_labels(
            full_patient_group_series,
            full_volume_series,
            full_treatment_series,
            full_mask_series,
        )

        inputs = {
            'covariates_series': torch.as_tensor(
                encoder_dataset.scale(
                    all_covariates_series, encoder_dataset.covariates_stats
                )
            ).float(),
            'used_treatment_series': torch.as_tensor(all_used_treatment_series).float(),
            'next_treatment': torch.as_tensor(all_next_treatment).float(),
        }
        with torch.no_grad():
            encoder_output = encoder_trainer.infer(inputs)
            next_outcome_series = encoder_dataset.unscale(
                ei.asnumpy(encoder_output['estimated_outcomes_series']),
                encoder_dataset.outcomes_stats,
            )

        rmse = root_mean_squared_error(
            next_outcome_series[all_mask_series], all_outcomes_series[all_mask_series]
        )
        mae = mean_absolute_error(
            next_outcome_series[all_mask_series], all_outcomes_series[all_mask_series]
        )
        rmses.append(rmse)
        maes.append(mae)

print(np.mean(rmses), np.mean(maes))
