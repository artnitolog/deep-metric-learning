import os
import shutil
import torch
import numpy as np

from transformers import get_constant_schedule_with_warmup

from pytorch_metric_learning import testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import pytorch_metric_learning.utils.logging_presets as logging_presets


def get_trainer(
    model,
    train_dataset,
    test_dataset,
    loss_fn,
    device,
    lr,
    batch_size,
    warmup_iters,
    iters_per_epoch,
    sampler,
    num_workers,
    patience_epochs,
    inference_batch_size,
    logs_folder,
    weights_folder,
    tensorboard_folder
):
    for folder in [logs_folder, weights_folder, tensorboard_folder]:
        shutil.rmtree(folder, ignore_errors=True)

    record_keeper, _, _ = logging_presets.get_record_keeper(
        logs_folder, tensorboard_folder
    )

    hooks = logging_presets.HookContainer(
        record_keeper,
        primary_metric="precision_at_1",
        validation_split_name="test",
        log_freq=10,
    )

    accuracy_calculator = AccuracyCalculator(
        k=1,
        include=('precision_at_1',)
    )

    tester = testers.GlobalEmbeddingSpaceTester(
        batch_size=inference_batch_size,
        end_of_testing_hook=hooks.end_of_testing_hook,
        dataloader_num_workers=num_workers,
        accuracy_calculator=accuracy_calculator,
        dataset_labels=test_dataset.labels,
        set_min_label_to_zero=True,
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester=tester,
        dataset_dict={"test": test_dataset},
        model_folder=weights_folder,
        patience=patience_epochs
    )

    model.to(device)

    parameters = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_iters
    )

    trainer = trainers.MetricLossOnly(
        models={"trunk": model},
        optimizers={"trunk_optimizer": optimizer},
        batch_size=batch_size,
        loss_funcs={"metric_loss": loss_fn},
        lr_schedulers={"trunk_scheduler_by_iteration": scheduler},
        mining_funcs={},
        dataset=train_dataset,
        sampler=sampler,
        dataloader_num_workers=num_workers,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
        set_min_label_to_zero=True,
        dataset_labels=train_dataset.labels,
        iterations_per_epoch=iters_per_epoch if iters_per_epoch > 0 else None
    )

    return trainer