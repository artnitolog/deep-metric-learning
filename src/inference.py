from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


def eval_model(
    test_dataset,
    model,
    inference_batch_size,
    num_workers,
):
    tester = testers.BaseTester(
        batch_size=inference_batch_size,
        dataloader_num_workers=num_workers,
        dataset_labels=test_dataset.labels,
        set_min_label_to_zero=True
    )
    print('Start inference...')
    embeddings, labels = tester.get_all_embeddings(
        dataset=test_dataset,
        trunk_model=model,
    )
    labels = labels.squeeze(1)
    accuracy_calculator = AccuracyCalculator(
        k=1024,
        recall_ks=(1, 2, 4, 8, 10, 16, 32, 100, 1000),
        include=(
            'AMI',
            'NMI',
            'recall_at_ks',
            'mean_average_precision',
            'mean_average_precision_at_r',
            'mean_reciprocal_rank',
            'precision_at_1',
            'r_precision'
        )
    )
    print('Start accuracy calculation...')
    metrics_dict = accuracy_calculator.get_accuracy(
        embeddings, embeddings, labels, labels, True
    )
    return metrics_dict
