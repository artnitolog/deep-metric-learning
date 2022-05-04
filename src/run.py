from torch.utils.tensorboard import SummaryWriter
SummaryWriter("fix_seg_fault")

import argparse
import os
import torch
import numpy as np
from pytorch_metric_learning.utils import common_functions as c_f
from pprint import pprint

from src.datasets.image_datasets import Cars196Dataset, SOPDataset, CUB200Dataset, Dogs130Dataset
from src.datasets.text_datasets import WOS134Dataset, News20Dataset
from src.datasets.splits import image_dataset_split_transform, text_dataset_split_transform

from src.models.image_models import convnext_model_provider
from src.models.text_models import distilbert_model_provider, distilbert_tokenizer_provider

from src.pml_providers import loss_provider, sampler_provider
from src.training import get_trainer
from src.inference import eval_model


image_model_list = ['convnext_tiny']
text_model_list = ['bert']

text_dataset_dict = {
    'wos134': WOS134Dataset,
    'news20': News20Dataset,    
}

img_dataset_dict = {
    'cars196': Cars196Dataset,
    'sop': SOPDataset,
    'cub200': CUB200Dataset,
    'dogs130': Dogs130Dataset,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    parser.add_argument(
        "--embedding_dim",
        required=True,
        type=int
    )

    parser.add_argument(
        "--model_name",
        required=True,
        type=str
    )

    parser.add_argument(
        "--dataset_name",
        required=True,
        type=str
    )

    parser.add_argument(
        "--loss_name",
        required=True,
        type=str
    )

    parser.add_argument(
        "--freeze_trunk",
        action="store_true",
    )

    parser.add_argument(
        "--download",
        action="store_true",
    )

    parser.add_argument(
        "--batch_size",
        required=True,
        type=int
    )

    parser.add_argument(
        "--inference_batch_size",
        required=True,
        type=int
    )

    parser.add_argument(
        "--num_epochs",
        required=True,
        type=int
    )

    parser.add_argument(
        "--iters_per_epoch",
        required=True,
        type=int
    )

    parser.add_argument(
        "--patience_epochs",
        required=True,
        type=int
    )

    parser.add_argument(
        "--lr",
        required=True,
        type=float
    )

    parser.add_argument(
        "--warmup_iters",
        required=True,
        type=int
    )

    parser.add_argument(
        "--num_workers",
        required=True,
        type=int
    )

    parser.add_argument(
        "--samples_per_class",
        required=True,
        type=int
    )

    parser.add_argument(
        "--instance_id",
        required=False,
        type=str,
        default=None
    )

    return parser.parse_args()


def set_random(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    c_f.COLLECT_STATS = True
    c_f.NUMPY_RANDOM = np.random.RandomState(seed)


def main():
    args = parse_args()

    logs_folder = "logs_folder"
    weights_folder = "weights_folder"
    tensorboard_folder = "tensorboard_folder"
    loss_kwargs = {}
    device = torch.device('cuda')

    args_dict = vars(args)
    print("Arguments:")
    pprint(args_dict)

    set_random(args.seed)

    if args.model_name in image_model_list:
        model = convnext_model_provider('convnext_tiny', download=args.download, embedding_dim=args.embedding_dim)
        dataset = img_dataset_dict[args.dataset_name](download=args.download)
        train_dataset, test_dataset = image_dataset_split_transform(dataset, random_state=args.seed)
    elif args.model_name in text_model_list:
        assert args.model_name == 'bert'
        model = distilbert_model_provider(download=args.download, freeze_trunk=args.freeze_trunk)
        tokenizer = distilbert_tokenizer_provider(download=args.download)
        dataset = text_dataset_dict[args.dataset_name](tokenizer=tokenizer, download=args.download, device=device)
        train_dataset, test_dataset = text_dataset_split_transform(dataset, random_state=args.seed)
    else:
        raise ValueError(f'Unknown model name {model_name}')
    model = model.to(device)

    loss_fn = loss_provider(
        loss_name=args.loss_name,
        embedding_dim=args.embedding_dim,
        num_classes=len(np.unique(train_dataset.labels)),
        samples_per_class=args.samples_per_class,
        loss_kwargs=loss_kwargs
    )

    sampler = sampler_provider(
        train_dataset=train_dataset,
        samples_per_class=args.samples_per_class,
    )

    trainer = get_trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss_fn=loss_fn,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        iters_per_epoch=args.iters_per_epoch,
        sampler=sampler,
        num_workers=args.num_workers,
        patience_epochs=args.patience_epochs,
        inference_batch_size=args.inference_batch_size,
        logs_folder=logs_folder,
        weights_folder=weights_folder,
        tensorboard_folder=tensorboard_folder
    )

    trainer.train(num_epochs=args.num_epochs)

    best_models = [p for p in os.listdir(f'{weights_folder}') if p.startswith('trunk_best')]
    assert len(best_models) == 1
    print(f"Loading best mode: {best_models[0]}")
    fpath = f'{weights_folder}/{best_models[0]}'
    model.load_state_dict(torch.load(fpath))
    model.to(device)
    model.eval()

    accs = eval_model(
        test_dataset=test_dataset,
        model=model,
        inference_batch_size=args.inference_batch_size,
        num_workers=args.num_workers,
    )

    print("Uploading results...")

    # Log if needed

    print('Done!')

if __name__ == "__main__":
    main()
