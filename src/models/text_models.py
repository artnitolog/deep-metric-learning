from pathlib import Path

from transformers import DistilBertModel, DistilBertTokenizerFast

from .utils import MLPHead, CombinedTextModel


def distilbert_tokenizer_provider(download=False):
    folder = 'distilbert_base_tokenizer'
    tokenizer = DistilBertTokenizerFast.from_pretrained(folder)
    return tokenizer


def distilbert_model_provider(embedding_dim=256, add_preclassifier=True, download=False, freeze_trunk=False):
    folder = 'distilbert_base_pretrained'
    trunk = DistilBertModel.from_pretrained(folder)
    if add_preclassifier:
        head = MLPHead([trunk.config.dim, trunk.config.dim, embedding_dim], dropout=0.2)
    else:
        head = MLPHead([trunk.config.dim, embedding_dim], dropout=0)
    model = CombinedTextModel(trunk, head, freeze_trunk=freeze_trunk)
    return model
