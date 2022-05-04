from pathlib import Path

from transformers import ConvNextModel

from .utils import MLPHead, CombinedModel

def convnext_model_provider(name, embedding_dim=128, download=False, freeze_trunk=False):
    folder = f'{name}_pretrained'
    trunk = ConvNextModel.from_pretrained(folder)
    if name == 'convnext_tiny':
        head = MLPHead([768, embedding_dim])
    else:
        raise ValueError(f'No model {name}')
    model = CombinedModel(trunk, 'pooler_output', head, freeze_trunk=freeze_trunk)
    return model
