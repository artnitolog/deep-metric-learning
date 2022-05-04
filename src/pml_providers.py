from pytorch_metric_learning import losses, samplers

contrastive_losses = [
    'AngularLoss',
    'ContrastiveLoss',
    'CentroidTripletLoss',
    'TripletMarginLoss',
    'GeneralizedLiftedStructureLoss',
    'IntraPairVarianceLoss',
    'LiftedStructureLoss',
    'MarginLoss',
    'MultiSimilarityLoss',
    'CentroidTripletLoss',
    'TupletMarginLoss',
    'CircleLoss',
    'SignalToNoiseRatioContrastiveLoss',
    'FastAPLoss',
    'NCALoss',
    'NTXentLoss',
    'SupConLoss',
]

classification_losses = [
    'LargeMarginSoftmaxLoss',
    'ArcFaceLoss',
    'CosFaceLoss',
    'SphereFaceLoss',
    'SoftTripleLoss',
    'NormalizedSoftmaxLoss',
    'SubCenterArcFaceLoss',
    'ProxyNCALoss',
    'ProxyAnchorLoss'
]

def loss_provider(
    loss_name,
    embedding_dim,
    num_classes,
    samples_per_class=1,
    loss_kwargs=None
):
    if loss_kwargs is None:
        loss_kwargs = {}
    clf_kwargs = dict(
        num_classes=num_classes,
        embedding_size=embedding_dim
    )

    loss_fn_class = getattr(losses, loss_name)
    if loss_name in contrastive_losses:
        loss_fn = loss_fn_class(**loss_kwargs)
    elif loss_name in classification_losses:
        loss_fn = loss_fn_class(**clf_kwargs, **loss_kwargs)
    else:
        raise ValueError(f'Unrecognized loss: {loss_name}')
    
    return loss_fn


def sampler_provider(train_dataset, samples_per_class):
    sampler = samplers.MPerClassSampler(
        train_dataset.labels,
        m=samples_per_class,
        length_before_new_iter=len(train_dataset)
    )
    return sampler