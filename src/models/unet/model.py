import segmentation_models_pytorch as smp
import torch


def build_unet(encoder_name="resnet18", encoder_weights="imagenet",
               in_channels=6, classes=1):
    """Build U-Net model for mangrove segmentation.

    Output is raw logits (no sigmoid). BCEWithLogitsLoss handles sigmoid.
    SMP handles 6-channel input by modifying encoder's first conv layer.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        decoder_use_batchnorm=True,
    )
    return model


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,} ({total / 1e6:.1f}M)")
    return total


def get_model_size_mb(model):
    """Estimate model size in MB."""
    size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"Model size: {size:.1f} MB")
    return size


if __name__ == "__main__":
    model = build_unet()
    count_parameters(model)
    get_model_size_mb(model)
    # Dummy forward pass to verify
    x = torch.randn(1, 6, 256, 256)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")  # Should be (1, 1, 256, 256)
