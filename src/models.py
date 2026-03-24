import torch.nn as nn
import segmentation_models_pytorch as smp

MODEL_REGISTRY = {
    "U-Net": "unet",
    "Attention U-Net": "unet_attention",
    "DeepLabV3+": "deeplabv3plus",
}


def build_model(architecture: str, encoder: str = "efficientnet-b4") -> nn.Module:
    common = dict(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    if architecture == "unet":
        return smp.Unet(**common)
    if architecture == "unet_attention":
        return smp.Unet(**common, decoder_attention_type="scse")
    if architecture == "deeplabv3plus":
        common["encoder_name"] = "resnet101"
        return smp.DeepLabV3Plus(**common)
    raise ValueError(f"Unknown architecture: {architecture}")
