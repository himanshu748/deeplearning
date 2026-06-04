from typing import Any

MODEL_REGISTRY = {
    "U-Net": "unet",
    "Attention U-Net": "unet_attention",
    "DeepLabV3+": "deeplabv3plus",
}


def build_model(architecture: str, encoder: str = "efficientnet-b4") -> Any:
    if architecture not in MODEL_REGISTRY.values():
        raise ValueError(f"Unknown architecture: {architecture}")

    smp = require_segmentation_models()
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
    raise AssertionError(f"Unhandled architecture: {architecture}")


def require_segmentation_models():
    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "segmentation-models-pytorch is required to build models. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return smp
