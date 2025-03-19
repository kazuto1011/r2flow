from torch.hub import load_state_dict_from_url

from r2flow.utils.inference import setup_model as _setup_model


def _get_url(key: str) -> str:
    return f"https://github.com/kazuto1011/r2flow/releases/download/weights/{key}.pth"


def pretrained_r2flow(config: str = "r2flow-kitti360-2rf", ckpt: str = None, **kwargs):
    """
    R2Flow proposed in "Fast LiDAR Data Generation with Rectified Flows".
    Please refer to the project release page for available pre-trained weights: https://github.com/kazuto1011/r2flow/releases/tag/weights.

    Args:
        config (str): Configuration string. (default: "r2flow-kitti360-2rf")
        ckpt (str): Path to a checkpoint file. If specified, config will be ignored. (default: None)
        **kwargs: Additional keyword arguments for model setup.

    Returns:
        tuple: A tuple of the model, LiDAR utilities, and a configuration dict.
    """
    if ckpt is None:
        ckpt = load_state_dict_from_url(_get_url(config), map_location="cpu")
    model, lidar_utils, cfg = _setup_model(ckpt, **kwargs)
    return model, lidar_utils, cfg
