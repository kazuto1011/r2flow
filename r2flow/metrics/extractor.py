import torch
from torch import nn

from .models import minkowskinet, pointnet, rangenet, spvcnn


class Identity(nn.Identity):
    def forward(self, x, *args, **kwargs):
        return x


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        resolution,
        metrics=("FRD", "FRID", "FPD", "FSVD", "FPVD"),
        compile=False,
        WEIGHT_URL="https://github.com/kazuto1011/r2flow/releases/download/weights/",
    ):
        super().__init__()
        H, W = resolution
        self.metrics = metrics
        self.WEIGHT_URL = WEIGHT_URL
        self.compile = compile
        self.models = nn.ModuleDict()

        # =============================================================================
        # Image-based
        # =============================================================================

        if "FRD" in self.metrics:
            _model, _preprocess = rangenet.rangenet53(
                weights=f"SemanticKITTI_{H}x{W}",
                compile=self.compile,
            )
            _postprocess = rangenet.PostProcess()
            self.models["FRD"] = nn.ModuleDict()
            self.models["FRD"]["extractor"] = _model
            self.models["FRD"]["preprocess"] = _preprocess
            self.models["FRD"]["postprocess"] = _postprocess

        if "FRID" in self.metrics:
            _model, _preprocess = rangenet.build_rangenet(
                url_or_file=self.WEIGHT_URL + "rangenet21_lidm.tar.gz",
                compile=self.compile,
            )
            _postprocess = rangenet.PostProcess()
            self.models["FRID"] = nn.ModuleDict()
            self.models["FRID"]["extractor"] = _model
            self.models["FRID"]["preprocess"] = _preprocess
            self.models["FRID"]["postprocess"] = _postprocess

        # =============================================================================
        # Point-based
        # =============================================================================

        if "FPD" in self.metrics:
            _model = pointnet.pretrained_model(
                dataset="shapenet",
                compile=self.compile,
            )
            self.models["FPD"] = nn.ModuleDict()
            self.models["FPD"]["extractor"] = _model
            self.models["FPD"]["preprocess"] = Identity()
            self.models["FPD"]["postprocess"] = Identity()

        # =============================================================================
        # Voxel-based
        # =============================================================================

        if "FSVD" in self.metrics:
            _model, _preprocess, _postprocess = minkowskinet.pretrained_model(
                self.WEIGHT_URL + "minkowskinet_lidm.tar.gz",
                compile=False,
            )
            self.models["FSVD"] = nn.ModuleDict()
            self.models["FSVD"]["extractor"] = _model
            self.models["FSVD"]["preprocess"] = _preprocess
            self.models["FSVD"]["postprocess"] = _postprocess

        if "FPVD" in self.metrics:
            _model, _preprocess, _postprocess = spvcnn.pretrained_model(
                self.WEIGHT_URL + "spvcnn_lidm.tar.gz",
                compile=False,
            )
            self.models["FPVD"] = nn.ModuleDict()
            self.models["FPVD"]["extractor"] = _model
            self.models["FPVD"]["preprocess"] = _preprocess
            self.models["FPVD"]["postprocess"] = _postprocess

    def forward(
        self, x: torch.Tensor, metrics: str, feature: str = None, agg_type: str = None
    ) -> torch.Tensor:
        assert metrics in self.models
        x = self.models[metrics]["preprocess"](x)
        x = self.models[metrics]["extractor"](x, feature=feature)
        x = self.models[metrics]["postprocess"](x, agg_type=agg_type)
        return x

    def available_metrics(self):
        return list(self.models.keys())
