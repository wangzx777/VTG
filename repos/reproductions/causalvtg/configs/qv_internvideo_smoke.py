"""Small CausalVTG smoke-test config using the existing QVHighlights features."""

import copy as _copy
import os as _os
import runpy as _runpy


_workspace = _os.environ.get("VTG_WORKSPACE", "/home/jia/usr_wangzx/VTG")
_repo = _os.path.join(_workspace, "repos", "CausalVTG")
_smoke_data = _os.environ.get(
    "CAUSALVTG_SMOKE_DATA_ROOT",
    _os.path.join(
        _workspace,
        "repo",
        "reproductions",
        "causalvtg",
        "runs",
        "qvhighlights",
        "20260819-smoke",
        "data",
    ),
)
_feature_root = _os.environ.get(
    "CAUSALVTG_QV_FEATURE_ROOT",
    "/media/jia/MyProject/flashvtg/QVHighlights/features",
)
_video_features = _os.path.join(
    _feature_root, "internvideo2_video", "qvhighlight_6b"
)
_query_features = _os.environ.get(
    "CAUSALVTG_QV_QUERY_FEATURES",
    _os.path.join(_smoke_data, "internvideo2_llama_text_feature"),
)

_official = _runpy.run_path(
    _os.path.join(
        _repo,
        "configs",
        "qvhighlights",
        "qvhighlights_internvideo.py",
    )
)

_base_ = _official["_base_"]
model = _copy.deepcopy(_official["model"])
data = _copy.deepcopy(_official["data"])
stages = _copy.deepcopy(_official["stages"])
hooks = _copy.deepcopy(_official["hooks"])

model["adapter_cfg"]["video_cluster_path_list"] = [
    _os.path.join(_smoke_data, "cluster_videoclip_smoke16.npz")
]
model["adapter_cfg"]["query_cluster_path"] = _os.path.join(
    _smoke_data, "cluster_llama_text_smoke16.npz"
)
model["adapter_cfg"]["num_clusters"] = 16

data["train"]["times"] = 1
data["train"]["dataset"]["label_path"] = _os.path.join(
    _smoke_data, "qvhighlights_train_smoke.jsonl"
)
data["train"]["dataset"]["cache_path"] = [_video_features]
data["train"]["dataset"]["query_path"] = _query_features
data["train"]["loader"].update(
    batch_size=4, num_workers=0, pin_memory=False
)

for _split in ("val", "test"):
    data[_split]["label_path"] = _os.path.join(
        _smoke_data, "qvhighlights_val_smoke.jsonl"
    )
    data[_split]["cache_path"] = [_video_features]
    data[_split]["query_path"] = _query_features
    data[_split]["loader"].update(
        batch_size=4, num_workers=0, pin_memory=False
    )

stages["epochs"] = 1
stages["warmup"]["steps"] = 1
stages["lr_schedule"]["step"] = [1]
