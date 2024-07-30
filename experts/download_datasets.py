"""
This file downloads the exact MuJoCo datasets used in the paper.
They will be downloaded to experts/<env_name>/<env_name>_demos.npz.

NOTE: for antmaze datasets, please run
    python experts/collect_demos.py --env antmaze-large-diverse-v2
    or
    python experts/collect_demos.py --env antmaze-large-play-v2
to generate the demonstration files.
"""

from huggingface_hub import snapshot_download


envs = ["Ant-v3", "Hopper-v3", "Humanoid-v3", "Walker2d-v3"]
for env in envs:
    snapshot_download(
        repo_id="jren123/hybrid_irl_expert_demos",
        repo_type="dataset",
        allow_patterns=f"{env}/{env}_demos*.npz",
        local_dir="experts",
    )
