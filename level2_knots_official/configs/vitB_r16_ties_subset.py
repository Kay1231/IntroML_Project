import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIT_ARCH = "ViT-B-32-CLIP"
CACHE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "hf_cache"))
HEAD_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "clip_heads"))
LOCAL_CLIP_SNAPSHOT = os.path.join(
    CACHE_DIR,
    "models--openai--clip-vit-base-patch32",
    "snapshots",
    "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
)

EVAL_DATASETS = [
    {
        "name": "mnist",
        "shuffle_train": True,
        "crop_ratio": 1.0,
        "clip_encodings": os.path.join(HEAD_DIR, VIT_ARCH, "mnist_head.pt"),
        "batch_size": 64,
        "num_workers": 0,
    },
    {
        "name": "svhn",
        "shuffle_train": True,
        "crop_ratio": 1.0,
        "clip_encodings": os.path.join(HEAD_DIR, VIT_ARCH, "svhn_head.pt"),
        "batch_size": 64,
        "num_workers": 0,
    },
    {
        "name": "gtsrb",
        "shuffle_train": True,
        "crop_ratio": 1.0,
        "clip_encodings": os.path.join(HEAD_DIR, VIT_ARCH, "gtsrb_head.pt"),
        "batch_size": 64,
        "num_workers": 0,
    },
]

config = {
    "dataset": EVAL_DATASETS,
    "model": {
        "name": "hf_clip",
        "base_type": LOCAL_CLIP_SNAPSHOT,
        "cachedir": CACHE_DIR,
        "bases": [
            "hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars",
            "hoffman-lab/KnOTS-ViT-B-32_lora_R16_dtd",
            "hoffman-lab/KnOTS-ViT-B-32_lora_R16_eurosat",
            "hoffman-lab/KnOTS-ViT-B-32_lora_R16_gtsrb",
            "hoffman-lab/KnOTS-ViT-B-32_lora_R16_mnist",
            "hoffman-lab/KnOTS-ViT-B-32_lora_R16_resisc45",
            "hoffman-lab/KnOTS-ViT-B-32_lora_R16_sun397",
            "hoffman-lab/KnOTS-ViT-B-32_lora_R16_svhn",
        ],
        "ft_config": {
            "type": "lora",
            "r": 16,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
        },
    },
    "task_merge_config": {
        "representation": "vector",
        "sign_resolve_mode": "sum_of_values",
        "scaling_coeffs": 0.3,
        "topK": 30,
        "merge_method": "ties",
        "merging_type": "mean",
        "dare": False,
        "dare_pruning_coeffs": 0.0,
    },
    "eval_type": "clip",
}
