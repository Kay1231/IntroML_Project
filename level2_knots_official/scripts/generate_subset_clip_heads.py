from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from dataset.templates import get_templates


CLASSNAMES = {
    "mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "svhn": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "gtsrb": [
        "red and white circle 20 kph speed limit",
        "red and white circle 30 kph speed limit",
        "red and white circle 50 kph speed limit",
        "red and white circle 60 kph speed limit",
        "red and white circle 70 kph speed limit",
        "red and white circle 80 kph speed limit",
        "end / de-restriction of 80 kph speed limit",
        "red and white circle 100 kph speed limit",
        "red and white circle 120 kph speed limit",
        "red and white circle red car and black car no passing",
        "red and white circle red truck and black car no passing",
        "red and white triangle road intersection warning",
        "white and yellow diamond priority road",
        "red and white upside down triangle yield right-of-way",
        "stop",
        "empty red and white circle",
        "red and white circle no truck entry",
        "red circle with white horizonal stripe no entry",
        "red and white triangle with exclamation mark warning",
        "red and white triangle with black left curve approaching warning",
        "red and white triangle with black right curve approaching warning",
        "red and white triangle with black double curve approaching warning",
        "red and white triangle rough / bumpy road warning",
        "red and white triangle car skidding / slipping warning",
        "red and white triangle with merging / narrow lanes warning",
        "red and white triangle with person digging / construction / road work warning",
        "red and white triangle with traffic light approaching warning",
        "red and white triangle with person walking warning",
        "red and white triangle with child and person walking warning",
        "red and white triangle with bicyle warning",
        "red and white triangle with snowflake / ice warning",
        "red and white triangle with deer warning",
        "white circle with gray strike bar no speed limit",
        "blue circle with white right turn arrow mandatory",
        "blue circle with white left turn arrow mandatory",
        "blue circle with white forward arrow mandatory",
        "blue circle with white forward or right turn arrow mandatory",
        "blue circle with white forward or left turn arrow mandatory",
        "blue circle with white keep right arrow mandatory",
        "blue circle with white keep left arrow mandatory",
        "blue circle with white arrows indicating a traffic circle",
        "white circle with gray strike bar indicating no passing for cars has ended",
        "white circle with gray strike bar indicating no passing for trucks has ended",
    ],
}


def build_head(model, tokenizer, dataset_name: str, device: str) -> torch.Tensor:
    templates = get_templates(dataset_name)
    classnames = CLASSNAMES[dataset_name]
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc=f"{dataset_name} classes"):
            embeddings = []
            for template in templates:
                tokenized = tokenizer(template(classname))
                tokenized = {
                    key: torch.tensor(value, device=device).reshape(1, -1)
                    for key, value in tokenized.items()
                }
                embedding = model.text_projection(model.text_model(**tokenized)[1])
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding)
            embedding = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
            embedding = embedding / embedding.norm()
            zeroshot_weights.append(embedding)
        weights = torch.stack(zeroshot_weights, dim=0).to(device)
        weights = torch.transpose(weights, 0, 2)
        weights = weights * model.logit_scale.exp()
        weights = weights.squeeze().float()
        return torch.transpose(weights, 0, 1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CLIP classification heads for the Table 1 subset.")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "svhn", "gtsrb"])
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--cache-dir", default="../hf_cache")
    parser.add_argument("--output-dir", default="clip_heads/ViT-B-32-CLIP")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(args.cache_dir).resolve()))

    model = CLIPModel.from_pretrained(args.model, cache_dir=args.cache_dir, use_safetensors=True).eval().to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)

    for dataset_name in args.datasets:
        output_path = output_dir / f"{dataset_name}_head.pt"
        if output_path.exists():
            print(f"Skip existing {output_path}")
            continue
        head = build_head(model, processor.tokenizer, dataset_name, args.device)
        torch.save(head, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
