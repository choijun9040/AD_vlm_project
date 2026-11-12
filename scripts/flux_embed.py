"""
D* 1단계 — FLUX.1-schnell 프롬프트 임베딩을 미리 만든다(개정 16, 2026-11-04). 변환기는 받지 않는다(디스크).
프롬프트: PartiPrompts(google-research/parti PartiPrompts.tsv) 앞 100개, 파일 순서. FluxPipeline.encode_prompt(파이프라인 기본: max_sequence_length 256, bf16).

    /workspace/venv_sd/bin/python scripts/flux_embed.py --tsv data/PartiPrompts.tsv --out eval_results/cmp/flux_prompts.pt
"""
import argparse
import csv
from pathlib import Path

import torch

REPO = "black-forest-labs/FLUX.1-schnell"


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--start", type=int, default=0, help="프롬프트 시작 번호(0부터). 개정 20 측정용은 100")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from diffusers import FluxPipeline
    local = snapshot_download(REPO, allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                                                    "tokenizer/*", "tokenizer_2/*", "vae/*"])
    with open(a.tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    prompts = [r["Prompt"] for r in rows][a.start:a.start + a.n]
    pipe = FluxPipeline.from_pretrained(local, transformer=None, vae=None, torch_dtype=torch.bfloat16).to("cuda")
    pe, pp = [], []
    with torch.no_grad():
        for p in prompts:
            e, pooled, _ids = pipe.encode_prompt(prompt=p, prompt_2=p, max_sequence_length=256)
            pe.append(e.cpu()); pp.append(pooled.cpu())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": prompts, "prompt_embeds": pe, "pooled": pp, "repo": REPO, "max_sequence_length": 256,
                "dtype": "bfloat16", "source": f"google-research/parti PartiPrompts.tsv ({a.start + 1}~{a.start + len(prompts)}번)", "start": a.start}, a.out)
    print(f"저장: {a.out} ({len(prompts)}개, {tuple(pe[0].shape)})")


if __name__ == "__main__":
    main()
