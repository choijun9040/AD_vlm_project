"""
INT8 SNR의 스케일 불변 예시 (학위 논문 Ⅴ장 2절, 24.0065 dB).
  합성 텐서 — 표준정규 256×1280(seed 0), 849번 채널만 9.33배(채널 증폭기 흉내) — 를 c배 했을 때
  텐서 단위 대칭 INT8 SNR과 fp16 여유(65,504 ÷ max|x|)를 출력한다. 모델·이미지는 쓰지 않는다.
실행:
    python scripts/demo_scale_invariance.py
"""
import torch

FP16_MAX = 65504


def int8_snr(x):
    scale = (x.abs().max() / 127.0).clamp_min(1e-12)
    xq = torch.clamp(torch.round(x / scale), -127, 127) * scale
    return 10 * torch.log10(x.pow(2).mean() / (x - xq).pow(2).mean()).item()


def main():
    torch.manual_seed(0)
    x = torch.randn(256, 1280)
    x[:, 849] *= 9.33
    print(f"{'배율 c':>10s}{'max|x|':>14s}{'fp16 여유':>12s}{'INT8 SNR':>12s}")
    for c in (1, 100, 1e4, 7.5e4, 1e6):
        y = x * c
        print(f"{c:>10.0f}{y.abs().max().item():>14,.0f}{FP16_MAX / y.abs().max().item():>12.3g}"
              f"{int8_snr(y):>12.4f}")


if __name__ == "__main__":
    main()
