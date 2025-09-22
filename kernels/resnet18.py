# benchmark_resnet18_seg.py
# pip install torch torchvision numpy

import argparse, time, statistics, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18

# ---- Model: ResNet18 encoder + tiny decoder -> [B,3,64,64] ----
class ResNet18Seg(nn.Module):
    def __init__(self, n_classes=3, in_ch=1):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # 64->32
        self.enc1 = base.maxpool                                    # 32->16
        self.enc2 = base.layer1                                     # 16
        self.enc3 = base.layer2                                     # 8
        self.enc4 = base.layer3                                     # 4
        self.enc5 = base.layer4                                     # 2 (512 ch)

        # lightweight decoder back to 64x64
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)  # 2->4
        self.c4  = nn.Conv2d(256, 256, 3, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)  # 4->8
        self.c3  = nn.Conv2d(128, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64,  2, 2)  # 8->16
        self.c2  = nn.Conv2d(64,  64,  3, padding=1)
        self.up1 = nn.ConvTranspose2d(64,  64,  2, 2)  # 16->32
        self.c1  = nn.Conv2d(64,  64,  3, padding=1)
        self.up0 = nn.ConvTranspose2d(64,  64,  2, 2)  # 32->64
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):                    # x: [B,1,64,64]
        x0 = self.enc0(x)                    # 32x32
        x1 = self.enc1(x0)                   # 16x16
        x2 = self.enc2(x1)                   # 16x16
        x3 = self.enc3(x2)                   # 8x8
        x4 = self.enc4(x3)                   # 4x4
        x5 = self.enc5(x4)                   # 2x2

        y  = F.relu(self.c4(self.up4(x5)))   # 4x4
        y  = F.relu(self.c3(self.up3(y)))    # 8x8
        y  = F.relu(self.c2(self.up2(y)))    # 16x16
        y  = F.relu(self.c1(self.up1(y)))    # 32x32
        y  = self.up0(y)                     # 64x64
        return self.out(y)                   # [B,3,64,64] (logits)

def benchmark(model, device, batch_size=1, steps=200, warmup=50, use_amp=False):
    model.eval().to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # synthetic input close to paper’s setting: grayscale 64x64
    x = torch.randn(batch_size, 1, 64, 64, device=device)

    # optional autocast for inference
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    autocast = torch.cuda.amp.autocast if device.type == "cuda" and use_amp else torch.cpu.amp.autocast

    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            with autocast(dtype=amp_dtype) if use_amp else torch.no_grad():
                y = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # timed runs
    times = []
    times_backbone = []  # raw resnet18 (enc0..enc5)
    times_head = []      # additional upsampling head (up4..out)
    with torch.inference_mode():
        for _ in range(steps):
            t0 = time.perf_counter()
            with autocast(dtype=amp_dtype) if use_amp else torch.no_grad():
                y = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

            # backbone (raw resnet18)
            t0 = time.perf_counter()
            with autocast(dtype=amp_dtype) if use_amp else torch.no_grad():
                x0 = model.enc0(x)
                x1 = model.enc1(x0)
                x2 = model.enc2(x1)
                x3 = model.enc3(x2)
                x4 = model.enc4(x3)
                z  = model.enc5(x4)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_backbone.append((t1 - t0) * 1000.0)

            # head (upsampling decoder)
            t0 = time.perf_counter()
            with autocast(dtype=amp_dtype) if use_amp else torch.no_grad():
                y  = F.relu(model.c4(model.up4(z)))
                y  = F.relu(model.c3(model.up3(y)))
                y  = F.relu(model.c2(model.up2(y)))
                y  = F.relu(model.c1(model.up1(y)))
                y  = model.up0(y)
                _  = model.out(y)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_head.append((t1 - t0) * 1000.0)

    ms_mean = float(np.mean(times))
    ms_p50  = float(np.percentile(times, 50))
    ms_p95  = float(np.percentile(times, 95))
    fps = (1000.0 / ms_mean) * batch_size
    bb_ms_mean = float(np.mean(times_backbone)) if times_backbone else 0.0
    head_ms_mean = float(np.mean(times_head)) if times_head else 0.0

    return {
        "batch_size": batch_size,
        "steps": steps,
        "use_amp": use_amp,
        "latency_ms_mean": ms_mean,
        "latency_ms_p50": ms_p50,
        "latency_ms_p95": ms_p95,
        "throughput_fps": fps,
        "backbone_ms_mean": bb_ms_mean,
        "head_ms_mean": head_ms_mean,
    }

def maybe_compile(model, do_compile):
    if not do_compile:
        return model
    try:
        return torch.compile(model)  # PyTorch 2.x
    except Exception:
        return model  # silently fall back

def main():
    ap = argparse.ArgumentParser(description="Benchmark ResNet18 encoder-decoder @ 64x64 (inference only).")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--batches", type=str, default="1,8,32,64", help="Comma-separated batch sizes to test.")
    ap.add_argument("--steps", type=int, default=200, help="Timed iterations per batch size.")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup iterations per batch size.")
    ap.add_argument("--amp", action="store_true", help="Enable autocast (fp16 on CUDA, bf16 on CPU if available).")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile if available.")
    args = ap.parse_args()

    dev = (
        torch.device("cuda") if (args.device in ["auto","cuda"] and torch.cuda.is_available()) else torch.device("cpu")
    )

    model = ResNet18Seg()
    model = maybe_compile(model, args.compile)

    for bs in [int(s) for s in args.batches.split(",")]:
        stats = benchmark(model, dev, batch_size=bs, steps=args.steps, warmup=args.warmup, use_amp=args.amp)
        print(
            f"[device={dev.type} bs={stats['batch_size']} amp={stats['use_amp']}] "
            f"cnn_total_mean={stats['latency_ms_mean']:.2f}ms  backbone_mean={stats['backbone_ms_mean']:.2f}ms  "
            f"head_mean={stats['head_ms_mean']:.2f}ms  p50={stats['latency_ms_p50']:.2f}ms  "
            f"p95={stats['latency_ms_p95']:.2f}ms  FPS~{stats['throughput_fps']:.1f}"
        )

if __name__ == "__main__":
    main()
