"""Upscale: every hand-zoom clip goes through a 4x super-resolution model on the Apple GPU.

The model (Real-ESRGAN) works frame by frame, which on its own makes static areas shimmer:
each frame gets slightly different invented texture. So the frames are then blended over time
wherever the low-res input did not change (background, table, resting objects) and left alone
wherever it did (the moving hand). The result is steady where the scene is steady and sharp
where it moves. The original low-res clip is always kept next to the upscaled one.

Usage:
  uv run python -m relentless.upscale raw.mp4 upscaled.mp4 [--model general|x4plus]
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import media
from .config import MODELS, UPSCALE_MODELS, device



class Upscaler:
    def __init__(self, name: str = "general", temporal: int = 2, noise: float = 6.0, max_height: int = 1080):
        """`temporal` = how many neighbouring frames each side take part in the blend (0 turns it off);
        `noise` = grey levels (0-255) of change in the input still treated as "nothing moved";
        `max_height` caps the output (width follows at 16:9), e.g. 1080 or 720."""
        from spandrel import ImageModelDescriptor, ModelLoader

        spec = UPSCALE_MODELS[name]
        path = MODELS / spec["file"]
        if not path.exists():
            MODELS.mkdir(parents=True, exist_ok=True)
            print(f"downloading {spec['file']} ...", flush=True)
            urllib.request.urlretrieve(spec["url"], path)
        self.name, self.dev = name, device()
        model = ModelLoader().load_from_file(str(path))
        if not isinstance(model, ImageModelDescriptor):
            raise RuntimeError(f"{path.name} is not an image-to-image model")
        self.model = model.to(self.dev).eval()
        self.scale = int(model.scale)
        self.temporal, self.sigma = temporal, noise / 255.0
        self.max_h, self.max_w = max_height, media.even(max_height * 16 / 9)

    @torch.inference_mode()
    def _sr(self, lr: torch.Tensor) -> torch.Tensor:
        return self.model(lr).clamp_(0, 1)

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        return torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)

    @torch.inference_mode()
    def _blend(self, lr: list[torch.Tensor], sr: list[torch.Tensor], i: int) -> torch.Tensor:
        """Frame i, averaged with its neighbours wherever the low-res input stayed the same."""
        total, weight = sr[i].clone(), torch.ones_like(sr[i][:, :1])
        for k in range(len(sr)):
            if k == i:
                continue
            change = F.avg_pool2d((lr[k] - lr[i]).abs().mean(dim=1, keepdim=True), 5, stride=1, padding=2)
            w = torch.exp(-(change / self.sigma) ** 2)
            w = F.interpolate(w, size=sr[i].shape[-2:], mode="bilinear", align_corners=False)
            total += w * sr[k]
            weight += w
        return total / weight

    def video(self, src: Path, dst: Path, progress=None) -> dict:
        """Upscale a clip file. Returns sizes and timing."""
        t0 = time.time()
        with media.Reader(src) as rd:
            fps, n = rd.fps, rd.count
            # Inputs whose 4x would exceed the cap are shrunk first: same output, no wasted GPU time.
            shrink = min(1.0, self.max_w / (rd.width * self.scale), self.max_h / (rd.height * self.scale))
            in_size = (media.even(rd.width * shrink), media.even(rd.height * shrink))
            out_size = (in_size[0] * self.scale, in_size[1] * self.scale)
            r = self.temporal
            lr_q: deque = deque(maxlen=2 * r + 1)
            sr_q: deque = deque(maxlen=2 * r + 1)
            written = 0

            def emit(writer, centre: int) -> None:
                nonlocal written
                out = self._blend(list(lr_q), list(sr_q), centre) if r else sr_q[centre]
                img = (out[0].permute(1, 2, 0) * 255.0).round().clamp_(0, 255).byte().cpu().numpy()[:, :, ::-1]
                writer.write(img)
                written += 1

            with media.Writer(dst, *out_size, fps, crf=17, preset="fast") as writer:
                for i in range(n):
                    frame = rd.read(i)
                    if frame is None:
                        break
                    if (frame.shape[1], frame.shape[0]) != in_size:
                        frame = cv2.resize(frame, in_size, interpolation=cv2.INTER_AREA)
                    lr = self._to_tensor(frame)
                    lr_q.append(lr)
                    sr_q.append(self._sr(lr))
                    if len(sr_q) == 2 * r + 1:
                        # the queue is full: its middle frame now has all its neighbours
                        if written == 0:
                            for c in range(r):       # the first frames have no earlier neighbours
                                emit(writer, c)
                        emit(writer, r)
                    if progress and i % 10 == 0:
                        progress(i / max(1, n))
                # flush the tail (or everything, for clips shorter than the window)
                start = r + 1 if written else 0
                for c in range(start, len(sr_q)):
                    emit(writer, c)
        if self.dev == "mps":
            torch.mps.empty_cache()
        secs = time.time() - t0
        return {"model": self.name, "scale": self.scale, "frames": written, "seconds": round(secs, 2),
                "in_size": list(in_size), "out_size": list(out_size),
                "fps_processed": round(written / max(secs, 1e-6), 1)}


def compare(raw: Path, upscaled: Path, dst: Path, height: int = 540) -> Path:
    """Side by side: the raw crop (enlarged without smoothing, so you see its real pixels) and the upscaled clip."""
    f = (f"[0:v]scale=-2:{height}:flags=neighbor,setsar=1[a];[1:v]scale=-2:{height}:flags=bicubic,setsar=1[b];"
         f"[a][b]hstack=inputs=2:shortest=1[out]")
    media._run(["-y", "-i", str(raw), "-i", str(upscaled), "-filter_complex", f, "-map", "[out]", "-an",
                *media.X264, "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(dst)])
    return dst


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--model", choices=sorted(UPSCALE_MODELS), default="general")
    ap.add_argument("--temporal", type=int, default=2)
    ap.add_argument("--max-height", type=int, default=1080)
    a = ap.parse_args(argv)
    info = Upscaler(a.model, a.temporal, max_height=a.max_height).video(a.src, a.dst)
    print(f"{a.src.name}: {info['in_size'][0]}x{info['in_size'][1]} -> {info['out_size'][0]}x{info['out_size'][1]}, "
          f"{info['frames']} frames in {info['seconds']}s ({info['fps_processed']} fps) with {info['model']}")


if __name__ == "__main__":
    main()
