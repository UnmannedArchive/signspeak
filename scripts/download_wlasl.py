"""Download a real WLASL video subset for the glosses in config.GLOSSES.

Only pulls from sources that serve video without auth: four direct-mp4 domains
(tried first, they're short single-sign clips) and YouTube via yt-dlp (top-up).
Each download is validated by actually opening it with OpenCV; junk is deleted.

Run from the repo root:  python -m scripts.download_wlasl
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse

import cv2

from asl import config as C

DIRECT = {
    "signstock.blob.core.windows.net",
    "media.asldeafined.com",
    "media.spreadthesign.com",
    "s3-us-west-1.amazonaws.com",
}
YT = {"www.youtube.com", "youtu.be"}
CAP_PER_GLOSS = 25
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def is_valid_video(path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    return ok


def download_direct(url: str, dest) -> bool:
    subprocess.run(
        ["curl", "-sS", "-m", "40", "-L", "-A", UA, "-o", str(dest), url],
        capture_output=True,
    )
    if dest.exists() and is_valid_video(dest):
        return True
    dest.unlink(missing_ok=True)
    return False


def download_youtube(url: str, dest) -> bool:
    # format 18 is progressive 360p mp4 — works without a JS runtime.
    subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--socket-timeout", "20", "-R", "2",
         "--no-playlist", "--no-warnings", "-q",
         "-f", "18/mp4[height<=480]/best[height<=480]/best",
         "--merge-output-format", "mp4", "-o", str(dest), url],
        capture_output=True,
    )
    if dest.exists() and is_valid_video(dest):
        return True
    dest.unlink(missing_ok=True)
    return False


def main():
    C.WLASL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    data = {e["gloss"]: e for e in json.load(open(C.WLASL_JSON))}
    wanted = [g for g in C.GLOSSES if g in data]

    totals = {}
    for gloss in wanted:
        insts = data[gloss]["instances"]
        # direct domains first (fast, clean), then youtube
        def rank(i):
            net = urllib.parse.urlparse(i.get("url", "")).netloc
            return 0 if net in DIRECT else (1 if net in YT else 2)
        insts = sorted(insts, key=rank)

        got = 0
        for inst in insts:
            if got >= CAP_PER_GLOSS:
                break
            url = inst.get("url", "")
            net = urllib.parse.urlparse(url).netloc
            if net not in DIRECT and net not in YT:
                continue
            dest = C.WLASL_VIDEO_DIR / f"{inst['video_id']}.mp4"
            if dest.exists() and is_valid_video(dest):
                got += 1
                continue
            ok = download_youtube(url, dest) if net in YT else download_direct(url, dest)
            if ok:
                got += 1
            print(f"  {gloss:10s} {got:2d}  {'ok ' if ok else 'skip'} {net}", flush=True)
        totals[gloss] = got
        print(f"== {gloss}: {got} clips", flush=True)

    print("\nSUMMARY")
    for g in wanted:
        print(f"  {g:10s} {totals.get(g,0)}")
    print("usable glosses (>= MIN_SAMPLES):",
          sum(1 for g in wanted if totals.get(g, 0) >= C.MIN_SAMPLES_PER_GLOSS))


if __name__ == "__main__":
    main()
