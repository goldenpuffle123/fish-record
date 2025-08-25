import decord
import numpy as np
from pathlib import Path
import select_dialog
import cv2

def export_frames(video_path: str, output_folder: str, num_frames: int, num_start: int = 0):
    dir = Path(output_folder) / Path(video_path).stem
    dir.mkdir(parents=True, exist_ok=True)
    vr = decord.VideoReader(video_path)
    nums = np.rint(np.linspace(0, len(vr)-1, num_frames)).astype(int)
    batch = vr.get_batch(nums).asnumpy()
    for i, frame in enumerate(batch):
        out = dir / f"im_{i + num_start:04d}.png"
        cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    print(f"Exported {len(batch)} frames to {dir}")
    del vr

def main():
    video_paths = select_dialog.get_filepaths(msg = "select videos", filter="*.mp4")
    out = select_dialog.get_dir(msg = "select save dir")
    num_start = 0
    for v in video_paths:
        export_frames(v, out, 20, num_start)
        num_start += 20
if __name__ == "__main__":
    main()