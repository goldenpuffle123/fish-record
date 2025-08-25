from pathlib import Path

def add_idx(dir: str, idx: int):
    root = Path(dir)
    images = [p for p in root.rglob("*.png")]
    for image in images:
        num = int(image.stem.split("_")[-1])
        num+=idx
        # print(str(root / f"im_{num:04d}{image.suffix}"))
        image.rename(root / f"im_{num:04d}{image.suffix}")

add_idx("dataset_10cm_new", 480)