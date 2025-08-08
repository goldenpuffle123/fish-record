from pathlib import Path

paths = Path("synced_videos").glob("*5cm.mp4")
paths_range: list[Path] = []
for path in paths:
    if str(path.stem) >= "data_250807-165037":
        paths_range.append(path)
for path in paths_range:
    new_name = str(path).replace("5cm","10cm")
    path.rename(new_name)
