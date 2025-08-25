from pathlib import Path
import shutil

def move_images_to_main(root_dir):
    root = Path(root_dir)
    # Find all image files recursively (common image extensions)
    images = [p for p in root.rglob("*")]

    for img_path in images:
        dest = root / img_path.name
        shutil.move(str(img_path), str(dest))

    # Remove empty subdirectories
    for subdir in sorted(root.rglob('*'), reverse=True):
        if subdir.is_dir() and subdir != root:
            try:
                subdir.rmdir()
                print(f"Removed empty directory {subdir}")
            except OSError:
                pass

if __name__ == "__main__":
    move_images_to_main("dataset_10cm_new")