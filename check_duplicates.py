
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parent
SPLITS = ["Training", "Testing"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hash_to_paths = defaultdict(list)
split_counts = defaultdict(int)

for split in SPLITS:
    split_dir = ROOT / "data" / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {split_dir}")

    for path in split_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_to_paths[digest].append(path)
            split_counts[split] += 1

print("Image counts:")
for split in SPLITS:
    print(f"  {split}: {split_counts[split]}")

duplicates = {
    digest: paths
    for digest, paths in hash_to_paths.items()
    if len(paths) > 1
}

cross_split = {
    digest: paths
    for digest, paths in duplicates.items()
    if len({p.parts[-3] for p in paths}) > 1
}

print(f"\nExact duplicate groups: {len(duplicates)}")
print(f"Groups shared across Training and Testing: {len(cross_split)}")

if duplicates:
    print("\nDuplicate files:")
    for paths in duplicates.values():
        print("\n" + " <-> ".join(str(p.relative_to(ROOT)) for p in paths))

if cross_split:
    print("\nWARNING: Exact duplicates cross the Training/Testing boundary.")
    print("Investigate these before reporting final test performance.")
else:
    print("\nNo exact duplicates were found across Training and Testing.")

print("\nNote: Exact hashing does not detect resized, cropped, or near-duplicate images.")
