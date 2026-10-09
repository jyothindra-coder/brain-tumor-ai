
from pathlib import Path
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
import imagehash

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "Training"
TEST_DIR = ROOT / "data" / "Testing"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HASH_DISTANCE_THRESHOLD = 5

def get_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

def calculate_hash(path):
    try:
        with Image.open(path) as image:
            return imagehash.phash(image.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as error:
        print(f"Could not read {path}: {error}")
        return None

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError("Check that data/Training and data/Testing exist.")

train_images = get_images(TRAIN_DIR)
test_images = get_images(TEST_DIR)

print(f"Training images: {len(train_images)}")
print(f"Testing images: {len(test_images)}")
print("Calculating perceptual hashes...")

train_hashes = []
for path in train_images:
    value = calculate_hash(path)
    if value is not None:
        train_hashes.append((path, value))

test_hashes = []
for path in test_images:
    value = calculate_hash(path)
    if value is not None:
        test_hashes.append((path, value))

# Bucket by class label for easier review.
# This does not assume that cross-class matches are impossible.
matches = []

for test_path, test_hash in test_hashes:
    for train_path, train_hash in train_hashes:
        distance = test_hash - train_hash

        if distance <= HASH_DISTANCE_THRESHOLD:
            matches.append((distance, train_path, test_path))

matches.sort(key=lambda item: item[0])

print(f"\nCandidate cross-split near-duplicate pairs: {len(matches)}")
print(f"Perceptual hash distance threshold: {HASH_DISTANCE_THRESHOLD}")

if matches:
    print("\nReview these candidate pairs manually:")
    for distance, train_path, test_path in matches:
        print(f"\nDistance: {distance}")
        print(f"  Training: {train_path.relative_to(ROOT)}")
        print(f"  Testing:  {test_path.relative_to(ROOT)}")
else:
    print("\nNo candidate pairs found at this threshold.")

print(
    "\nImportant: these are candidates, not confirmed duplicates. "
    "Perceptual hashes can produce false matches and miss related images."
)