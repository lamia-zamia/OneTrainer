from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ORIGINAL_DATASET = Path(r"D:\lora-datasets\Projects\alterkyon_new\concepts\test2")
OUTPUT = HERE / "reference_style_sheet.png"
MANIFEST = HERE / "reference_style_manifest.json"
SEED = 20261027
COUNT = 8
CELL = 320
LABEL_HEIGHT = 28
MARGIN = 12
SAFE_CLOTHING_TAGS = {
    "clothes", "clothing", "dress", "shirt", "jacket", "coat", "uniform", "suit",
    "pants", "trousers", "skirt", "sweater", "hoodie", "cardigan", "robe", "top",
}
EXCLUDED_TAGS = {
    "nude", "naked", "nsfw", "explicit", "sex", "sexual", "porn", "topless", "bottomless",
    "underwear", "lingerie", "bikini", "nipples", "breasts", "cleavage", "pussy", "penis",
    "cum", "cum on body", "sex toy", "masturbation", "penetration", "anal", "vaginal",
    "fellatio", "oral", "rape", "young", "child", "loli", "shota", "teen", "teenage",
    "minor", "underage", "adolescent", "schoolgirl", "schoolboy",
}


def main() -> None:
    candidates = []
    for path in sorted(ORIGINAL_DATASET.glob("*.png"), key=lambda path: path.stem):
        caption_path = path.with_suffix(".txt")
        if not caption_path.is_file():
            continue
        tags = {tag.strip().casefold() for tag in caption_path.read_text(encoding="utf-8").split(",")}
        if tags & EXCLUDED_TAGS or not tags & SAFE_CLOTHING_TAGS:
            continue
        candidates.append(path)
    if len(candidates) < COUNT:
        raise RuntimeError(f"Expected at least {COUNT} training images in {ORIGINAL_DATASET}")
    selected = random.Random(SEED).sample(candidates, COUNT)
    columns = 4
    rows = (COUNT + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (MARGIN + columns * (CELL + MARGIN), MARGIN + rows * (CELL + LABEL_HEIGHT + MARGIN)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    records = []
    for index, path in enumerate(selected, start=1):
        column = (index - 1) % columns
        row = (index - 1) // columns
        x = MARGIN + column * (CELL + MARGIN)
        y = MARGIN + row * (CELL + LABEL_HEIGHT + MARGIN)
        with Image.open(path) as source:
            image = source.convert("RGB")
            source_size = list(image.size)
            image.thumbnail((CELL, CELL))
            sheet.paste(image, (x + (CELL - image.width) // 2, y + LABEL_HEIGHT))
        draw.text((x, y), f"Reference {index:02}", fill="black", font=font)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({
            "reference_id": f"Reference {index:02}",
            "source_file": str(path),
            "sha256": digest,
            "source_size": source_size,
        })
    sheet.save(OUTPUT, optimize=True)
    MANIFEST.write_text(json.dumps({
        "seed": SEED,
        "caption_filter": "requires a clothing tag and excludes explicit, sexual, underwear, and age-risk tags",
        "references": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
