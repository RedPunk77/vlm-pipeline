"""Картинка с рамками до и после проверки для разбора ошибок"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def save_comparison(image_path, result, destination):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    stages = [("baseline", (235, 155, 30)), ("verified", (35, 190, 100))]
    if "ground_truth" in result:
        stages.append(("ground_truth", (60, 150, 240)))
    thumb = image.copy()
    thumb.thumbnail((640, 640))
    width, height = thumb.size
    canvas = Image.new("RGB", (width * len(stages), height + 36), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for index, (stage, color) in enumerate(stages):
        offset = index * width
        canvas.paste(thumb, (offset, 36))
        scene = result[stage]
        draw.text((offset + 10, 10), stage + (" / invalid JSON" if scene is None else ""), fill=color)
        for region in (scene or {}).get("objects", []):
            x1, y1, x2, y2 = region["bbox"]
            box = (offset + x1 * width / image.width, 36 + y1 * height / image.height,
                   offset + x2 * width / image.width, 36 + y2 * height / image.height)
            draw.rectangle(box, outline=color, width=2)
            label = region["label"]
            draw.text((box[0] + 2, box[1] + 2), label, fill=color, stroke_width=1, stroke_fill="black")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
