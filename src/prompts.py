"""Короткий запрос с явным примером формата ответа"""
import json

PROMPT_VERSION = "structured-v2"
SYSTEM_PROMPT = (
    'You are a visual annotation assistant. Output exactly one JSON object with the key "objects". '
    'Every item must have exactly four keys: "label", "kind", "bbox", "text". '
    'Always use the key "bbox", never "bbox_2d". Never return a top-level array. '
    'Do not use Markdown fences or explanations.'
)


def build_prompt(width, height, object_labels=None):
    prompt = (
        f"Inspect this {width} by {height} pixel image. Locate each visible object. "
        "Use tight bounding boxes [x1,y1,x2,y2] in pixels of this image. "
        "Use English singular labels and kind=object, text=null for ordinary objects. "
        "Do not guess hidden objects. "
    )
    if object_labels is not None:
        prompt += "Only annotate these classes: " + json.dumps(object_labels) + ". Do not annotate text or other classes. "
    else:
        prompt += 'Also locate readable text using label="text", kind="text", and copy the words into text. Prefer COCO labels for ordinary objects. '
    prompt += (
        'Required output format, example structure only: '
        '{"objects":[{"label":"person","kind":"object","bbox":[10,20,30,40],"text":null}]}. '
        'Replace the example values with what is actually visible. '
        'If nothing matches the task, return {"objects":[]}. '
        'Remember: the response starts with {"objects": and uses bbox, not bbox_2d.'
    )
    return prompt
