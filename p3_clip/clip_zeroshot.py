import clip
import torch
from PIL import Image
import json
import os
from tqdm import tqdm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    json_path = os.path.join(parent_dir, "hw2_data/clip_zeroshot/id2label.json")
    img_dir = os.path.join(parent_dir, "hw2_data/clip_zeroshot/val")

    with open(json_path, "r") as f:
        id2label = json.load(f)
    print(f"Loaded {len(id2label)} labels")

    image_paths = [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if f.endswith(".png")
    ]

    texts = [f"A photo of {label}" for label in id2label.values()]
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Track successful and failed cases
    successes = []
    failures = []

    correct = 0
    total = 0
    pbar = tqdm(image_paths, desc="Evaluating accuracy")
    for image_path in pbar:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predicted_id = similarities.argmax().item()
        filename = os.path.basename(image_path)
        true_id = int(filename.split("_")[0])

        # Store prediction results
        case = {
            "image": filename,
            "true_label": id2label[str(true_id)],
            "predicted_label": id2label[str(predicted_id)],
            "confidence": similarities[0][predicted_id].item(),
        }

        if predicted_id == true_id:
            correct += 1
            if len(successes) < 5:
                successes.append(case)
        else:
            if len(failures) < 5:
                failures.append(case)
        total += 1

    accuracy = correct / total * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    print("\nSuccesses:")
    for case in successes:
        print(
            f"{case['image']} - True/Pred: {case['true_label']} @ confidence: {case['confidence']:.1f}%"
        )

    print("\nFailures:")
    for case in failures:
        print(
            f"{case['image']} - True: {case['true_label']}, Pred: {case['predicted_label']} @ confidence: {case['confidence']:.1f}%"
        )


if __name__ == "__main__":
    main()
