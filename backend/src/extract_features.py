# extract_features.py

from features import FeatureExtractor
from data import UniversalFakeDetectDataset

import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# PURE FUNCTIONS ONLY ABOVE MAIN
# ============================================================

def cpu_features_worker(extractor, img):
    freq = extractor.extract_frequency_features(img)
    comp = extractor.extract_compression_features(img)
    return freq, comp


def extract_clip_batch(extractor, device, batch_items):
    tensors = []
    pil_images = []
    labels = []

    for path, label in batch_items:
        try:
            img = Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)
            tensors.append(extractor.preprocess(img))
            pil_images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")

    if not tensors:
        return None, None, None

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        feats = extractor.clip_model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu(), pil_images, labels


def extract_and_save_batched(
    extractor,
    dataset,
    output_file,
    device,
    batch_size,
    cpu_workers,
):
    print(f"\nExtracting {len(dataset)} images")
    print(f"Batch size: {batch_size}")
    print(f"CPU workers: {cpu_workers}")

    all_features = []
    all_labels = []

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
        for b in tqdm(range(num_batches), desc="Processing batches"):
            start = b * batch_size
            end = min(start + batch_size, len(dataset))
            batch_items = [dataset[i] for i in range(start, end)]

            clip_feats, pil_images, labels = extract_clip_batch(
                extractor, device, batch_items
            )
            if clip_feats is None:
                continue

            cpu_results = list(pool.map(
                lambda img: cpu_features_worker(extractor, img),
                pil_images
            ))

            for i, (freq, comp) in enumerate(cpu_results):
                freq = torch.from_numpy(freq).float()
                comp = torch.from_numpy(comp).float()

                if extractor.is_fitted:
                    freq = (freq - torch.from_numpy(extractor.freq_mean)) / torch.from_numpy(extractor.freq_std)
                    comp = (comp - torch.from_numpy(extractor.comp_mean)) / torch.from_numpy(extractor.comp_std)

                combined = torch.cat([
                    clip_feats[i],
                    freq,
                    comp
                ])

                all_features.append(combined.numpy())
                all_labels.append(labels[i])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(
        output_file,
        features=np.asarray(all_features, dtype=np.float32),
        labels=np.asarray(all_labels, dtype=np.int64),
    )

    print(f"✓ Saved {len(all_features)} features to {output_file}")


# ============================================================
# ENTRY POINT (ONLY EXECUTED ONCE)
# ============================================================
def main():
    device = "cuda"
    batch_size = 32
    cpu_workers = min(8, os.cpu_count() or 4)

    print("=" * 60)
    print("FAST BATCHED FEATURE EXTRACTION (WINDOWS SAFE)")
    print("=" * 60)

    print("\nInitializing feature extractor...")
    extractor = FeatureExtractor(device=device)

    extractor.load_normalizers("checkpoints/normalizers.npz")

    print("\nWarming up...")
    dummy = Image.new("RGB", (256, 256))
    extractor.extract_clip_features(dummy)
    torch.cuda.synchronize()
    print("✓ Warmup complete")

    print("\nLoading datasets...")
    train_dataset = UniversalFakeDetectDataset("../data/training")
    eval_dataset = UniversalFakeDetectDataset("../data/eval")

    extract_and_save_batched(
        extractor,
        train_dataset,
        "data/features/train_features.npz",
        device,
        batch_size,
        cpu_workers,
    )

    extract_and_save_batched(
        extractor,
        eval_dataset,
        "data/features/eval_features.npz",
        device,
        batch_size,
        cpu_workers,
    )

    print("\nDONE")


if __name__ == "__main__":
    main()
