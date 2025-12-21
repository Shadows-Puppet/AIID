import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from features import FeatureExtractor
from train import Classifier  # reuse your model definition


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
TEST_ROOT = "../data/testing"
CHECKPOINT = "checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    model = Classifier(input_dim=775)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"✓ Loaded model from {CHECKPOINT}")
    return model


# ------------------------------------------------------------
# LOAD FEATURE EXTRACTOR
# ------------------------------------------------------------
def load_extractor():
    extractor = FeatureExtractor(device=DEVICE)
    extractor.load_normalizers("checkpoints/normalizers.npz")
    return extractor


# ------------------------------------------------------------
# COLLECT TEST FILES
# ------------------------------------------------------------
def collect_test_images(root):
    samples = []

    for generator in os.listdir(root):
        gen_path = os.path.join(root, generator)
        if not os.path.isdir(gen_path):
            continue

        for label_dir in os.listdir(gen_path):
            label = 1 if "fake" in label_dir else 0
            label_path = os.path.join(gen_path, label_dir)

            for fname in os.listdir(label_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    samples.append({
                        "path": os.path.join(label_path, fname),
                        "label": label,
                        "generator": generator
                    })

    return samples


# ------------------------------------------------------------
# EXTRACT FEATURES FOR ONE IMAGE
# ------------------------------------------------------------
def extract_features(img_path, extractor):
    img = Image.open(img_path).convert("RGB").resize((256, 256))

    with torch.no_grad():
        clip = extractor.extract_clip_features(img)
        freq = extractor.extract_frequency_features(img)
        comp = extractor.extract_compression_features(img)

    freq = (freq - extractor.freq_mean) / extractor.freq_std
    comp = (comp - extractor.comp_mean) / extractor.comp_std

    return np.concatenate([clip, freq, comp], axis=0)


# ------------------------------------------------------------
# MAIN TESTING LOOP
# ------------------------------------------------------------
def evaluate_on_test():
    model = load_model()
    extractor = load_extractor()

    samples = collect_test_images(TEST_ROOT)
    print(f"\n✓ Found {len(samples)} test images")

    all_labels = []
    all_probs = []
    all_preds = []
    all_generators = []

    for sample in tqdm(samples, desc="Testing"):
        feats = extract_features(sample["path"], extractor)
        feats = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)[0, 1].item()
            pred = torch.argmax(logits, dim=1).item()

        all_labels.append(sample["label"])
        all_probs.append(probs)
        all_preds.append(pred)
        all_generators.append(sample["generator"])

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------
    acc = accuracy_score(all_labels, all_preds) * 100
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)

    print("\n" + "="*60)
    print("OVERALL TEST PERFORMANCE")
    print("="*60)
    print(f"Accuracy: {acc:.2f}%")
    print(f"AUC:      {auc:.4f}")
    print(f"AP:       {ap:.4f}")

    # --------------------------------------------------------
    # PER-GENERATOR METRICS
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("PER-GENERATOR PERFORMANCE")
    print("="*60)

    gens = sorted(set(all_generators))
    for gen in gens:
        idx = [i for i, g in enumerate(all_generators) if g == gen]

        g_labels = [all_labels[i] for i in idx]
        g_preds = [all_preds[i] for i in idx]
        g_probs = [all_probs[i] for i in idx]

        g_acc = accuracy_score(g_labels, g_preds) * 100
        g_auc = roc_auc_score(g_labels, g_probs)

        print(f"{gen:15s} | Acc: {g_acc:6.2f}% | AUC: {g_auc:.4f}")


if __name__ == "__main__":
    evaluate_on_test()
