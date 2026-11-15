
"""
Image Forgery Detection - Quality Analysis Module
Compares original and tampered images using MSE, PSNR, AMBE and SSIM.
"""

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from skimage.metrics import structural_similarity as ssim

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def calculate_mse(original, tampered):
    original = original.astype(np.float64)
    tampered = tampered.astype(np.float64)
    return float(np.mean((original - tampered) ** 2))


def calculate_psnr(original, tampered):
    mse = calculate_mse(original, tampered)

    if mse == 0:
        return float("inf")

    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ambe(original, tampered):
    return float(abs(np.mean(original) - np.mean(tampered)))


def calculate_ssim(original, tampered):
    return float(ssim(original, tampered, data_range=255))


def get_image_files(folder):
    return sorted(
        [
            file for file in Path(folder).iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda file: file.name.lower()
    )


def get_image_key(filename, pattern):
    match = re.match(pattern, filename.stem, re.IGNORECASE)
    return match.group(1).lower() if match else None


def match_image_pairs(original_folder, tampered_folder):
    original_files = get_image_files(original_folder)
    tampered_files = get_image_files(tampered_folder)

    original_map = {}
    tampered_map = {}

    for file in original_files:
        key = get_image_key(file, r"^(img\d+)")
        if key:
            original_map.setdefault(key, []).append(file)

    for file in tampered_files:
        key = get_image_key(file, r"^(img\d+)_")
        if key:
            tampered_map.setdefault(key, []).append(file)

    common_keys = sorted(
        set(original_map) & set(tampered_map),
        key=lambda value: int(re.search(r"\d+", value).group())
    )

    pairs = []

    for key in common_keys:
        if len(original_map[key]) != 1 or len(tampered_map[key]) != 1:
            print(f"Ambiguous image pair for {key}. Skipping.")
            continue

        pairs.append((original_map[key][0], tampered_map[key][0]))

    print(f"\nOriginal image files: {len(original_files)}")
    print(f"Tampered image files: {len(tampered_files)}")
    print(f"Matched image pairs: {len(pairs)}")

    for original_file, tampered_file in pairs:
        print(f"{original_file.name}  <->  {tampered_file.name}")

    if len(pairs) != len(original_files):
        print("Warning: Some original images could not be matched.")

    return pairs


def load_and_preprocess_image(original_path, tampered_path):
    try:
        img_original = cv2.imread(str(original_path))
        img_tampered = cv2.imread(str(tampered_path))

        if img_original is None or img_tampered is None:
            raise ValueError("Unable to read one or both images.")

        if img_original.shape[:2] != img_tampered.shape[:2]:
            print(
                f"Dimension mismatch: {original_path.name} and "
                f"{tampered_path.name}. Resizing to common dimensions."
            )

            height = min(img_original.shape[0], img_tampered.shape[0])
            width = min(img_original.shape[1], img_tampered.shape[1])

            img_original = cv2.resize(img_original, (width, height))
            img_tampered = cv2.resize(img_tampered, (width, height))

        original_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        tampered_gray = cv2.cvtColor(img_tampered, cv2.COLOR_BGR2GRAY)

        return original_gray, tampered_gray

    except (cv2.error, ValueError, OSError) as error:
        print(f"Error processing {original_path.name}: {error}")
        return None, None


def analyze_image_pairs(original_folder, tampered_folder):
    pairs = match_image_pairs(original_folder, tampered_folder)

    results = {
        "image_name": [],
        "tampered_image_name": [],
        "mse": [],
        "psnr": [],
        "ambe": [],
        "ssim": []
    }

    print("\n" + "=" * 60)
    print("IMAGE FORGERY DETECTION - QUALITY ANALYSIS")
    print("=" * 60)

    for original_file, tampered_file in pairs:
        print(f"\nProcessing: {original_file.name} vs {tampered_file.name}")

        img_orig, img_tam = load_and_preprocess_image(
            original_file, tampered_file
        )

        if img_orig is None or img_tam is None:
            continue

        mse_val = calculate_mse(img_orig, img_tam)
        psnr_val = calculate_psnr(img_orig, img_tam)
        ambe_val = calculate_ambe(img_orig, img_tam)
        ssim_val = calculate_ssim(img_orig, img_tam)

        results["image_name"].append(original_file.name)
        results["tampered_image_name"].append(tampered_file.name)
        results["mse"].append(mse_val)
        results["psnr"].append(psnr_val)
        results["ambe"].append(ambe_val)
        results["ssim"].append(ssim_val)

        print(
            f"MSE: {mse_val:.2f} | PSNR: {psnr_val:.2f} dB | "
            f"AMBE: {ambe_val:.2f} | SSIM: {ssim_val:.4f}"
        )

    return results


def create_graphs(results, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.DataFrame(results)

    if df.empty:
        print("No valid image pairs available for graph generation.")
        return

    metrics = ["mse", "psnr", "ambe", "ssim"]

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plt.bar(df["image_name"], df[metric])
        plt.xlabel("Image Pairs")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filepath = os.path.join(output_folder, f"{metric}_graph.png")
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filepath}")

    relative_difference_score = (
        1 - df["ssim"].clip(lower=0, upper=1)
    )

    plt.figure(figsize=(12, 6))
    bars = plt.bar(df["image_name"], relative_difference_score)

    plt.xlabel("Image Pairs")
    plt.ylabel("Relative Difference Score")
    plt.title("Relative Image Difference Summary")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    for bar, score in zip(bars, relative_difference_score):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    filepath = os.path.join(output_folder, "forgery_summary.png")
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filepath}")


def save_results_csv(results, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.DataFrame(results)

    if df.empty:
        print("No results available for CSV export.")
        return df

    df["Forgery_Indicator"] = df["ssim"].apply(
        lambda value: "Possible structural difference"
        if value < 0.8 else "Not flagged by SSIM threshold"
    )

    df["Quality_Assessment"] = df["psnr"].apply(
        lambda value: "Poor" if value < 25
        else ("Good" if value > 30 else "Average")
    )

    csv_path = os.path.join(output_folder, "image_quality_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to: {csv_path}")
    print("\nSUMMARY STATISTICS")
    print(df[["mse", "psnr", "ambe", "ssim"]].describe())

    return df


def main():
    original_folder = "dataset/original"
    tampered_folder = "dataset/tampered"
    graphs_folder = "dataset/graphs"

    if not os.path.isdir(original_folder):
        print(f"Original images directory not found: {original_folder}")
        return

    if not os.path.isdir(tampered_folder):
        print(f"Tampered images directory not found: {tampered_folder}")
        return

    results = analyze_image_pairs(original_folder, tampered_folder)

    if not results["image_name"]:
        print("No valid image pairs were analyzed.")
        return

    create_graphs(results, graphs_folder)
    save_results_csv(results, graphs_folder)

    print("\n" + "=" * 60)
    print("QUALITY ANALYSIS COMPLETED")
    print(f"Results saved in: {graphs_folder}")
    print("=" * 60)


if __name__ == "__main__":
    main()
