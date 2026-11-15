
"""
Image Forgery Detection using Basic Image Processing
Detects suspicious regions by comparing original and tampered images.
"""

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageTamperingDetector:

    def __init__(
        self,
        block_size=32,
        threshold_diff=30,
        threshold_edge=0.3,
        threshold_block=0.85
    ):
        self.block_size = block_size
        self.threshold_diff = threshold_diff
        self.threshold_edge = threshold_edge
        self.threshold_block = threshold_block

    def load_images(self, original_path, tampered_path):
        original = cv2.imread(str(original_path))
        tampered = cv2.imread(str(tampered_path))

        if original is None or tampered is None:
            raise ValueError(
                f"Could not load images: {original_path} or {tampered_path}"
            )

        if original.shape[:2] != tampered.shape[:2]:
            print("  Resizing tampered image to match original dimensions.")
            tampered = cv2.resize(
                tampered,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return original, tampered

    def image_subtraction_method(self, original, tampered):
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_tamp = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_orig, gray_tamp)

        _, mask = cv2.threshold(
            diff,
            self.threshold_diff,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, diff

    def edge_inconsistency_method(self, original, tampered):
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_tamp = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

        gray_orig = cv2.GaussianBlur(gray_orig, (5, 5), 1)
        gray_tamp = cv2.GaussianBlur(gray_tamp, (5, 5), 1)

        edges_orig = cv2.Canny(gray_orig, 50, 150)
        edges_tamp = cv2.Canny(gray_tamp, 50, 150)

        edge_diff = cv2.absdiff(edges_orig, edges_tamp)

        kernel = np.ones((3, 3), np.uint8)
        edge_diff = cv2.dilate(edge_diff, kernel, iterations=2)

        _, mask = cv2.threshold(
            edge_diff,
            self.threshold_edge * 255,
            255,
            cv2.THRESH_BINARY
        )

        return mask, edge_diff

    def calculate_similarity(self, block1, block2):
        block1 = block1.astype(np.float64)
        block2 = block2.astype(np.float64)

        std1 = np.std(block1)
        std2 = np.std(block2)

        if std1 == 0 and std2 == 0:
            return 1.0 if np.mean(block1) == np.mean(block2) else 0.0

        if std1 == 0 or std2 == 0:
            return 0.0

        corr = np.corrcoef(block1.flatten(), block2.flatten())[0, 1]

        if not np.isfinite(corr):
            return 0.0

        return float(np.clip((corr + 1) / 2, 0, 1))

    def block_based_analysis(self, original, tampered):
        height, width = original.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_tamp = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):

                block_orig = gray_orig[
                    i:min(i + self.block_size, height),
                    j:min(j + self.block_size, width)
                ]

                block_tamp = gray_tamp[
                    i:min(i + self.block_size, height),
                    j:min(j + self.block_size, width)
                ]

                if block_orig.size == 0 or block_tamp.size == 0:
                    continue

                similarity = self.calculate_similarity(
                    block_orig,
                    block_tamp
                )

                if similarity < self.threshold_block:
                    mask[
                        i:min(i + self.block_size, height),
                        j:min(j + self.block_size, width)
                    ] = 255

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def combine_results(self, masks):
        if not masks:
            raise ValueError("No detection masks were provided.")

        combined = np.zeros_like(masks[0], dtype=np.float32)

        for mask in masks:
            combined += mask.astype(np.float32)

        combined /= len(masks)

        _, final_mask = cv2.threshold(
            combined,
            100,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((7, 7), np.uint8)
        final_mask = cv2.morphologyEx(
            final_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        )

        return final_mask

    def draw_bounding_boxes(self, image, mask):
        result = image.copy()

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(
                    result,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

        return result

    def create_heatmap(self, mask):
        heatmap = cv2.GaussianBlur(
            mask.astype(np.float32),
            (51, 51),
            20
        )

        if np.max(heatmap) > 0:
            heatmap = cv2.normalize(
                heatmap,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

        heatmap = heatmap.astype(np.uint8)

        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    def save_results(
        self,
        original,
        result_boxes,
        heatmap,
        diff_mask,
        edge_mask,
        block_mask,
        diff_image,
        edge_diff,
        final_mask,
        output_dir,
        filename
    ):
        results_dir = os.path.join(output_dir, "results")
        masks_dir = os.path.join(output_dir, "masks")

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        base_name = Path(filename).stem

        cv2.imwrite(
            os.path.join(results_dir, f"{base_name}_result_boxes.jpg"),
            result_boxes
        )

        cv2.imwrite(
            os.path.join(results_dir, f"{base_name}_heatmap.jpg"),
            heatmap
        )

        cv2.imwrite(
            os.path.join(masks_dir, f"{base_name}_diff_mask.jpg"),
            diff_mask
        )

        cv2.imwrite(
            os.path.join(masks_dir, f"{base_name}_edge_mask.jpg"),
            edge_mask
        )

        cv2.imwrite(
            os.path.join(masks_dir, f"{base_name}_block_mask.jpg"),
            block_mask
        )

        cv2.imwrite(
            os.path.join(masks_dir, f"{base_name}_final_mask.jpg"),
            final_mask
        )

        cv2.imwrite(
            os.path.join(masks_dir, f"{base_name}_difference_image.jpg"),
            diff_image
        )

        cv2.imwrite(
            os.path.join(masks_dir, f"{base_name}_edge_difference.jpg"),
            edge_diff
        )

        print(f"  Results saved to: {output_dir}")

    def display_results(
        self,
        original,
        tampered,
        result_boxes,
        heatmap,
        final_mask
    ):
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        tampered_rgb = cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result_boxes, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        images = [
            (original_rgb, "Original Image"),
            (tampered_rgb, "Tampered Image"),
            (result_rgb, "Detected Regions"),
            (heatmap_rgb, "Difference Heatmap"),
            (final_mask, "Final Detection Mask")
        ]

        for ax, (image, title) in zip(axes.flat, images):
            if title == "Final Detection Mask":
                ax.imshow(image, cmap="gray")
            else:
                ax.imshow(image)

            ax.set_title(title)
            ax.axis("off")

        axes.flat[5].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def detect_tampering(
        self,
        original_path,
        tampered_path,
        output_dir,
        show=False
    ):
        print(f"\nProcessing: {os.path.basename(original_path)}")

        original, tampered = self.load_images(
            original_path,
            tampered_path
        )

        print("  Applying image subtraction...")
        diff_mask, diff_image = self.image_subtraction_method(
            original,
            tampered
        )

        print("  Detecting edge inconsistencies...")
        edge_mask, edge_diff = self.edge_inconsistency_method(
            original,
            tampered
        )

        print("  Performing block-based analysis...")
        block_mask = self.block_based_analysis(
            original,
            tampered
        )

        print("  Combining detection results...")
        final_mask = self.combine_results([
            diff_mask,
            edge_mask,
            block_mask
        ])

        heatmap = self.create_heatmap(final_mask)
        heatmap_overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap,
            0.4,
            0
        )

        result_boxes = self.draw_bounding_boxes(
            original,
            final_mask
        )

        self.save_results(
            original,
            result_boxes,
            heatmap_overlay,
            diff_mask,
            edge_mask,
            block_mask,
            diff_image,
            edge_diff,
            final_mask,
            output_dir,
            os.path.basename(original_path)
        )

        if show:
            self.display_results(
                original,
                tampered,
                result_boxes,
                heatmap_overlay,
                final_mask
            )

        return final_mask


def get_image_key(filename, pattern):
    match = re.match(pattern, Path(filename).stem, re.IGNORECASE)
    return match.group(1).lower() if match else None


def main():
    dataset_dir = "dataset"

    original_dir = os.path.join(dataset_dir, "original")
    tampered_dir = os.path.join(dataset_dir, "tampered")
    output_dir_base = os.path.join(dataset_dir, "output")

    if not os.path.isdir(original_dir):
        print(f"Original directory not found: {original_dir}")
        return

    if not os.path.isdir(tampered_dir):
        print(f"Tampered directory not found: {tampered_dir}")
        return

    original_files = sorted(
        [
            file for file in Path(original_dir).iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda file: file.name.lower()
    )

    tampered_files = sorted(
        [
            file for file in Path(tampered_dir).iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda file: file.name.lower()
    )

    print(f"Original images: {len(original_files)}")
    print(f"Tampered images: {len(tampered_files)}")

    original_map = {}
    tampered_map = {}

    for file in original_files:
        key = get_image_key(file.name, r"^(img\d+)")

        if key:
            original_map.setdefault(key, []).append(file)

    for file in tampered_files:
        key = get_image_key(file.name, r"^(img\d+)_")

        if key:
            tampered_map.setdefault(key, []).append(file)

    common_keys = sorted(
        set(original_map) & set(tampered_map),
        key=lambda value: int(re.search(r"\d+", value).group())
    )

    detector = ImageTamperingDetector(
        block_size=32,
        threshold_diff=30,
        threshold_edge=0.3,
        threshold_block=0.85
    )

    successful = 0
    failed = 0

    for key in common_keys:
        if len(original_map[key]) != 1 or len(tampered_map[key]) != 1:
            print(f"Ambiguous image pair: {key}. Skipping.")
            failed += 1
            continue

        original_path = original_map[key][0]
        tampered_path = tampered_map[key][0]

        output_dir = os.path.join(
            output_dir_base,
            f"run_{key}"
        )

        try:
            detector.detect_tampering(
                original_path,
                tampered_path,
                output_dir,
                show=False
            )
            successful += 1

        except Exception as error:
            print(f"Error processing {key}: {error}")
            failed += 1

    print("\n" + "=" * 55)
    print("IMAGE FORGERY DETECTION SUMMARY")
    print("=" * 55)
    print(f"Matched pairs: {len(common_keys)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed or skipped: {failed}")
    print(f"Results location: {output_dir_base}")
    print("=" * 55)


if __name__ == "__main__":
    main()
