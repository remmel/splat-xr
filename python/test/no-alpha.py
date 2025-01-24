import cv2
import numpy as np
from pathlib import Path

from python.utils import remove_alpha, remove_alpha_uint8


def remove_alpha_channel(input_path: Path) -> None:
    """
    Remove alpha channel from a PNG image using OpenCV and save with '-no-alpha' suffix.

    Args:
        input_path (Path): Path to the PNG image file
    """
    # Read image with alpha channel (BGRA format in OpenCV - [0-255] np.uint8)
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Error: Could not read {input_path}")
        return

    # Check if image has alpha channel (4 channels)
    if img.shape[2] != 4:
        print(f"Image {input_path.name} doesn't have an alpha channel.")
        return

    output_path = input_path.parent / f"{input_path.stem}-no-alpha{input_path.suffix}"

    img_no_alpha = remove_alpha_uint8(img, False)

    cv2.imwrite(str(output_path), img_no_alpha)
    print(f"Saved image without alpha channel: {output_path.name}")


if __name__ == "__main__":
    import sys

    # if len(sys.argv) != 2:
    #     print("Usage: python remove_alpha.py <path>")
    #     print("Path can be either a single PNG file or a directory")
    #     sys.exit(1)

    # path = sys.argv[1]
    img_path = Path.home() / "workspace/splat-render/python/test/axis_webgl.png"
    remove_alpha_channel(img_path)