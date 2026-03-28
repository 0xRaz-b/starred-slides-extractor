#!/usr/bin/env python3
"""
extract_starred_slides.py
-------------------------
Scans a PDF slide deck for pages containing a star marker and
outputs a new PDF containing only those pages.

Detection strategy:
    1. Text search  -- Unicode star characters (e.g. u2605, u2606) in extracted text
    2. Visual search -- small red blob in the top-right corner of each slide

Dependencies:
    pip install pypdf pdfplumber pdf2image opencv-python numpy Pillow

Usage:
    python extract_starred_slides.py input.pdf
    python extract_starred_slides.py input.pdf output.pdf
    python extract_starred_slides.py input.pdf --no-visual
    python extract_starred_slides.py input.pdf --no-text
    python extract_starred_slides.py input.pdf --dpi 200
"""

import sys
import re
import argparse
import numpy as np
from pathlib import Path

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
import cv2


# Unicode characters recognized as stars
STAR_PATTERN = re.compile(r"[★☆✩✪✫✬✭✮✯✰⭐🌟✦✧]")

# HSV ranges for red detection
# Red wraps around the hue axis in HSV, so two ranges are needed
STAR_COLOR_LOWER1 = np.array([0,    50, 200])
STAR_COLOR_UPPER1 = np.array([10,  255, 255])
STAR_COLOR_LOWER2 = np.array([170,  50, 200])
STAR_COLOR_UPPER2 = np.array([180, 255, 255])


def detect_star_text(pdf_path: str) -> set:
    """
    Extract text from each page and search for Unicode star characters.
    Returns a set of 0-indexed page numbers.
    """
    starred = set()
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if STAR_PATTERN.search(text):
                print(f"  page {i+1} -- star character found in text")
                starred.add(i)
    return starred


def detect_star_visual(pdf_path: str, dpi: int = 150) -> set:
    """
    Render each page as an image and look for a small red blob in the
    top-right corner (top 20%, right 25% of the slide).
    Returns a set of 0-indexed page numbers.
    """
    starred = set()
    images = convert_from_path(pdf_path, dpi=dpi)

    for i, pil_img in enumerate(images):
        img = np.array(pil_img.convert("RGB"))
        h_img, w_img = img.shape[:2]

        # Crop region of interest: top-right corner
        roi = img[0:int(h_img * 0.20), int(w_img * 0.75):w_img]
        roi_area = roi.shape[0] * roi.shape[1]

        # Build a mask for red pixels
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, STAR_COLOR_LOWER1, STAR_COLOR_UPPER1)
        mask2 = cv2.inRange(hsv, STAR_COLOR_LOWER2, STAR_COLOR_UPPER2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # Look for contours whose area is consistent with a small star icon
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (roi_area * 0.005) < area < (roi_area * 0.15):
                print(f"  page {i+1} -- red star detected in top-right corner")
                starred.add(i)
                break

    return starred


def extract_starred_pages(
    input_path: str,
    output_path: str,
    use_text: bool = True,
    use_visual: bool = True,
    dpi: int = 150,
) -> list:
    """
    Detect starred pages and write a filtered PDF.
    Returns the list of retained page numbers (1-indexed).
    """
    print(f"\nProcessing: {input_path}")
    reader = PdfReader(input_path)
    total = len(reader.pages)
    print(f"{total} page(s) found\n")

    starred = set()

    if use_text:
        print("Running text-based detection...")
        starred |= detect_star_text(input_path)
        if not starred:
            print("  no Unicode star characters found")

    if use_visual:
        print("\nRunning visual detection...")
        starred |= detect_star_visual(input_path, dpi=dpi)

    if not starred:
        print("\nNo starred pages detected.")
        print("Hints:")
        print("  - Make sure stars are red and positioned in the top-right corner")
        print("  - Try a higher DPI (--dpi 200) for better accuracy")
        return []

    sorted_pages = sorted(starred)
    page_nums = [p + 1 for p in sorted_pages]

    print(f"\n{len(sorted_pages)} starred slide(s) out of {total}:")
    print(f"  pages: {page_nums}")

    writer = PdfWriter()
    for idx in sorted_pages:
        writer.add_page(reader.pages[idx])

    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"\nOutput written to: {output_path}")
    return page_nums


def main():
    parser = argparse.ArgumentParser(
        description="Extract starred slides from a PDF slide deck.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python extract_starred_slides.py lecture.pdf
  python extract_starred_slides.py lecture.pdf out.pdf
  python extract_starred_slides.py lecture.pdf --no-visual
  python extract_starred_slides.py lecture.pdf --dpi 200
        """,
    )
    parser.add_argument("input", help="path to the source PDF")
    parser.add_argument(
        "output", nargs="?", default=None,
        help="path to the output PDF (default: <input>_starred.pdf)",
    )
    parser.add_argument(
        "--no-text", action="store_true",
        help="disable Unicode star character detection",
    )
    parser.add_argument(
        "--no-visual", action="store_true",
        help="disable visual (color-based) detection",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="rendering resolution for visual detection (default: 150)",
    )

    args = parser.parse_args()

    if args.no_text and args.no_visual:
        print("error: --no-text and --no-visual cannot be used together.")
        sys.exit(1)

    input_path = args.input
    output_path = args.output or (Path(input_path).stem + "_starred.pdf")

    extract_starred_pages(
        input_path=input_path,
        output_path=output_path,
        use_text=not args.no_text,
        use_visual=not args.no_visual,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()