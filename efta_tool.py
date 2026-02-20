#!/usr/bin/env python3
"""
EFTA PDF Tool
=============
Split and/or extract images from EFTA-numbered PDF files.

Modes:
  --split             Split multi-page PDFs into one page per file.
  --extract-images    Extract all embedded images in native format.
  --split --extract-images   Both at once.

Input:
  A single PDF file or a directory of PDFs.

Output location:
  (default)           Same directory as the source PDF.
  --output-dir PATH   Send all output to a specific directory.
  --organize          Create a subfolder per EFTA number alongside the source.
                      e.g. EFTA00000008.pdf → EFTA00000008/

Naming conventions:
  Split pages:
    EFTA00000004.pdf (3 pages) →
      EFTA00000004.pdf  (page 1, keeps original name)
      EFTA00000005.pdf  (page 2)
      EFTA00000006.pdf  (page 3)

  Extracted images (single image per page):
    EFTA00000004.jpg

  Extracted images (multiple images per page):
    EFTA00000004[1].jpg
    EFTA00000004[2].png

Reference log:
  Automatically generates efta_reference.csv and efta_reference.json.
  CSV is written incrementally; JSON is generated from CSV at the end.

Other:
  --dry-run           Preview without making changes.
  -r, --recursive     Search subdirectories recursively.
  --resume            Skip files already processed (auto-detects from output).
  --workers N         Process N files in parallel (default: 1).
  --engine fitz|pypdf Image extraction backend (default: fitz).
                      fitz:  true extraction, original bytes/DPI, smaller files.
                      pypdf: faster, re-encodes non-JPEG (larger files, 72 DPI).

Requirements:
  pip install PyMuPDF pypdf
  Optional for --engine pypdf: pip install Pillow

Examples:
  python efta_tool.py /path/to/pdfs --split
  python efta_tool.py EFTA00000008.pdf --extract-images --organize
  python efta_tool.py /path/to/pdfs --split --extract-images --output-dir /path/to/output
  python efta_tool.py /path/to/pdfs --split --extract-images --dry-run
  python efta_tool.py /path/to/pdfs --split -r
  python efta_tool.py /path/to/pdfs --extract-images --resume
  python efta_tool.py /path/to/pdfs --extract-images --workers 8
  python efta_tool.py /path/to/pdfs --extract-images --engine pypdf --workers 16
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader, PdfWriter

# Optional imports — checked at runtime based on --engine
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


EFTA_PATTERN = re.compile(r'^(EFTA)(\d{8})\.pdf$', re.IGNORECASE)
EFTA_OUTPUT_PATTERN = re.compile(r'^EFTA(\d{8})(?:\[\d+\])?\.\w+$', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_efta_filename(filename: str) -> tuple[str, int] | None:
    match = EFTA_PATTERN.match(filename)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def build_efta_filename(prefix: str, number: int, ext: str = ".pdf") -> str:
    return f"{prefix}{number:08d}{ext}"


def build_stem(prefix: str, number: int) -> str:
    return f"{prefix}{number:08d}"


def ensure_dir(path: Path):
    if path.is_dir():
        return
    if path.exists():
        raise FileExistsError(
            f"'{path}' exists but is not a directory. "
            f"It may be corrupted or a file with the same name. "
            f"Please remove it or choose a different --output-dir."
        )
    path.mkdir(parents=True, exist_ok=True)


def resolve_output_dir(source_pdf: Path, output_dir: Path | None,
                       organize: bool, prefix: str, base_number: int) -> Path:
    if output_dir:
        return output_dir
    if organize:
        return source_pdf.parent / build_stem(prefix, base_number)
    return source_pdf.parent


def detect_resume_point(pdf_files: list[Path], output_dir: Path | None,
                        organize: bool) -> set[str]:
    skip_files = set()
    dirs_to_scan = set()
    file_to_check = {}
    for filepath in pdf_files:
        parsed = parse_efta_filename(filepath.name)
        if not parsed:
            continue
        prefix, base_number = parsed
        check_dir = resolve_output_dir(filepath, output_dir, organize, prefix, base_number)
        dirs_to_scan.add(check_dir)
        stem = build_stem(prefix, base_number)
        file_to_check[filepath.name] = stem

    existing_stems = set()
    for d in dirs_to_scan:
        if not d.is_dir():
            continue
        try:
            for f in d.iterdir():
                match = EFTA_OUTPUT_PATTERN.match(f.name)
                if match:
                    existing_stems.add(f"EFTA{match.group(1)}")
        except PermissionError:
            continue

    for fname, stem in file_to_check.items():
        if stem in existing_stems:
            skip_files.add(fname)

    return skip_files


# ---------------------------------------------------------------------------
# Reference log
# ---------------------------------------------------------------------------

class ReferenceLog:
    FIELDNAMES = ["output_file", "type", "parent_document", "page", "source_pages"]

    def __init__(self, log_dir: Path, append: bool = False):
        self.log_dir = log_dir
        self.csv_path = log_dir / "efta_reference.csv"
        self.json_path = log_dir / "efta_reference.json"
        self._count = 0

        if append and self.csv_path.exists():
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    self._count = sum(1 for _ in csv.DictReader(f))
            except Exception:
                pass
        else:
            try:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                    writer.writeheader()
            except Exception:
                pass

    def add(self, new_records: list[dict]):
        if not new_records:
            return
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                for record in new_records:
                    writer.writerow(record)
            self._count += len(new_records)
        except Exception:
            pass

    def finalize(self):
        try:
            records = []
            with open(self.csv_path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        r["page"] = int(r["page"])
                        r["source_pages"] = int(r["source_pages"])
                    except (ValueError, KeyError):
                        pass
                    records.append(r)
            log_data = {
                "generated": datetime.now().isoformat(),
                "total_files": len(records),
                "records": records,
            }
            tmp_path = self.json_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)
            tmp_path.replace(self.json_path)
        except Exception:
            pass

    @property
    def count(self):
        return self._count


# ---------------------------------------------------------------------------
# Image extraction — fitz (PyMuPDF): true extraction, preserves original bytes
# ---------------------------------------------------------------------------

def extract_images_fitz(filepath: Path, page_index: int,
                        efta_stem: str, dest_dir: Path) -> list[str]:
    """
    Extract all embedded images from a single page using PyMuPDF.
    Writes the raw image bytes as stored in the PDF — no decoding/re-encoding.

    Single image  → EFTA########.ext
    Multiple      → EFTA########[1].ext, EFTA########[2].ext, ...
    """
    extracted = []
    try:
        doc = fitz.open(str(filepath))
        page = doc[page_index]
        images = page.get_images(full=True)

        if not images:
            doc.close()
            return []

        multi = len(images) > 1

        for idx, img_info in enumerate(images, start=1):
            xref = img_info[0]
            try:
                img_data = doc.extract_image(xref)
                ext = f".{img_data['ext']}"
                raw_bytes = img_data["image"]

                if multi:
                    final_name = f"{efta_stem}[{idx}]{ext}"
                else:
                    final_name = f"{efta_stem}{ext}"

                output_path = dest_dir / final_name
                with open(str(output_path), "wb") as f:
                    f.write(raw_bytes)
                extracted.append(final_name)
            except Exception:
                continue

        doc.close()
    except Exception:
        pass

    return extracted


# ---------------------------------------------------------------------------
# Image extraction — pypdf: faster, but re-encodes non-JPEG images
# ---------------------------------------------------------------------------

def get_image_ext(filter_name: str) -> str:
    mapping = {
        "/DCTDecode": ".jpg",
        "/JPXDecode": ".jp2",
        "/FlateDecode": ".png",
        "/CCITTFaxDecode": ".tiff",
        "/JBIG2Decode": ".png",
        "/LZWDecode": ".png",
    }
    return mapping.get(filter_name, ".png")


def extract_images_pypdf(page, efta_stem: str, dest_dir: Path) -> list[str]:
    """
    Extract images using pypdf. JPEG/JP2 are written as raw bytes (lossless).
    Other formats are decoded and re-encoded via PIL (larger files, 72 DPI).
    """
    image_objects = []
    try:
        resources = page.get("/Resources")
        if not resources or "/XObject" not in resources:
            return []
        x_objects = resources["/XObject"].get_object()
        for obj_name in x_objects:
            obj = x_objects[obj_name].get_object()
            if obj.get("/Subtype") == "/Image":
                image_objects.append(obj)
    except Exception:
        return []

    if not image_objects:
        return []

    multi = len(image_objects) > 1
    extracted = []

    for idx, obj in enumerate(image_objects, start=1):
        try:
            filters = obj.get("/Filter", "")
            if isinstance(filters, list):
                filter_name = str(filters[0]) if filters else ""
            else:
                filter_name = str(filters)

            ext = get_image_ext(filter_name)
            final_name = f"{efta_stem}[{idx}]{ext}" if multi else f"{efta_stem}{ext}"
            output_path = dest_dir / final_name

            # JPEG/JP2: raw bytes (identical to fitz for these formats)
            if filter_name in ("/DCTDecode", "/JPXDecode"):
                with open(str(output_path), "wb") as f:
                    f.write(obj._data)
                extracted.append(final_name)
                continue

            # Other formats: decode via PIL
            if not HAS_PIL:
                continue

            width = int(obj["/Width"])
            height = int(obj["/Height"])
            color_space = obj.get("/ColorSpace", "/DeviceRGB")
            bits = int(obj.get("/BitsPerComponent", 8))
            data = obj.get_data()

            cs = str(color_space)
            if isinstance(color_space, list):
                cs = str(color_space[0])

            if cs in ("/DeviceRGB", "/CalRGB"):
                mode = "RGB"
            elif cs in ("/DeviceGray", "/CalGray"):
                mode = "L"
            elif cs == "/DeviceCMYK":
                mode = "CMYK"
            else:
                mode = "RGB"
            if bits == 1:
                mode = "1"

            img = Image.frombytes(mode, (width, height), data)
            if mode == "CMYK":
                img = img.convert("RGB")
            img.save(str(output_path))
            extracted.append(final_name)

        except Exception:
            continue

    return extracted


# ---------------------------------------------------------------------------
# Core: process a single PDF
# ---------------------------------------------------------------------------

def check_pdf_conflicts(dest_dir: Path, prefix: str, base_number: int,
                        page_count: int) -> list[str]:
    conflicts = []
    for i in range(1, page_count):
        target = build_efta_filename(prefix, base_number + i)
        if (dest_dir / target).exists():
            conflicts.append(target)
    return conflicts


def process_pdf(filepath: Path, do_split: bool, do_images: bool,
                output_dir: Path | None, organize: bool,
                dry_run: bool, engine: str = "fitz") -> dict:
    """Process a single EFTA PDF. Returns a serializable result dict."""
    filename = filepath.name
    parsed = parse_efta_filename(filename)
    if not parsed:
        return {"file": filename, "status": "skipped", "reason": "not a valid EFTA filename"}

    prefix, base_number = parsed
    dest_dir = resolve_output_dir(filepath, output_dir, organize, prefix, base_number)

    try:
        reader = PdfReader(str(filepath))
        page_count = len(reader.pages)
    except Exception as e:
        return {"file": filename, "status": "error", "reason": f"failed to read PDF: {e}"}

    # --- Dry run ---
    if dry_run:
        result = {"file": filename, "status": "dry-run", "pages": page_count,
                  "dest": str(dest_dir)}
        if do_split and page_count > 1:
            result["split_files"] = [build_efta_filename(prefix, base_number + i)
                                     for i in range(page_count)]
        if do_images:
            if do_split and page_count > 1:
                result["image_files"] = [f"{build_stem(prefix, base_number + i)}.<ext>"
                                         for i in range(page_count)]
            else:
                result["image_files"] = [f"{build_stem(prefix, base_number)}.<ext>"
                                         + (" (per page)" if page_count > 1 else "")]
        return result

    # Ensure output dir exists
    if dest_dir != filepath.parent:
        ensure_dir(dest_dir)

    split_files = []
    image_files = []
    img_errors = 0

    # === Split + optionally extract images ===
    if do_split and page_count > 1:
        conflicts = check_pdf_conflicts(dest_dir, prefix, base_number, page_count)
        if conflicts:
            return {
                "file": filename, "status": "error",
                "reason": f"target filename(s) already exist: {', '.join(conflicts)}",
            }

        try:
            # Write pages 2+ first
            for i in range(1, page_count):
                writer = PdfWriter()
                writer.add_page(reader.pages[i])
                out_name = build_efta_filename(prefix, base_number + i)
                out_path = dest_dir / out_name
                with open(str(out_path), "wb") as f:
                    writer.write(f)
                split_files.append(out_name)

                if do_images:
                    stem = build_stem(prefix, base_number + i)
                    if engine == "fitz":
                        imgs = extract_images_fitz(out_path, 0, stem, dest_dir)
                    else:
                        tmp_reader = PdfReader(str(out_path))
                        imgs = extract_images_pypdf(tmp_reader.pages[0], stem, dest_dir)
                    if imgs:
                        image_files.extend(imgs)
                    else:
                        img_errors += 1

            # Overwrite original with page 1
            writer = PdfWriter()
            writer.add_page(reader.pages[0])
            page1_path = dest_dir / filename
            if dest_dir == filepath.parent:
                with open(str(filepath), "wb") as f:
                    writer.write(f)
                page1_path = filepath
            else:
                with open(str(page1_path), "wb") as f:
                    writer.write(f)
            split_files.insert(0, filename)

            if do_images:
                stem = build_stem(prefix, base_number)
                if engine == "fitz":
                    imgs = extract_images_fitz(page1_path, 0, stem, dest_dir)
                else:
                    try:
                        p1_reader = PdfReader(str(page1_path))
                        imgs = extract_images_pypdf(p1_reader.pages[0], stem, dest_dir)
                    except Exception:
                        imgs = []
                if imgs:
                    image_files = imgs + image_files
                else:
                    img_errors += 1

        except Exception as e:
            for partial in split_files:
                if partial != filename:
                    try:
                        (dest_dir / partial).unlink()
                    except OSError:
                        pass
            return {"file": filename, "status": "error", "reason": f"failed during split: {e}"}

    # === Extract images only (no split) ===
    elif do_images and not do_split:
        try:
            for page_idx in range(page_count):
                page_number = base_number + page_idx
                stem = build_stem(prefix, page_number)

                if engine == "fitz":
                    imgs = extract_images_fitz(filepath, page_idx, stem, dest_dir)
                else:
                    # pypdf needs per-page access; for multi-page, use temp files
                    if page_count > 1:
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            tmp_path = Path(tmp.name)
                        try:
                            tmp_writer = PdfWriter()
                            tmp_writer.add_page(reader.pages[page_idx])
                            with open(str(tmp_path), "wb") as f:
                                tmp_writer.write(f)
                            tmp_reader = PdfReader(str(tmp_path))
                            imgs = extract_images_pypdf(tmp_reader.pages[0], stem, dest_dir)
                        finally:
                            tmp_path.unlink(missing_ok=True)
                    else:
                        imgs = extract_images_pypdf(reader.pages[page_idx], stem, dest_dir)
                if imgs:
                    image_files.extend(imgs)
                else:
                    img_errors += 1

        except Exception as e:
            return {"file": filename, "status": "error", "reason": f"image extraction failed: {e}"}

    # === Split only, single page ===
    elif do_split and page_count <= 1:
        if dest_dir != filepath.parent:
            shutil.copy2(str(filepath), str(dest_dir / filename))
            split_files.append(filename)
        else:
            return {"file": filename, "status": "skipped", "reason": "single page, no split needed"}

    result = {"file": filename, "status": "processed", "pages": page_count,
              "dest": str(dest_dir)}
    if split_files:
        result["split_files"] = split_files
    if image_files:
        result["image_files"] = image_files
    if img_errors:
        result["image_errors"] = img_errors
    if not split_files and not image_files:
        result["status"] = "skipped"
        result["reason"] = "nothing to do"
    return result


# ---------------------------------------------------------------------------
# Build reference records
# ---------------------------------------------------------------------------

def build_reference_records(result: dict) -> list[dict]:
    records = []
    filename = result["file"]
    page_count = result.get("pages", 1)
    parsed = parse_efta_filename(filename)
    if not parsed:
        return records
    prefix, base_number = parsed

    for split_file in result.get("split_files", []):
        split_parsed = parse_efta_filename(split_file)
        if split_parsed:
            _, split_num = split_parsed
            page = split_num - base_number + 1
        else:
            page = "?"
        records.append({
            "output_file": split_file,
            "type": "pdf_page",
            "parent_document": filename,
            "page": page,
            "source_pages": page_count,
        })

    for img_file in result.get("image_files", []):
        img_match = re.match(r'(EFTA)(\d{8})', img_file, re.IGNORECASE)
        if img_match:
            img_num = int(img_match.group(2))
            page = img_num - base_number + 1
        else:
            page = "?"
        records.append({
            "output_file": img_file,
            "type": "image",
            "parent_document": filename,
            "page": page,
            "source_pages": page_count,
        })

    return records


# ---------------------------------------------------------------------------
# Handle a result (runs in main process)
# ---------------------------------------------------------------------------

def handle_result(result: dict, ref_log: ReferenceLog | None,
                  output_dir: Path | None, organize: bool,
                  stats: dict):
    status = result["status"]

    if status in ("processed", "dry-run"):
        if status == "processed" and ref_log:
            records = build_reference_records(result)
            ref_log.add(records)

        dest_note = ""
        if result.get("dest") and (output_dir or organize):
            dest_note = f" → {result['dest']}"
        page_info = f" ({result.get('pages', '?')} pages)" if result.get("pages", 1) > 1 else ""
        print(f"  [OK]    {result['file']}{page_info}{dest_note}")

        if "split_files" in result:
            for name in result["split_files"]:
                print(f"          -> {name}")
        if "image_files" in result:
            for img in result["image_files"]:
                print(f"          -> {img}  (image)")
        if result.get("image_errors"):
            print(f"          ({result['image_errors']} page(s) had no extractable image)")

        stats["processed"] += 1
        stats["split_pages"] += len(result.get("split_files", []))
        stats["images"] += len(result.get("image_files", []))
        stats["img_errors"] += result.get("image_errors", 0)

    elif status == "skipped":
        print(f"  [SKIP]  {result['file']} — {result['reason']}")
        stats["skipped"] += 1

    elif status == "error":
        print(f"  [ERROR] {result['file']} — {result['reason']}")
        stats["errors"] += 1

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EFTA PDF Tool — split pages and/or extract images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/pdfs --split
  %(prog)s /path/to/pdfs --extract-images
  %(prog)s /path/to/pdfs --split --extract-images
  %(prog)s EFTA00000008.pdf --extract-images --organize
  %(prog)s /path/to/pdfs --split --output-dir /tmp/output
  %(prog)s /path/to/pdfs --split --extract-images --dry-run
  %(prog)s /path/to/pdfs --split -r
  %(prog)s /path/to/pdfs --extract-images --resume
  %(prog)s /path/to/pdfs --extract-images --workers 8
        """,
    )
    parser.add_argument(
        "input", type=str,
        help="A single EFTA PDF file, or a directory containing them.",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Split multi-page PDFs into one file per page.",
    )
    parser.add_argument(
        "--extract-images", action="store_true",
        help="Extract all embedded images in native format.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Send all output to this directory.",
    )
    parser.add_argument(
        "--organize", action="store_true",
        help="Create a subfolder per EFTA number (e.g. EFTA00000008/).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would happen without making changes.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Search subdirectories recursively for EFTA PDFs.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip files that already have output in the destination directory.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=f"Number of parallel worker threads (default: 1). "
             f"Your system has {os.cpu_count()} cores.",
    )
    parser.add_argument(
        "--engine", type=str, default="fitz", choices=["fitz", "pypdf"],
        help="Image extraction engine (default: fitz). "
             "fitz: true extraction, preserves original bytes/DPI, smaller files. "
             "pypdf: faster, but re-encodes non-JPEG images (larger files, 72 DPI).",
    )
    args = parser.parse_args()

    if not args.split and not args.extract_images:
        parser.error("Specify at least one of --split or --extract-images.")

    if args.output_dir and args.organize:
        parser.error("Use --output-dir or --organize, not both.")

    if args.workers < 1:
        parser.error("--workers must be at least 1.")

    # Validate engine dependencies
    if args.engine == "fitz" and not HAS_FITZ:
        parser.error("--engine fitz requires PyMuPDF. Install with: pip install PyMuPDF")
    if args.engine == "pypdf" and not HAS_PIL:
        print("Warning: Pillow not installed. pypdf engine can only extract JPEG/JP2 images.")
        print("         Install with: pip install Pillow\n")

    cpu_count = os.cpu_count() or 1
    if args.workers > cpu_count:
        print(f"Warning: --workers {args.workers} exceeds CPU count ({cpu_count}). "
              f"Consider using {max(1, cpu_count - 2)} or fewer.\n")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if input_path.is_file():
        if not EFTA_PATTERN.match(input_path.name):
            print(f"Error: '{input_path.name}' is not a valid EFTA filename.", file=sys.stderr)
            sys.exit(1)
        pdf_files = [input_path]
    elif input_path.is_dir():
        if args.recursive:
            pdf_files = sorted(
                [f for f in input_path.rglob("*.pdf") if EFTA_PATTERN.match(f.name)],
                key=lambda f: f.name,
            )
        else:
            pdf_files = sorted(
                [f for f in input_path.iterdir() if EFTA_PATTERN.match(f.name)],
                key=lambda f: f.name,
            )
        if not pdf_files:
            search_type = "recursively in" if args.recursive else "in"
            print(f"No EFTA PDF files found {search_type} '{input_path}'.")
            sys.exit(0)
    else:
        print(f"Error: '{input_path}' is not a file or directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        ensure_dir(output_dir)

    # Resume
    skipped_resume = set()
    if args.resume and not args.dry_run:
        skipped_resume = detect_resume_point(pdf_files, output_dir, args.organize)
        if skipped_resume:
            original_count = len(pdf_files)
            pdf_files = [f for f in pdf_files if f.name not in skipped_resume]
            print(f"Resume: skipping {len(skipped_resume)} already-processed file(s), "
                  f"{len(pdf_files)} remaining of {original_count}")
            if not pdf_files:
                print("Nothing left to process.")
                sys.exit(0)

    # Header
    mode_parts = []
    if args.split:
        mode_parts.append("split")
    if args.extract_images:
        mode_parts.append("extract-images")

    print(f"Mode: {' + '.join(mode_parts)}")
    file_count = len(pdf_files)
    if args.recursive:
        dir_count = len(set(f.parent for f in pdf_files))
        print(f"Input: {input_path} ({file_count} file{'s' if file_count != 1 else ''} "
              f"across {dir_count} director{'ies' if dir_count != 1 else 'y'}, recursive)")
    else:
        print(f"Input: {input_path} ({file_count} file{'s' if file_count != 1 else ''})")
    if output_dir:
        print(f"Output: {output_dir}")
    elif args.organize:
        print(f"Output: organized subfolders alongside source")
    else:
        print(f"Output: same directory as source")
    if args.workers > 1:
        print(f"Workers: {args.workers} threads ({cpu_count} cores available)")
    if args.extract_images:
        engine_desc = "fitz (true extraction)" if args.engine == "fitz" else "pypdf (fast, re-encodes)"
        print(f"Engine: {engine_desc}")
    if args.dry_run:
        print("=== DRY RUN MODE ===")
    print()

    # Initialize reference log
    ref_log = None
    if not args.dry_run:
        if output_dir:
            log_dir = output_dir
        else:
            log_dir = input_path if input_path.is_dir() else input_path.parent
        ref_log = ReferenceLog(log_dir, append=args.resume)

    # Stats
    stats = {"processed": 0, "skipped": 0, "errors": 0,
             "split_pages": 0, "images": 0, "img_errors": 0}
    interrupted = False

    # ===================================================================
    # Process files
    # ===================================================================
    if args.workers == 1:
        for filepath in pdf_files:
            try:
                result = process_pdf(
                    filepath, do_split=args.split, do_images=args.extract_images,
                    output_dir=output_dir, organize=args.organize,
                    dry_run=args.dry_run, engine=args.engine,
                )
                handle_result(result, ref_log, output_dir, args.organize, stats)
            except KeyboardInterrupt:
                print("\n\nInterrupted — stopping gracefully...")
                interrupted = True
                break
    else:
        shutdown = False

        def signal_handler(signum, frame):
            nonlocal interrupted, shutdown
            if not interrupted:
                interrupted = True
                shutdown = True
                print("\n\nInterrupted — finishing in-progress files, please wait...")
            else:
                print("\nForce quit.")
                os._exit(1)

        original_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for filepath in pdf_files:
                    if shutdown:
                        break
                    future = executor.submit(
                        process_pdf,
                        filepath, args.split, args.extract_images,
                        output_dir, args.organize,
                        args.dry_run, args.engine,
                    )
                    futures[future] = filepath

                for future in as_completed(futures):
                    if shutdown:
                        for f in futures:
                            f.cancel()
                        break
                    filepath = futures[future]
                    try:
                        result = future.result()
                        handle_result(result, ref_log, output_dir, args.organize, stats)
                    except Exception as e:
                        print(f"  [FATAL] {filepath.name} — unhandled exception: {e}")
                        stats["errors"] += 1

                if shutdown:
                    executor.shutdown(wait=False, cancel_futures=True)
        finally:
            signal.signal(signal.SIGINT, original_handler)

    # Finalize reference log
    if ref_log:
        ref_log.finalize()

    # Summary
    print()
    if interrupted:
        print("=== INTERRUPTED — partial results below ===")
    if ref_log and ref_log.count:
        print(f"Reference log: {ref_log.csv_path.name}, {ref_log.json_path.name} "
              f"({ref_log.count} total records)")
    parts = []
    if stats["processed"]:
        parts.append(f"{stats['processed']} processed")
    if stats["skipped"]:
        parts.append(f"{stats['skipped']} skipped")
    if stats["errors"]:
        parts.append(f"{stats['errors']} errors")
    if skipped_resume:
        parts.append(f"{len(skipped_resume)} resumed-over")
    print(f"Summary: {', '.join(parts)}")
    if stats["split_pages"]:
        print(f"         {stats['split_pages']} split page files")
    if stats["images"]:
        print(f"         {stats['images']} images extracted")
    if stats["img_errors"]:
        print(f"         {stats['img_errors']} pages with no extractable image")
    if interrupted:
        print("\nUse --resume to continue where you left off.")


if __name__ == "__main__":
    main()
