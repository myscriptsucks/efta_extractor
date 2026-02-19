# EFTA PDF Tool

A command-line utility for bulk splitting and image extraction of EFTA-numbered PDF files. Designed for processing large document sets where PDFs follow the naming convention `EFTA` + 8-digit number (e.g. `EFTA00000004.pdf`).

## What It Does

### Page Splitting

Splits multi-page PDFs into individual single-page files with sequential EFTA numbering, matching actual page numbers.

```
EFTA00000004.pdf (3 pages)
  → EFTA00000004.pdf  (page 1 — keeps original name)
  → EFTA00000005.pdf  (page 2 — incremented by 1)
  → EFTA00000006.pdf  (page 3 — incremented by 2)
```

### Image Extraction

Extracts all embedded images from PDF pages in their native format (JPEG stays JPEG, etc.). Images are named after their source page's EFTA number.

Single image per page:
```
EFTA00000004.pdf
  → EFTA00000004.jpg
```

Multiple images per page:
```
EFTA00000004.pdf
  → EFTA00000004[1].jpg
  → EFTA00000004[2].png
```

Brackets and sequential numbering are only added when a page contains more than one embedded image.

The tool extracts images using **pypdf** as the primary method. For JPEG and JPEG2000 images, raw stream bytes are written directly, preserving original quality and metadata exactly. For other encodings, images are decoded via PIL and saved in an appropriate format. If pypdf can't decode an image, **pdfimages** (poppler) is used as an optional fallback.

### Combined Mode

Split pages and extract images in a single pass. Each split page gets its own images extracted with correctly numbered filenames.

### Reference Log

Every run automatically generates `efta_reference.csv` and `efta_reference.json` — a lookup table mapping every output file back to its parent document and page number. This lets you take any split page or extracted image and immediately find where it came from.

**CSV example:**

| output_file | type | parent_document | page | source_pages |
|---|---|---|---|---|
| EFTA00000004.pdf | pdf_page | EFTA00000004.pdf | 1 | 3 |
| EFTA00000005.pdf | pdf_page | EFTA00000004.pdf | 2 | 3 |
| EFTA00000006.pdf | pdf_page | EFTA00000004.pdf | 3 | 3 |
| EFTA00000004.jpg | image | EFTA00000004.pdf | 1 | 3 |
| EFTA00000005.jpg | image | EFTA00000004.pdf | 2 | 3 |

**JSON example:**

```json
{
  "generated": "2026-02-18T14:30:00.000000",
  "total_files": 5,
  "records": [
    {
      "output_file": "EFTA00000005.pdf",
      "type": "pdf_page",
      "parent_document": "EFTA00000004.pdf",
      "page": 2,
      "source_pages": 3
    }
  ]
}
```

The log is saved in the output directory (or the source directory if no output dir is specified). It is written **incrementally after each file** so that progress is preserved if the script is interrupted. It is not generated during `--dry-run`.

---

## Requirements

### Python

- Python 3.10+

### Python Packages

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pypdf Pillow
```

### Optional: poppler-utils

**poppler-utils** provides `pdfimages`, used as a fallback image extractor for images that pypdf can't decode (certain CCITT fax, JBIG2, or unusual color space encodings). The tool works fine without it for most files. Install it if you see extraction failures.

#### Linux

```bash
# Arch / CachyOS / Manjaro
sudo pacman -S poppler

# Debian / Ubuntu / Mint
sudo apt install poppler-utils

# Fedora / RHEL
sudo dnf install poppler-utils

# openSUSE
sudo zypper install poppler-tools
```

`pdfimages` will be available immediately — no PATH changes needed.

#### macOS

```bash
brew install poppler
```

#### Windows

Poppler does not have an official Windows installer. Use one of these methods:

**Option A — Conda (easiest if you use Anaconda/Miniconda):**

```powershell
conda install -c conda-forge poppler
```

This handles PATH automatically within the conda environment.

**Option B — Manual install:**

1. Download the latest poppler release for Windows from:
   https://github.com/osber/poppler-builds/releases
   (or search "poppler windows builds" — the Release-XX.XX.X-X.zip file)

2. Extract the ZIP to a permanent location, e.g.:
   ```
   C:\poppler\
   ```

3. Add poppler's `bin` directory to your system PATH:
   - Open **Start** → search **"Environment Variables"** → click **"Edit the system environment variables"**
   - Click **Environment Variables...**
   - Under **System variables**, find `Path` and click **Edit**
   - Click **New** and add:
     ```
     C:\poppler\Library\bin
     ```
     (adjust if your extracted folder structure differs — look for the folder containing `pdfimages.exe`)
   - Click **OK** on all dialogs

4. Open a **new** terminal and verify:
   ```powershell
   pdfimages -v
   ```
   You should see something like `pdfimages version 24.02.0`.

**Option C — Scoop package manager:**

```powershell
scoop install poppler
```

**Option D — MSYS2:**

```bash
pacman -S mingw-w64-x86_64-poppler
```
The binary will be in your MSYS2 `bin` directory, which MSYS2 adds to PATH within its shell.

#### Verifying the Install

On any platform, confirm it's working:

```bash
pdfimages -v
```

If the tool can't find `pdfimages` at runtime, it will print nothing — pypdf handles extraction on its own. You only need poppler if you encounter images that pypdf can't extract.

---

## Usage

```
python efta_tool.py <input> [options]
```

`<input>` can be a single PDF file or a directory of PDFs.

### Options

| Flag | Description |
|---|---|
| `--split` | Split multi-page PDFs into one file per page |
| `--extract-images` | Extract all embedded images in native format |
| `-r`, `--recursive` | Search subdirectories recursively for EFTA PDFs |
| `--resume` | Skip files that already have output in the destination |
| `--workers N` | Process N files in parallel across CPU cores (default: 1) |
| `--output-dir PATH` | Send all output to a specific directory |
| `--organize` | Create a subfolder per EFTA number alongside the source |
| `--dry-run` | Preview what would happen without making changes |

At least one of `--split` or `--extract-images` is required.

`--output-dir` and `--organize` are mutually exclusive.

### Examples

**Split all PDFs in a directory:**

```bash
python efta_tool.py /path/to/pdfs --split
```

**Extract images only (PDFs stay intact):**

```bash
python efta_tool.py /path/to/pdfs --extract-images
```

**Split and extract images together:**

```bash
python efta_tool.py /path/to/pdfs --split --extract-images
```

**Process a single file:**

```bash
python efta_tool.py EFTA00000008.pdf --split --extract-images
```

**Organize output into subfolders:**

```bash
python efta_tool.py /path/to/pdfs --split --organize
```

This creates a folder per EFTA number:

```
/path/to/pdfs/
  EFTA00000008.pdf          (original, untouched)
  EFTA00000008/
    EFTA00000008.pdf        (page 1)
    EFTA00000009.pdf        (page 2)
    EFTA00000010.pdf        (page 3)
```

**Send everything to a specific folder:**

```bash
python efta_tool.py /path/to/pdfs --extract-images --output-dir /tmp/images
```

**Preview before running:**

```bash
python efta_tool.py /path/to/pdfs --split --extract-images --dry-run
```

**Process all subdirectories recursively:**

```bash
python efta_tool.py /path/to/pdfs --split -r
python efta_tool.py /path/to/pdfs --split --extract-images -r
```

Output files are placed alongside each source PDF in its own directory.

**Resume after an interruption:**

```bash
python efta_tool.py /path/to/pdfs --extract-images --resume
```

The `--resume` flag scans the output directory for existing EFTA files and skips any source PDFs that already have output present. The reference log is appended to rather than overwritten. This is useful for large datasets where the script may be interrupted or encounter errors partway through.

**Process with multiple workers:**

```bash
python efta_tool.py /path/to/pdfs --extract-images --workers 8
python efta_tool.py /path/to/pdfs --split --extract-images --workers 4 --resume
```

Each worker is a separate process with its own CPU core — true parallel execution, not limited by Python's GIL. Worker count should generally be `cpu_count - 2` or less to avoid choking system resources. The script will warn you if you exceed your core count.

---

## Output Location Behavior

| Flags | Where output goes |
|---|---|
| *(none)* | Same directory as the source PDF |
| `--output-dir /some/path` | Everything goes to `/some/path` |
| `--organize` | Subfolder per EFTA number next to the source (e.g. `EFTA00000008/`) |

The reference log (`efta_reference.csv` and `efta_reference.json`) is always written to the top-level output directory — the `--output-dir` if specified, otherwise the source directory.

---

## Error Handling

- **Corrupt PDFs** are logged as `[ERROR]` and skipped — the script continues processing remaining files.
- **Filename conflicts** are detected before writing. If a target filename already exists, the file is skipped with an error message.
- **Partial failure safety**: when splitting, pages 2+ are written first. The original file is only overwritten with page 1 as the final step, so if the script crashes mid-process the original multi-page PDF is still intact.
- **Image extraction failures** are counted and reported in the summary but don't stop processing.

---

## File Naming Details

### Split Pages

The 8-digit number after `EFTA` is treated as a sequential document ID. Each page increments this number by 1 starting from the original.

### Extracted Images

Images are named based on the EFTA number of the page they came from:

- **Single image per page:** `EFTA00000004.jpg`
- **Multiple images per page:** `EFTA00000004[1].jpg`, `EFTA00000004[2].png`

The file extension reflects the native format of the embedded image (`.jpg`, `.png`, `.jp2`, `.tiff`, etc.).

### Reference Log Fields

| Field | Description |
|---|---|
| `output_file` | Name of the output file (PDF page or image) |
| `type` | `pdf_page` or `image` |
| `parent_document` | Original multi-page PDF this file came from |
| `page` | Page number within the parent document (1-based) |
| `source_pages` | Total number of pages in the parent document |

---

## Typical Workflow

```bash
# 1. Preview what will happen
python efta_tool.py /path/to/efta_pdfs --split --extract-images --dry-run

# 2. Run for real
python efta_tool.py /path/to/efta_pdfs --split --extract-images

# 3. Check the reference log to trace any file back to its source
#    Open efta_reference.csv in a spreadsheet, or search the JSON:
grep "EFTA00000005" efta_reference.json

# Or organized into subfolders
python efta_tool.py /path/to/efta_pdfs --split --extract-images --organize

# Or recursively through subdirectories
python efta_tool.py /path/to/efta_pdfs --split --extract-images -r
```
