#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

DEFAULT_PATTERN = "weight_decay = 0.01"
DEFAULT_REPLACEMENT = "weight_decay = 0.001"
DEFAULT_EXTS = [".py", ".txt", ".md", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".json", ".ipynb"]

def is_probably_binary(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            chunk = f.read(2048)
        if b"\x00" in chunk:
            return True
    except Exception:
        return True
    return False

def try_read_text(p: Path, encodings=("utf-8", "utf-8-sig", "cp1252")) -> Tuple[Optional[str], Optional[str]]:
    for enc in encodings:
        try:
            # newline="" preserves existing CRLF/LF as-is
            with p.open("r", encoding=enc, newline="") as f:
                return f.read(), enc
        except Exception:
            continue
    return None, None

def write_text(p: Path, content: str, encoding: str) -> None:
    # newline="" avoids altering line endings
    with p.open("w", encoding=encoding, newline="") as f:
        f.write(content)

def should_process(p: Path, all_files: bool, exts: Iterable[str]) -> bool:
    if not p.is_file():
        return False
    if all_files:
        return True
    return p.suffix.lower() in exts

def replace_in_file(
    p: Path,
    pattern: str,
    replacement: str,
    dry_run: bool,
    backup_ext: str,
) -> Tuple[int, bool]:
    """
    Returns (num_replacements, changed_flag)
    """
    if is_probably_binary(p):
        return 0, False

    txt, enc = try_read_text(p)
    if txt is None:
        return 0, False

    occurrences = txt.count(pattern)
    if occurrences == 0:
        return 0, False

    new_txt = txt.replace(pattern, replacement)

    if dry_run:
        return occurrences, True

    # Make backup
    if backup_ext:
        try:
            shutil.copy2(p, p.with_name(p.name + backup_ext))
        except Exception as e:
            print(f"WARNING: Could not create backup for {p}: {e}", file=sys.stderr)

    # Atomic-ish write: write to temp then replace
    tmp_path = p.with_name(p.name + ".tmp___wd")
    write_text(tmp_path, new_txt, enc or "utf-8")
    os.replace(tmp_path, p)

    return occurrences, True

def walk_files(root: Path, recursive: bool):
    if recursive:
        yield from root.rglob("*")
    else:
        yield from root.iterdir()

def main():
    ap = argparse.ArgumentParser(
        description="Replace 'weight_decay = 0.001' with 'weight_decay = 0.001' in files."
    )
    ap.add_argument("folder", type=str, help="Folder to scan (e.g., C:\\Projects\\repo)")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Show what would change, but do not write files")
    ap.add_argument("-r", "--recursive", action="store_true", default=True, help="Recurse into subfolders (default: on)")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false", help="Process only top-level files")
    ap.add_argument("--all-files", action="store_true", help="Consider all file types (not just text-ish extensions)")
    ap.add_argument(
        "--exts",
        type=str,
        default=",".join(DEFAULT_EXTS),
        help=f"Comma-separated list of file extensions to include (default: {', '.join(DEFAULT_EXTS)})",
    )
    ap.add_argument("--backup-ext", type=str, default="", help="Backup extension to append to originals ('' disables)")
    ap.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Exact text to search for")
    ap.add_argument("--replacement", type=str, default=DEFAULT_REPLACEMENT, help="Replacement text")

    args = ap.parse_args()
    root = Path(args.folder).expanduser()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Build the extensions list safely (no stray 's' var)
    raw_exts = [part.strip() for part in args.exts.split(",") if part.strip()]
    exts = tuple(x if x.startswith(".") else f".{x}" for x in raw_exts)

    total_files = 0
    changed_files = 0
    total_replacements = 0

    for p in walk_files(root, args.recursive):
        if not should_process(p, args.all_files, exts):
            continue

        total_files += 1
        occurrences, changed = replace_in_file(
            p,
            pattern=args.pattern,
            replacement=args.replacement,
            dry_run=args.dry_run,
            backup_ext=args.backup_ext,
        )
        if changed:
            action = "WOULD CHANGE" if args.dry_run else "CHANGED"
            print(f"{action}: {p}  ({occurrences} replacement{'s' if occurrences != 1 else ''})")
            changed_files += 1
            total_replacements += occurrences
        # Check \_
        occurrences, changed = replace_in_file(
            p,
            pattern=args.pattern.replace("_", r"\_"),
            replacement=args.replacement.replace("_", r"\_"),
            dry_run=args.dry_run,
            backup_ext=args.backup_ext,
        )
        if changed:
            action = "WOULD CHANGE" if args.dry_run else "CHANGED"
            print(f"{action}: {p}  ({occurrences} replacement{'s' if occurrences != 1 else ''})")
            changed_files += 1
            total_replacements += occurrences

    mode = "DRY RUN (no files modified)" if args.dry_run else "WRITE"
    print("-" * 60)
    print(f"Mode: {mode}")
    print(f"Scanned files: {total_files}")
    print(f"Files with matches: {changed_files}")
    print(f"Total replacements: {total_replacements}")
    if not args.dry_run and args.backup_ext:
        print(f"Backups saved next to originals with extension: {args.backup_ext}")

if __name__ == "__main__":
    main()
