"""Bounded archive extraction to a new dataset directory; no input deletion."""
import shutil
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath


def checked_members(archive, root, max_bytes=8_000_000_000, max_members=50_000):
    root = Path(root).resolve()
    members = archive.infolist()
    if len(members) > max_members or sum(i.file_size for i in members) > max_bytes:
        raise ValueError("Archive exceeds extraction bounds")
    seen = set(); result = []
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for info in members:
        name = info.filename
        parts = PurePosixPath(name).parts
        win = PureWindowsPath(name)
        if not parts or name.startswith("/") or "\\" in name or win.drive or win.is_absolute():
            raise ValueError("Absolute or nonportable archive path")
        if any(p in (".", "..") or ":" in p or p.endswith((".", " ")) or p.split(".")[0].upper() in reserved for p in parts):
            raise ValueError("Unsafe archive path component")
        target = root.joinpath(*parts).resolve()
        if root not in target.parents:
            raise ValueError("Archive path escapes extraction root")
        key = str(target).casefold()
        if key in seen: raise ValueError("Duplicate or case-colliding archive member")
        seen.add(key)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode): raise ValueError("Archive symlink rejected")
        if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("Archive special file rejected")
        result.append((info, target))
    return result


def extract_new_dataset(archive, root, max_bytes=8_000_000_000, max_members=50_000):
    root = Path(root).resolve()
    if root.exists(): raise ValueError("Extraction directory already exists")
    members = checked_members(archive, root, max_bytes, max_members)
    root.mkdir(parents=True)
    for info, target in members:
        if info.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # ZipExtFile verifies CRC when fully read; exclusive file creation prevents overwrite.
        with archive.open(info) as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    return {"members": len(members), "uncompressed_bytes": sum(i.file_size for i, _ in members)}
