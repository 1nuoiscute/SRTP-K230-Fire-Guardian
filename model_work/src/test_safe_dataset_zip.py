import io
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from safe_dataset_zip import checked_members, extract_new_dataset


class SafeZipTests(unittest.TestCase):
    def archive(self, names):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in names:
                archive.writestr(name, b"dataset payload")
        return zipfile.ZipFile(buffer)

    def test_no_escape_or_windows_ambiguous_paths(self):
        for name in ("../outside", "/outside", "C:/outside", "folder\\outside", "x/file:stream", "x/CON.txt", "x/file.", "x/file "):
            with self.subTest(name=name), self.archive([name]) as archive:
                # Windows ZipInfo normalizes backslashes on fixture construction.
                # Restore the externally supplied name to test the validator itself.
                archive.infolist()[0].filename = name
                with self.assertRaises(ValueError): checked_members(archive, Path("output"))

    def test_case_collision_symlink_and_bounds_rejected(self):
        with self.archive(["a.txt", "A.txt"]) as archive:
            with self.assertRaises(ValueError): checked_members(archive, Path("output"))
        with self.archive(["link"]) as archive:
            archive.infolist()[0].external_attr = (stat.S_IFLNK | 0o777) << 16
            with self.assertRaises(ValueError): checked_members(archive, Path("output"))
        with self.archive(["a.txt"]) as archive:
            with self.assertRaises(ValueError): checked_members(archive, Path("output"), max_bytes=1)
            with self.assertRaises(ValueError): checked_members(archive, Path("output"), max_members=0)

    def test_new_directory_only_and_actual_extraction(self):
        with tempfile.TemporaryDirectory() as tmp, self.archive(["nested/images/a.jpg", "nested/labels/a.txt"]) as archive:
            root = Path(tmp) / "dataset"
            report = extract_new_dataset(archive, root)
            self.assertEqual(report["members"], 2)
            self.assertEqual((root / "nested/images/a.jpg").read_bytes(), b"dataset payload")
            with self.assertRaises(ValueError): extract_new_dataset(archive, root)
            self.assertEqual((root / "nested/images/a.jpg").read_bytes(), b"dataset payload")


if __name__ == "__main__": unittest.main()
