"""Unit tests for the Pexels/v20 source identity guard; no audit inputs are opened."""
import subprocess
import sys
import unittest
from pathlib import Path

from audit_pexels_9685496_v20_dedupe_20261029 import (
    EXPECTED_SAMPLE_MANIFEST_SHA256,
    EXPECTED_SOURCE_SHA256,
    validate_sampling_manifest,
)


class SamplingManifestIdentityTests(unittest.TestCase):
    def test_identity_import_and_validation_work_without_vision_packages(self):
        source_dir = str(Path(__file__).resolve().parent)
        child = r"""
import importlib.abc, sys
class BlockVisionPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cv2', 'numpy'}:
            raise ModuleNotFoundError('blocked optional vision package: ' + fullname)
sys.meta_path.insert(0, BlockVisionPackages())
sys.path.insert(0, sys.argv[1])
from audit_pexels_9685496_v20_dedupe_20261029 import EXPECTED_SAMPLE_MANIFEST_SHA256, EXPECTED_SOURCE_SHA256, validate_sampling_manifest
manifest = {'source_sha256_before_decode': EXPECTED_SOURCE_SHA256, 'source_sha256_after_decode': EXPECTED_SOURCE_SHA256}
validate_sampling_manifest(manifest, EXPECTED_SAMPLE_MANIFEST_SHA256, EXPECTED_SOURCE_SHA256)
"""
        subprocess.run(
            [sys.executable, "-B", "-c", child, source_dir],
            check=True,
            capture_output=True,
            text=True,
        )

    def valid_manifest(self):
        return {
            "source_sha256_before_decode": EXPECTED_SOURCE_SHA256,
            "source_sha256_after_decode": EXPECTED_SOURCE_SHA256,
        }

    def test_accepts_pinned_manifest_and_matching_source_identities(self):
        validate_sampling_manifest(
            self.valid_manifest(),
            EXPECTED_SAMPLE_MANIFEST_SHA256,
            EXPECTED_SOURCE_SHA256,
        )

    def test_rejects_wrong_manifest_sha256(self):
        with self.assertRaisesRegex(ValueError, "sampling_manifest SHA-256 mismatch"):
            validate_sampling_manifest(
                self.valid_manifest(),
                "0" * 64,
                EXPECTED_SOURCE_SHA256,
            )

    def test_rejects_before_decode_source_misbinding(self):
        manifest = self.valid_manifest()
        manifest["source_sha256_before_decode"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_sha256_before_decode"):
            validate_sampling_manifest(
                manifest,
                EXPECTED_SAMPLE_MANIFEST_SHA256,
                EXPECTED_SOURCE_SHA256,
            )

    def test_rejects_after_decode_source_misbinding(self):
        manifest = self.valid_manifest()
        manifest["source_sha256_after_decode"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_sha256_after_decode"):
            validate_sampling_manifest(
                manifest,
                EXPECTED_SAMPLE_MANIFEST_SHA256,
                EXPECTED_SOURCE_SHA256,
            )


if __name__ == "__main__":
    unittest.main()