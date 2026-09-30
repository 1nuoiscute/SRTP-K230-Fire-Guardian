import copy
import unittest
from source_metadata import checked_metadata_corrections


class SourceMetadataTests(unittest.TestCase):
    def setUp(self):
        self.raw = {"file": "source.jpg", "sha256": "bound-image", "source_page": "https://commons.wikimedia.org/wiki/File:Example.jpg",
                    "width": 20, "height": 10, "original_sha1": "bound-original", "license": "Public domain", "license_url": "", "author": "raw display name"}
        self.review = copy.deepcopy(self.raw)

    def test_unchanged_metadata_has_no_corrections(self):
        self.assertEqual(checked_metadata_corrections(self.review, self.raw), {})

    def test_verified_empty_public_domain_link_and_author_recorded(self):
        self.review.update(license_url=self.raw["source_page"] + "#Licensing", author="Verified username", author_verification="Viewed original author and public-domain release")
        corrections = checked_metadata_corrections(self.review, self.raw)
        self.assertEqual(set(corrections), {"license_url", "author"})
        self.assertEqual(corrections["license_url"]["raw"], "")
        self.assertEqual(corrections["author"]["raw"], self.raw["author"])

    def test_arbitrary_license_url_rejected(self):
        self.review.update(license_url="https://example.com/permission", author_verification="verified")
        with self.assertRaisesRegex(ValueError, "License URL correction"):
            checked_metadata_corrections(self.review, self.raw)

    def test_missing_verification_rejected(self):
        self.review["author"] = "different"
        with self.assertRaisesRegex(ValueError, "Author correction"):
            checked_metadata_corrections(self.review, self.raw)

    def test_existing_license_url_cannot_be_replaced(self):
        self.raw["license_url"] = "https://original.example/license"
        self.review.update(license_url=self.raw["source_page"] + "#Licensing", author_verification="verified")
        with self.assertRaisesRegex(ValueError, "License URL correction"):
            checked_metadata_corrections(self.review, self.raw)

    def test_bytes_or_license_identity_cannot_be_changed(self):
        for field in ("sha256", "license"):
            with self.subTest(field=field):
                reviewed = copy.deepcopy(self.review)
                reviewed[field] = "changed"
                with self.assertRaisesRegex(ValueError, "source identity differs"):
                    checked_metadata_corrections(reviewed, self.raw)
