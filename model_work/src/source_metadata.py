"""Bind reviewed source identities while recording justified metadata corrections."""


def checked_metadata_corrections(reviewed, acquired):
    for key in ("sha256", "source_page", "width", "height", "original_sha1", "license"):
        if reviewed[key] != acquired[key]:
            raise ValueError(f"Reviewed source identity differs: {reviewed['file']}/{key}")
    corrections = {}
    basis = reviewed.get("author_verification", "")
    raw_url = acquired.get("license_url", "")
    review_url = reviewed.get("license_url", "")
    if raw_url != review_url:
        if raw_url or reviewed["license"] != "Public domain" or review_url != reviewed["source_page"] + "#Licensing" or not basis:
            raise ValueError("License URL correction lacks the reviewed original public-domain release")
        corrections["license_url"] = {"raw": raw_url, "reviewed": review_url, "basis": basis}
    if reviewed.get("author", "") != acquired.get("author", ""):
        if not reviewed.get("author") or not basis:
            raise ValueError("Author correction lacks a recorded source verification")
        corrections["author"] = {"raw": acquired.get("author", ""), "reviewed": reviewed["author"], "basis": basis}
    return corrections
