"""Render manually supplied fire-envelope revisions without changing source data.

Input proposals are human/Codex visual decisions, never detector pseudo-labels.
Generated labels remain pending until overlays and original identities are reviewed.
"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from data_integrity import sha256, validate_yolo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-review", type=Path, required=True)
    p.add_argument("--proposals", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise SystemExit("Refusing overwrite")
    source = json.loads(a.source_review.read_text(encoding="utf-8"))
    proposals = json.loads(a.proposals.read_text(encoding="utf-8"))
    if proposals["source_review_sha256"] != sha256(a.source_review): raise SystemExit("Source review identity changed")
    by_index = {r["index"]: r for r in source["images"]}
    output = []; seen = set(); panels = []
    a.out.mkdir(parents=True); (a.out / "labels").mkdir()
    for proposal in proposals["proposals"]:
        index = proposal["index"]
        if index in seen or index not in by_index: raise SystemExit("Unknown/duplicate proposed index")
        seen.add(index); row = by_index[index]
        if row["split"] != "train" or row["provenance"] != "kitchen_stove_fire": raise SystemExit("Revisions limited to original stove train")
        image, label = Path(row["image_path"]), Path(row["label_path"])
        if sha256(image) != row["sha256"] or sha256(label) != row["label_sha256"]: raise SystemExit("Source bytes/label changed")
        im = Image.open(image).convert("RGB"); w, h = im.size
        if [w, h] != row["size"]: raise SystemExit("Source dimensions changed")
        old = im.copy(); draw = ImageDraw.Draw(old)
        for box in row["boxes_xyxy"]: draw.rectangle(box, outline="#ff55ff", width=3)
        new = im.copy(); draw = ImageDraw.Draw(new)
        decision = proposal["decision"]
        new_boxes = proposal.get("boxes_xyxy")
        if decision == "proposed_revision":
            if not new_boxes or len(new_boxes) != len(row["boxes_xyxy"]): raise SystemExit("This review preserves original burner instance count")
            lines = []
            for box in new_boxes:
                x1, y1, x2, y2 = box
                if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h: raise SystemExit("Invalid proposed pixel box")
                lines.append(f"0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
                draw.rectangle(box, outline="#55ff55", width=3)
            text = "\n".join(lines) + "\n"; validate_yolo(text)
            proposed_label = a.out / "labels" / label.name
            proposed_label.write_text(text, encoding="utf-8")
            new_label_sha = sha256(proposed_label)
        elif decision == "hold" and new_boxes is None:
            new_label_sha = None
        else: raise SystemExit("Unknown decision")
        panel = Image.new("RGB", (1280, 680), "#202020")
        old.thumbnail((640, 640)); new.thumbnail((640, 640))
        panel.paste(old, (0, 30)); panel.paste(new, (640, 30))
        pd = ImageDraw.Draw(panel)
        pd.text((8, 8), f"Source index {index} OLD magenta", fill="white")
        pd.text((648, 8), "PROPOSED green" if decision == "proposed_revision" else "HELD - no new labels", fill="white")
        panels.append(panel)
        output.append({"source_index": index, "image": str(image), "image_sha256": row["sha256"],
                       "original_label": str(label), "original_label_sha256": row["label_sha256"],
                       "proposed_label": str((a.out / "labels" / label.name).resolve()) if new_label_sha else None,
                       "proposed_label_sha256": new_label_sha, "size": [w, h],
                       "old_boxes_xyxy": row["boxes_xyxy"], "new_boxes_xyxy": new_boxes,
                       "proposal": proposal, "approved_for_training": False})
    sheets = []
    for page in range((len(panels)+1)//2):
        sheet = Image.new("RGB", (1280, 1360), "#202020")
        for j, panel in enumerate(panels[page*2:page*2+2]): sheet.paste(panel, (0, j*680))
        path = a.out / f"sheet_{page+1:02d}.jpg"; sheet.save(path, quality=95)
        sheets.append({"file": path.name, "sha256": sha256(path)})
    report = {"role": "pending manual visible-flame-envelope revision; no source mutation, no training",
              "source_review_sha256": sha256(a.source_review), "proposals_sha256": sha256(a.proposals),
              "script_sha256": sha256(Path(__file__)), "policy": proposals["policy"], "sheets": sheets, "images": output}
    (a.out / "review_pending.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(output), "proposed": sum(r["proposed_label_sha256"] is not None for r in output), "sheets": sheets}, indent=2))


if __name__ == "__main__": main()
