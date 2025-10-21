# save_vqav2_images_all_fields.py
import os
import json
import csv
from datasets import load_dataset
from PIL import Image

INPUT_JSON = "vqav2_filtered_categorized_checkpoint_7200.json"  # your file path
OUT_DIR = "vqav2_images"          # where to save images
MAP_CSV  = "vqav2_rows_with_images.csv"  # <-- CSV with all dataset fields (+ saved_image_path)

def read_question_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qids = set()
    for v in data.values():
        if isinstance(v, list):
            for item in v:
                qid = item.get("question_id")
                if isinstance(qid, int):
                    qids.add(qid)
    return qids

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    wanted_qids = read_question_ids(INPUT_JSON)
    print(f"Found {len(wanted_qids)} question_ids in {INPUT_JSON}")
    if not wanted_qids:
        print("No question_ids found; exiting.")
        return

    # Load VQAv2 (validation split) from HF
    ds = load_dataset("lmms-lab/VQAv2", split="validation")

    # Filter to your ids
    wanted_qids_set = set(wanted_qids)
    sub = ds.filter(lambda ex: ex["question_id"] in wanted_qids_set)
    print(f"Matched {len(sub)} records in VQAv2 for your question_ids.")

    # Determine CSV columns: all dataset columns except 'image'
    csv_columns = [c for c in sub.column_names if c != "image"]
    # We'll append a new column with where we saved the file
    csv_columns.append("saved_image_path")

    # Write CSV rows as we save images
    with open(MAP_CSV, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=csv_columns)
        writer.writeheader()

        seen_qids = set()
        for ex in sub:
            qid = ex["question_id"]
            pil_img = ex["image"]

            out_path = os.path.join(OUT_DIR, f"{qid}.jpg")
            if isinstance(pil_img, Image.Image):
                if pil_img.mode not in ("RGB", "L"):
                    pil_img = pil_img.convert("RGB")
                pil_img.save(out_path, format="JPEG", quality=95)
            else:
                Image.fromarray(pil_img).save(out_path, format="JPEG", quality=95)

            # Build a row with all dataset fields (except image)
            row = {k: ex[k] for k in sub.column_names if k != "image"}

            # Normalize for CSV (ints/strings fine; lists get JSON-encoded if present)
            for k, v in list(row.items()):
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v, ensure_ascii=False)

            row["saved_image_path"] = out_path.replace("\\", "/")
            writer.writerow(row)
            seen_qids.add(qid)

    # Report any question_ids that weren't matched
    missing = sorted(wanted_qids - seen_qids)
    if missing:
        with open("missing_question_ids.txt", "w", encoding="utf-8") as f:
            for q in missing:
                f.write(str(q) + "\n")
        print(f"WARNING: {len(missing)} question_ids were not found; see missing_question_ids.txt")

    print(f"Saved CSV with {len(sub)} rows to '{MAP_CSV}'")
    print(f"Images saved under '{OUT_DIR}/'")

if __name__ == "__main__":
    main()
