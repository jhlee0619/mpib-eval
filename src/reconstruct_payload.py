import json
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="MPIB V2 Payload Reconstructor.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--payload_db", type=str, default=None)
    return parser.parse_args()

def reconstruct_clinical_record(record: dict, registry: dict = None) -> dict:
    evidence_list = record.get("contexts", [])
    if not evidence_list:
        return record

    processed_evidence = []
    for doc in evidence_list:
        content = doc.get("text", "")
        if "[REDACTED_PAYLOAD]" in content:
            template_id = doc.get("template_id")
            if registry and template_id in registry:
                refined_doc = doc.copy()
                refined_doc["text"] = registry[template_id]
                refined_doc["reconstruction_status"] = "restored_fidelity"
                processed_evidence.append(refined_doc)
            else:
                hook = doc.get("reconstruction_hook", {})
                rule = doc.get("rule_family_id", "UnknownRule")
                if hook:
                    meta = hook.get("features", {})
                    word_count = meta.get("words", 10)
                    mock_content = f"[ADVERSARIAL_PAYLOAD_PLACEHOLDER: {rule}] " + " ".join(["(payload)"] * word_count)
                    refined_doc = doc.copy()
                    refined_doc["text"] = mock_content
                    refined_doc["reconstruction_status"] = "structural_mock"
                    processed_evidence.append(refined_doc)
                else:
                    processed_evidence.append(doc)
        else:
            processed_evidence.append(doc)
    record["contexts"] = processed_evidence
    return record

def main():
    args = parse_args()
    if not os.path.exists(args.input_file):
        sys.exit(1)
    registry_map = {}
    if args.payload_db and os.path.exists(args.payload_db):
        with open(args.payload_db, "r", encoding="utf-8") as f:
            registry_map = json.load(f)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.input_file, "r", encoding="utf-8") as f_in, open(args.output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            if record.get("vector") == "V2":
                record = reconstruct_clinical_record(record, registry_map)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("[MPIB] Reconstruction Complete.")

if __name__ == "__main__":
    main()
