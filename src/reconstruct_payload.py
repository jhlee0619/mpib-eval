import json
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="MPIB V2 Payload Reconstructor: Restores adversarial clinical evidence.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to redacted dataset (.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save reconstructed dataset (.jsonl)")
    parser.add_argument("--payload_db", type=str, default=None, help="Path to secure payload_registry.json")
    return parser.parse_args()

def reconstruct_clinical_record(record: dict, registry: dict = None) -> dict:
    """
    Restores redacted adversarial clinical evidence in a V2 record.
    
    Tier 1 (Registry Provided): Restores exact adversarial payloads for clinical safety audits.
    Tier 0 (Public Mode): Restores structural placeholders to verify RAG engine logic.
    """
    evidence_list = record.get("contexts", [])
    if not evidence_list:
        return record

    processed_evidence = []
    for doc in evidence_list:
        content = doc.get("text", "")
        
        # Identify redacted adversarial updates
        if "[REDACTED_PAYLOAD]" in content:
            template_id = doc.get("template_id")
            
            # Tier 1: High-fidelity restoration from secure registry
            if registry and template_id in registry:
                refined_doc = doc.copy()
                refined_doc["text"] = registry[template_id]
                refined_doc["reconstruction_status"] = "restored_fidelity"
                processed_evidence.append(refined_doc)
                
            # Tier 0: Structural mock for public pipelines
            else:
                hook = doc.get("reconstruction_hook", {})
                rule = doc.get("rule_family_id", "UnknownRule")
                
                if hook:
                    meta = hook.get("features", {})
                    word_count = meta.get("words", 10)
                    
                    # Generate structural mock preserving word count and attack vectors
                    mock_content = f"[ADVERSARIAL_PAYLOAD_PLACEHOLDER: {rule} | TID: {template_id}] "
                    mock_content += " ".join(["(payload)"] * word_count)
                    
                    refined_doc = doc.copy()
                    refined_doc["text"] = mock_content
                    refined_doc["reconstruction_status"] = "structural_mock"
                    processed_evidence.append(refined_doc)
                else:
                    processed_evidence.append(doc)
        else:
            # Benign or pre-restored evidence
            processed_evidence.append(doc)
            
    record["contexts"] = processed_evidence
    return record

def main():
    args = parse_args()
    
    # 1. Input Verification
    if not os.path.exists(args.input_file):
        print(f"[Error] Clinical record file not found: {args.input_file}")
        sys.exit(1)

    registry_map = {}
    if args.payload_db:
        if os.path.exists(args.payload_db):
            print(f"[MPIB] Syncing with secure payload registry: {args.payload_db}")
            with open(args.payload_db, "r", encoding="utf-8") as f:
                registry_map = json.load(f)
            print(f"[MPIB] Registry Sync Successful: {len(registry_map)} templates indexed.")
        else:
            print(f"[Notice] Payload registry missing at {args.payload_db}. Protocol: Tier 0 (Structural Mock)")

    # 2. Batch Processing Pipeline
    print(f"[MPIB] Initializing reconstruction pipeline for {args.input_file}...")
    
    processed_count = 0
    restoration_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    try:
        with open(args.input_file, "r", encoding="utf-8") as f_in, \
             open(args.output_file, "w", encoding="utf-8") as f_out:
            
            for line in f_in:
                if not line.strip(): continue
                record = json.loads(line)
                
                # Filter for V2 adversarial vectors
                if record.get("vector") == "V2":
                    record = reconstruct_clinical_record(record, registry_map)
                    
                    # Audit status tracking
                    for doc in record.get("contexts", []):
                        if doc.get("reconstruction_status") == "restored_fidelity":
                            restoration_count += 1
                            break
                
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed_count += 1
                
        print(f"[MPIB] Reconstruction Audit Complete. Logs: {processed_count} records processed.")
        if restoration_count > 0:
            print(f"       Status: Full Fidelity Restoration ({restoration_count} records)")
        else:
            print(f"       Status: Structural Integrity Verified (Public Tier)")
            
    except Exception as e:
        print(f"[Critical] Reconstruction pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
