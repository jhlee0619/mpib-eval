import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="MPIB V2 Payload Reconstructor")
    parser.add_argument("--input_file", type=str, required=True, help="Path to redacted .jsonl file (e.g., test.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save reconstructed .jsonl file")
    parser.add_argument("--payload_db", type=str, default=None, help="Path to payload_registry_v1.x.json (e.g., data/restricted/payload_registry_v1.x.json)")
    return parser.parse_args()

def reconstruct_sample(sample, payload_db=None):
    """
    Reconstructs the redacted V2 payload.
    - If payload_db is provided: Exact restoration (Tier 1).
    - If payload_db is None: Structural placeholder restoration (Tier 0).
    """
    contexts = sample.get("contexts", [])
    reconstructed_contexts = []
    
    for ctx in contexts:
        text = ctx.get("text", "")
        
        # Check if this context is redacted
        if "[REDACTED_PAYLOAD]" in text:
            template_id = ctx.get("template_id")
            
            # Mode A: Exact Restoration (if DB is present)
            if payload_db and template_id in payload_db:
                ctx_new = ctx.copy()
                ctx_new["text"] = payload_db[template_id]
                ctx_new["reconstructed_method"] = "exact"
                reconstructed_contexts.append(ctx_new)
                
            # Mode B: Structural Placeholder (Public)
            else:
                hook = ctx.get("reconstruction_hook", {})
                rule_family = ctx.get("rule_family_id")
                
                if hook and rule_family:
                    feat = hook.get("features", {})
                    num_words = feat.get("words", 10)
                    
                    # Mock reconstruction for structural validity
                    reconstructed_text = f"[RECONSTRUCTED_PAYLOAD: {rule_family} | TID: {template_id}] "
                    reconstructed_text += " ".join(["(payload)"] * num_words)
                    
                    ctx_new = ctx.copy()
                    ctx_new["text"] = reconstructed_text
                    ctx_new["reconstructed_method"] = "structural_placeholder"
                    reconstructed_contexts.append(ctx_new)
                else:
                    # Fallback (Should not happen if metadata is intact)
                    reconstructed_contexts.append(ctx)
        else:
            reconstructed_contexts.append(ctx)
            
    sample["contexts"] = reconstructed_contexts
    return sample

def main():
    args = parse_args()
    
    # Load Payload DB if provided
    payload_mapping = {}
    if args.payload_db:
        if os.path.exists(args.payload_db):
            print(f"[Info] Loading Payload DB from {args.payload_db}...")
            with open(args.payload_db, "r", encoding="utf-8") as f:
                payload_mapping = json.load(f)
            print(f"       Loaded {len(payload_mapping)} entries.")
        else:
            print(f"[Warn] Payload DB not found at {args.payload_db}. Using strictly structural reconstruction.")

    print(f"Reconstructing payloads from {args.input_file}...")
    
    count = 0
    restored_exact = 0
    
    with open(args.input_file, "r", encoding="utf-8") as fin, open(args.output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            sample = json.loads(line)
            
            if sample.get("vector") == "V2":
                sample = reconstruct_sample(sample, payload_mapping)
                # Check status for logging
                msg = sample["contexts"][-1].get("reconstructed_method", "")
                if msg == "exact": restored_exact += 1
            
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"Done. Processed {count} samples.")
    if args.payload_db and restored_exact > 0:
        print(f"✅ Exact Restoration Count: {restored_exact} samples")
    else:
        print(f"ℹ️  Structural Placeholder Mode used. (To perform exact restoration, provide --payload_db)")

if __name__ == "__main__":
    main()
