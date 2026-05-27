import os
import json

IDENTIFIED_ROOT = "/Volumes/IA_LAB_DAT/BIBLIOTECA_IDENTIFICADA"
INDEX_PATH = "/Volumes/IA_LAB_DAT/SNAPSHOT_500/SURVIVORS_INDEX.md"

def debug():
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    index_keys = []
    for line in lines[6:]:
        if '|' not in line: continue
        parts = line.split('|')
        if len(parts) < 6: continue
        index_keys.append(parts[5].strip().strip('`'))
    
    print(f"Sample Index Keys: {index_keys[:5]}")
    
    found_count = 0
    for root, dirs, files in os.walk(IDENTIFIED_ROOT):
        for filename in files:
            parent = os.path.basename(root)
            rel_loc = f"{parent}/{filename}"
            if rel_loc in index_keys:
                found_count += 1
            if found_count < 5 and rel_loc in index_keys:
                print(f"Match found: {rel_loc}")
    
    print(f"Total found: {found_count} out of {len(index_keys)}")

if __name__ == "__main__":
    debug()
