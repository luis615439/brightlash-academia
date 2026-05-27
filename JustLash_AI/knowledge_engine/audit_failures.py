import json
import os

INDEX_PATH = "/Volumes/IA_LAB_DAT/SNAPSHOT_500/SURVIVORS_INDEX.md"
IDENTIFIED_ROOT = "/Volumes/IA_LAB_DAT/BIBLIOTECA_IDENTIFICADA"

def audit_failures():
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    index_entries = []
    for line in lines[6:]:
        if '|' not in line or '---' in line: continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6: continue
        
        raw_loc = parts[5].strip('`')
        if not raw_loc or raw_loc == '...': continue
        
        filename_part = raw_loc.split('/')[-1] if '/' in raw_loc else raw_loc
        prefix = filename_part.split('.')[0].replace('...', '')
        subfolder = raw_loc.split('/')[0] if '/' in raw_loc else "Unknown"
        
        index_entries.append({
            "id": parts[1],
            "title": parts[2],
            "prefix": prefix,
            "subfolder": subfolder,
            "raw_loc": raw_loc
        })

    # Get all physical files
    physical_files = {}
    for root, dirs, files in os.walk(IDENTIFIED_ROOT):
        parent = os.path.basename(root)
        for f in files:
            if f.startswith('.'): continue
            physical_files[f] = parent

    report = {
        "missing_physical": [],
        "prefix_mismatch": [],
        "ambiguous": []
    }

    found_prefixes = set()
    
    for entry in index_entries:
        matches = [f for f in physical_files if f.startswith(entry['prefix'])]
        
        if not matches:
            report["missing_physical"].append(entry)
        elif len(matches) > 1:
            report["ambiguous"].append({"entry": entry, "matches": matches})
        else:
            # Check if subfolder matches
            if physical_files[matches[0]] != entry['subfolder'] and entry['subfolder'] != "Unknown":
                report["prefix_mismatch"].append({"entry": entry, "found_in": physical_files[matches[0]]})
            else:
                found_prefixes.add(entry['prefix'])

    return report, len(index_entries)

if __name__ == "__main__":
    report, total = audit_failures()
    print(f"Audit of {total} entries completed.")
    print(f"Missing Physical: {len(report['missing_physical'])}")
    print(f"Ambiguous: {len(report['ambiguous'])}")
    print(f"Folder Mismatch: {len(report['prefix_mismatch'])}")
    
    with open("mision_diamante_failures.json", "w") as f:
        json.dump(report, f, indent=2)
