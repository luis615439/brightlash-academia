import os

def check():
    path = "/Volumes/IA_LAB_DAT/SNAPSHOT_500/SURVIVORS_INDEX.md"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    prefixes = []
    for line in lines[6:]:
        if '|' not in line or '---' in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6:
            continue
        
        raw_loc = parts[5].strip('`')
        prefixes.append(raw_loc)
    
    print(f"Total prefixes in index: {len(prefixes)}")
    # Count by Sub_XX
    counts = {}
    for p in prefixes:
        if '/' in p:
            sub = p.split('/')[0]
            counts[sub] = counts.get(sub, 0) + 1
    
    print(f"Counts per folder: {counts}")

if __name__ == "__main__":
    check()
