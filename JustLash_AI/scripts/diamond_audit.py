import sqlite3
import os
import requests
import sys

DB_PATH = "/Users/joseluis/Downloads/Claude Code/YTOPENROUTER/JustLash_AI/knowledge_engine/diamond_kb.db"
LIBRARY_PATH = "/Volumes/IA_LAB_DAT/IA_LAB_KNOWLEDGE_BASE"
API_BASE = "http://localhost:8000/api"

def audit():
    print("💎 DIAMOND VAULT AUDIT SYSTEM 💎")
    print("-" * 30)

    # 1. Database Check
    if not os.path.exists(DB_PATH):
        print("❌ Database not found at", DB_PATH)
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM files")
    total_files = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM files WHERE text_extracted = 1")
    indexed_files = cursor.fetchone()[0]
    
    print(f"📊 Database Stats:")
    print(f"   - Total files in DB: {total_files}")
    print(f"   - Files with text extracted: {indexed_files}")
    print(f"   - Progress: {(indexed_files/total_files*100):.2f}%" if total_files > 0 else "   - Progress: 0%")

    # 2. Check for zeroed-out files (Sampling)
    print("\n🔍 Corrupted File Detection (Sampling 20 files)...")
    cursor.execute("SELECT current_path FROM files LIMIT 20")
    samples = cursor.fetchall()
    corrupted = 0
    for (path,) in samples:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                head = f.read(100)
                if all(b == 0 for b in head) and len(head) > 0:
                    corrupted += 1
                    # print(f"   ⚠️ Corrupted: {os.path.basename(path)}")
        else:
            print(f"   ❌ Missing physically: {os.path.basename(path)}")
    
    if corrupted > 0:
        print(f"   ⚠️ WARNING: Found {corrupted}/20 sampled files with NULL bytes (corrupted/zeroed out).")
        print("      This explains why text extraction fails for these files.")
    else:
        print("   ✅ Sampled files look healthy (non-zero headers).")

    # 3. API Check
    print("\n🌐 API Connectivity Check...")
    try:
        resp = requests.get(f"{API_BASE}/stats", timeout=5)
        if resp.status_code == 200:
            print("   ✅ Backend API is ONLINE.")
            stats = resp.json()
            print(f"   ✅ Backend reports {stats.get('total_files')} files and {len(stats.get('categories', []))} categories.")
        else:
            print(f"   ❌ Backend API returned status {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Backend API is OFFLINE: {e}")

    # 4. Search Functionality Check
    print("\n🔎 Search Functionality Test...")
    try:
        test_query = "estrategia"
        resp = requests.post(f"{API_BASE}/search", json={"text": test_query, "top_k": 3}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ Search working. Found {len(data.get('sources', []))} sources.")
            if not data.get('sources'):
                print("   ⚠️ Search returned 0 sources. This might be because indexing is still in progress or files are corrupted.")
        else:
            print(f"   ❌ Search failed with status {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Search test failed: {e}")

    conn.close()
    print("-" * 30)
    print("Audit Complete.")

if __name__ == "__main__":
    audit()
