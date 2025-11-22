"""
Fix employee database to ensure all embeddings are in list format.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from pathlib import Path

db_path = Path("outputs/employee_db.pt")

if db_path.exists():
    print("Loading database...")
    db = torch.load(db_path, weights_only=False)
    
    print(f"Found {len(db)} employees")
    
    fixed = False
    for name, data in db.items():
        if isinstance(data, torch.Tensor):
            print(f"  Fixing {name}: tensor → [tensor]")
            db[name] = [data]
            fixed = True
        elif isinstance(data, list):
            print(f"  {name}: already in list format ({len(data)} embeddings)")
        else:
            print(f"  {name}: unknown format {type(data)}")
    
    if fixed:
        print("\nSaving fixed database...")
        torch.save(db, db_path)
        print("✓ Database fixed!")
    else:
        print("\n✓ Database already in correct format!")
else:
    print("No database found")
