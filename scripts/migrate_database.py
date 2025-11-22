"""
Database migration script to convert single-embedding format to multi-embedding format.
Creates backup before migration and supports rollback.
"""

import torch
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.config import EMPLOYEE_DB_PATH, OUTPUT_DIR


def backup_database(db_path):
    """
    Create backup of database file.
    
    Args:
        db_path: Path to database file
    
    Returns:
        Path to backup file
    """
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
    
    try:
        shutil.copy2(db_path, backup_path)
        print(f"✓ Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"✗ Error creating backup: {e}")
        return None


def detect_database_format(db_path):
    """
    Detect if database is in old (single embedding) or new (multi embedding) format.
    
    Args:
        db_path: Path to database file
    
    Returns:
        str: "single", "multi", "mixed", or "unknown"
    """
    try:
        db = torch.load(db_path, weights_only=False)
        
        if not isinstance(db, dict) or len(db) == 0:
            return "unknown"
        
        # Check first few entries
        formats = set()
        for name, value in list(db.items())[:5]:
            if isinstance(value, list):
                formats.add("multi")
            elif isinstance(value, torch.Tensor) and value.dim() >= 1:
                formats.add("single")
            else:
                formats.add("unknown")
        
        if "unknown" in formats:
            return "unknown"
        elif len(formats) == 1:
            return formats.pop()
        else:
            return "mixed"
            
    except Exception as e:
        print(f"Error detecting format: {e}")
        return "unknown"


def validate_database(db):
    """
    Validate database structure and contents.
    
    Args:
        db: Database dictionary
    
    Returns:
        tuple: (is_valid: bool, issues: list)
    """
    issues = []
    
    if not isinstance(db, dict):
        issues.append("Database is not a dictionary")
        return (False, issues)
    
    if len(db) == 0:
        issues.append("Database is empty")
        return (False, issues)
    
    for name, embeddings in db.items():
        if not isinstance(name, str):
            issues.append(f"Invalid name type: {type(name)}")
        
        if isinstance(embeddings, list):
            for i, emb in enumerate(embeddings):
                if not isinstance(emb, torch.Tensor):
                    issues.append(f"{name}: embedding {i} is not a tensor")
                elif emb.dim() != 2 or emb.shape[0] != 1:
                    issues.append(f"{name}: embedding {i} has invalid shape {emb.shape}")
                elif torch.isnan(emb).any():
                    issues.append(f"{name}: embedding {i} contains NaN values")
        elif isinstance(embeddings, torch.Tensor):
            if embeddings.dim() != 2 or embeddings.shape[0] != 1:
                issues.append(f"{name}: embedding has invalid shape {embeddings.shape}")
            elif torch.isnan(embeddings).any():
                issues.append(f"{name}: embedding contains NaN values")
        else:
            issues.append(f"{name}: invalid embedding type {type(embeddings)}")
    
    is_valid = len(issues) == 0
    return (is_valid, issues)


def migrate_single_to_multi(db_path, backup_path):
    """
    Migrate database from single-embedding to multi-embedding format.
    
    Args:
        db_path: Path to database file
        backup_path: Path to backup file
    
    Returns:
        bool: Success status
    """
    try:
        # Load database
        db = torch.load(db_path, weights_only=False)
        
        # Validate before migration
        is_valid, issues = validate_database(db)
        if not is_valid:
            print(f"✗ Database validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        # Convert single embeddings to lists
        migrated_count = 0
        for name, embedding in db.items():
            if isinstance(embedding, torch.Tensor):
                # Convert single tensor to list
                db[name] = [embedding]
                migrated_count += 1
        
        if migrated_count == 0:
            print("No entries needed migration (already in multi-embedding format)")
            return True
        
        # Validate after migration
        is_valid, issues = validate_database(db)
        if not is_valid:
            print(f"✗ Post-migration validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        # Save migrated database
        torch.save(db, db_path)
        print(f"✓ Migrated {migrated_count} employees to multi-embedding format")
        
        # Add version marker file
        version_file = db_path.parent / "db_version.txt"
        with open(version_file, 'w') as f:
            f.write(f"multi\n{datetime.now().isoformat()}\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        return False


def rollback_migration(db_path, backup_path):
    """
    Rollback migration by restoring from backup.
    
    Args:
        db_path: Path to database file
        backup_path: Path to backup file
    
    Returns:
        bool: Success status
    """
    try:
        if not backup_path.exists():
            print(f"✗ Backup not found: {backup_path}")
            return False
        
        shutil.copy2(backup_path, db_path)
        print(f"✓ Restored database from backup: {backup_path}")
        return True
        
    except Exception as e:
        print(f"✗ Rollback failed: {e}")
        return False


def main():
    """Main migration workflow."""
    print("=" * 70)
    print("DATABASE MIGRATION: Single-Embedding → Multi-Embedding")
    print("=" * 70)
    
    # Check if database exists
    if not EMPLOYEE_DB_PATH.exists():
        print(f"\n✗ Database not found: {EMPLOYEE_DB_PATH}")
        print("No migration needed. Database will be created in new format.")
        return
    
    # Detect current format
    print(f"\nDatabase: {EMPLOYEE_DB_PATH}")
    db_format = detect_database_format(EMPLOYEE_DB_PATH)
    print(f"Current format: {db_format}")
    
    if db_format == "multi":
        print("\n✓ Database already in multi-embedding format. No migration needed.")
        return
    elif db_format == "unknown":
        print("\n✗ Unable to detect database format. Manual inspection required.")
        return
    elif db_format == "mixed":
        print("\n⚠ Database has mixed format. Proceeding with migration...")
    elif db_format == "single":
        print("\n→ Migrating from single-embedding to multi-embedding format...")
    
    # Create backup
    print("\nStep 1: Creating backup...")
    backup_path = backup_database(EMPLOYEE_DB_PATH)
    if not backup_path:
        print("✗ Backup failed. Aborting migration.")
        return
    
    # Perform migration
    print("\nStep 2: Migrating database...")
    success = migrate_single_to_multi(EMPLOYEE_DB_PATH, backup_path)
    
    if success:
        print("\n" + "=" * 70)
        print("✓ MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Backup saved at: {backup_path}")
        print("\nYou can now use the new multi-embedding registration system.")
        print(f"To rollback, run: python {__file__} --rollback {backup_path}")
    else:
        print("\n" + "=" * 70)
        print("✗ MIGRATION FAILED")
        print("=" * 70)
        print(f"Database unchanged. Backup available at: {backup_path}")
        print("\nTo rollback, run: python {__file__} --rollback {backup_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate employee database format")
    parser.add_argument("--rollback", type=str, help="Path to backup file for rollback")
    args = parser.parse_args()
    
    if args.rollback:
        print("=" * 70)
        print("DATABASE ROLLBACK")
        print("=" * 70)
        backup_path = Path(args.rollback)
        success = rollback_migration(EMPLOYEE_DB_PATH, backup_path)
        if success:
            print("\n✓ Rollback completed successfully")
        else:
            print("\n✗ Rollback failed")
    else:
        main()
