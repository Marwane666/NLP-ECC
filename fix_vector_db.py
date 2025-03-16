"""
Utility script to fix file locking issues with the vector database.
This helps resolve problems with "file in use by another process" errors.
"""
import os
import sys
import shutil
import time
import yaml
import argparse
from load_env import load_env_file

def close_vector_db_connections(config_path: str = "config.yaml"):
    """
    Close any open connections to the vector database and fix file permissions.
    
    Args:
        config_path: Path to the configuration file
    """
    # First load environment variables
    load_env_file(verbose=False)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vector_db_dir = config['vector_db_directory']
    
    # Check if vector DB exists
    if not os.path.exists(vector_db_dir):
        print(f"Vector database directory {vector_db_dir} does not exist.")
        return False
    
    # Check for the SQLite database file
    db_file = os.path.join(vector_db_dir, "chroma.sqlite3")
    if not os.path.exists(db_file):
        print(f"SQLite database file not found at {db_file}")
        return False
    
    print(f"Attempting to fix file locking issues on {db_file}")
    print("This will create a backup of your vector database and recreate it.")
    
    # Create a backup
    backup_dir = f"{vector_db_dir}_backup_{int(time.time())}"
    try:
        print(f"Creating backup at {backup_dir}")
        shutil.copytree(vector_db_dir, backup_dir)
        print("Backup created successfully.")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Try to fix the database file
    try:
        # Attempt to recreate the vector database directory
        print(f"Removing {vector_db_dir} to clear file locks...")
        try:
            # First try to clear just the lock files
            for lock_file in [
                os.path.join(vector_db_dir, "chroma.sqlite3-shm"),
                os.path.join(vector_db_dir, "chroma.sqlite3-wal")
            ]:
                if os.path.exists(lock_file):
                    try:
                        os.remove(lock_file)
                        print(f"Removed lock file: {lock_file}")
                    except:
                        print(f"Could not remove lock file: {lock_file}")
            
            # If that fails, try to remove the entire directory
            shutil.rmtree(vector_db_dir)
            print(f"Removed {vector_db_dir}")
        except Exception as e:
            print(f"Error removing directory: {e}")
            print("You may need to close any applications that are accessing the database.")
            print("If you're running a Flask server, restart it after running this script.")
            return False
        
        # Recreate the directory
        os.makedirs(vector_db_dir, exist_ok=True)
        
        # Copy the backup files back, excluding lock files
        print(f"Restoring database from backup...")
        for item in os.listdir(backup_dir):
            # Skip SQLite lock files
            if item.endswith("-shm") or item.endswith("-wal"):
                continue
                
            src = os.path.join(backup_dir, item)
            dst = os.path.join(vector_db_dir, item)
            
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        print("Database restored successfully.")
        return True
        
    except Exception as e:
        print(f"Error fixing database: {e}")
        print("Restoring from backup...")
        try:
            if os.path.exists(vector_db_dir):
                shutil.rmtree(vector_db_dir)
            shutil.copytree(backup_dir, vector_db_dir)
            print("Restored from backup.")
        except Exception as restore_error:
            print(f"Error restoring from backup: {restore_error}")
            print(f"Your backup is still available at {backup_dir}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix vector database file locking issues')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    if close_vector_db_connections(args.config):
        print("\nDatabase file locking issues have been fixed.")
        print("You can now restart your application.")
    else:
        print("\nFailed to fix database file locking issues.")
        print("Try closing any applications that might be using the database or restart your computer.")
