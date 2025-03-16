"""
Wrapper script for the RAG system CLI.
This script loads environment variables from .env file and then runs the CLI.
"""
import os
import sys
import subprocess
from load_env import load_env_file

def main():
    """Load environment variables and run the CLI command."""
    # First try to load from .env file
    env_vars = load_env_file(verbose=False)
    
    if not env_vars.get('GITHUB_TOKEN'):
        print("Warning: GITHUB_TOKEN not found in .env file.")
        print("Checking if it's already set in the environment...")
        
        if not os.environ.get('GITHUB_TOKEN'):
            print("\nERROR: GITHUB_TOKEN is not set. Please run 'python example_env_setup.py' first.")
            sys.exit(1)
    
    # Get the CLI command and arguments
    cli_args = sys.argv[1:] if len(sys.argv) > 1 else ['--help']
    
    # Build the command to run cli.py with the same arguments
    cmd = [sys.executable, 'cli.py'] + cli_args
    
    print(f"Running command: python cli.py {' '.join(cli_args)}")
    print("Environment variables are loaded from .env file.\n")
    
    # Execute the CLI command in the current process environment
    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
