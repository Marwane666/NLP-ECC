"""
Fix compatibility issues between different versions of packages.
"""
import subprocess
import sys
import importlib
import pkg_resources

def check_pydantic_version():
    """Check the installed Pydantic version and ensure compatibility."""
    try:
        pydantic_version = pkg_resources.get_distribution("pydantic").version
        print(f"Installed Pydantic version: {pydantic_version}")
        
        # Check if it's version 1.x or 2.x
        is_v1 = pydantic_version.startswith("1.")
        is_v2 = pydantic_version.startswith("2.")
        
        if is_v1:
            print("You have Pydantic v1 installed, but LangChain Core requires Pydantic v2.")
            upgrade = input("Would you like to upgrade to Pydantic v2? (y/n): ").lower()
            if upgrade == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pydantic>=2.4.0"])
                print("Pydantic upgraded successfully.")
            else:
                print("Using Pydantic v1. Some features might not work correctly.")
                # You might need to downgrade langchain-core if you decide to stay with Pydantic v1
                downgrade_core = input("Would you like to downgrade langchain-core to be compatible with Pydantic v1? (y/n): ").lower()
                if downgrade_core == 'y':
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-core<0.1.0"])
                    print("langchain-core downgraded to be compatible with Pydantic v1.")
        elif is_v2:
            print("You have Pydantic v2 installed, which is compatible with current LangChain versions.")
        else:
            print(f"Unrecognized Pydantic version: {pydantic_version}")
            
    except pkg_resources.DistributionNotFound:
        print("Pydantic is not installed.")
        install = input("Would you like to install Pydantic v2? (y/n): ").lower()
        if install == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic>=2.4.0"])
            print("Pydantic installed successfully.")

def check_langchain_versions():
    """Check if installed LangChain versions are compatible."""
    packages = {
        "langchain": "0.0.325",
        "langchain-core": "0.1.17",
        "langchain-community": "0.0.13"
    }
    
    for package, min_version in packages.items():
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"Installed {package} version: {version}")
            
            # Compare versions
            if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                print(f"Warning: {package} version {version} is older than recommended version {min_version}.")
                upgrade = input(f"Would you like to upgrade {package} to version {min_version}? (y/n): ").lower()
                if upgrade == 'y':
                    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}>={min_version}"])
                    print(f"{package} upgraded successfully.")
                    
        except pkg_resources.DistributionNotFound:
            print(f"{package} is not installed.")
            install = input(f"Would you like to install {package}>={min_version}? (y/n): ").lower()
            if install == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}>={min_version}"])
                print(f"{package} installed successfully.")

def fix_all_compatibility():
    """Fix all compatibility issues."""
    print("Checking and fixing package compatibility issues...")
    check_pydantic_version()
    check_langchain_versions()
    
    print("\nCompatibility check completed.")
    print("You can now run the RAG system commands.")

if __name__ == "__main__":
    fix_all_compatibility()
