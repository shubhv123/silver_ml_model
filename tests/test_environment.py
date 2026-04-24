#!/usr/bin/env python
import sys
import importlib

def test_import(package_name):
    """Test if a package can be imported"""
    try:
        importlib.import_module(package_name)
        return f"✓ {package_name}"
    except ImportError as e:
        return f"✗ {package_name}: {str(e)}"

def main():
    """Test all required packages"""
    
    packages = [
        'numpy',
        'pandas',
        'yfinance',
        'fredapi',
        'qlib',
        'sklearn',
        'matplotlib',
        'seaborn',
        'plotly'
    ]
    
    print("Testing Environment Setup")
    print("="*50)
    
    all_passed = True
    for package in packages:
        result = test_import(package)
        print(result)
        if result.startswith('✗'):
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("✓ All packages installed successfully!")
        print("\nPython version:", sys.version)
    else:
        print("✗ Some packages are missing. Run: pip install -r requirements.txt")
        
if __name__ == "__main__":
    main()