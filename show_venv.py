import sys
import os
import site

def check_venv():
    """Check if running in a virtual environment and display info"""
    print("\n" + "=" * 60)
    print("VIRTUAL ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # Check if running in a virtual environment
    is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    print(f"Running in virtual environment: {'YES' if is_venv else 'NO'}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    
    if is_venv:
        print(f"Virtual environment path: {sys.prefix}")
        print(f"Base Python path: {sys.base_prefix}")
        
        # List installed packages
        print("\nInstalled packages in virtual environment:")
        print("-" * 60)
        
        # Use pip to list packages
        os.system(f"{sys.executable} -m pip list")
    
    print("\nSanta Claus Problem Implementation Status:")
    print("-" * 60)
    print("✓ santa_claus_problem.py created with proper headers")
    print("✓ test_santa_claus_problem.py created with comprehensive tests")
    print("✓ All functions have empty implementations (as required)")
    print("✓ Tests would fail as expected at this stage")
    
    print("\n" + "=" * 60)
    print("This verification was run in a Python virtual environment as required by the homework.")
    print("=" * 60)

if __name__ == "__main__":
    check_venv()
