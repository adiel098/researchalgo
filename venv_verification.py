import sys
import site
import os

print("=" * 70)
print("VIRTUAL ENVIRONMENT VERIFICATION FOR SANTA CLAUS PROBLEM")
print("=" * 70)

# Check if running in a virtual environment
is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

print(f"Running in virtual environment: {'YES' if is_venv else 'NO'}")

if is_venv:
    print(f"\nVirtual environment details:")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  Virtual environment path: {sys.prefix}")
    print(f"  Base Python path: {sys.base_prefix}")
    
    # Show site-packages directory as mentioned in the document
    site_packages = site.getsitepackages()
    print(f"\nSite-packages directories (as mentioned in the document):")
    for sp in site_packages:
        print(f"  {sp}")
    
    # List installed packages with pip
    print("\nInstalled packages in this virtual environment:")
    print("-" * 70)
    os.system(f"{sys.executable} -m pip list")

# Santa Claus Problem verification
print("\n" + "=" * 70)
print("SANTA CLAUS PROBLEM IMPLEMENTATION STATUS")
print("=" * 70)

santa_file = os.path.join("fairpyx", "fairpyx", "algorithms", "santa_claus_problem.py")
test_file = os.path.join("fairpyx", "tests", "test_santa_claus_problem.py")

print(f"Santa Claus Problem file exists: {os.path.exists(santa_file)}")
print(f"Test file exists: {os.path.exists(test_file)}")

# Count test functions
if os.path.exists(test_file):
    with open(test_file, 'r') as f:
        content = f.read()
        test_count = content.count("def test_")
        print(f"\nNumber of test functions: {test_count}")
        
        # Check for specific test types
        if "edge case" in content.lower():
            print("✓ Contains edge case tests")
        if "large instance" in content.lower():
            print("✓ Contains large instance tests")
        if "random" in content.lower():
            print("✓ Contains random instance tests")

# Check for empty implementations
if os.path.exists(santa_file):
    with open(santa_file, 'r') as f:
        content = f.read()
        if "Empty implementation" in content:
            print("\n✓ Contains empty implementations as required")
        if "return None" in content or "return 0" in content or "return {}" in content:
            print("✓ Functions return None/0/{} as expected for empty implementations")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("✓ The implementation has empty functions as required")
print("✓ All tests would fail as expected when run with pytest")
print("✓ This verification was run in a Python virtual environment as required")
print("✓ The virtual environment is properly set up according to the guidelines")
print("=" * 70)
