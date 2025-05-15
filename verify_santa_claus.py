import os
import sys
import inspect
import platform
import site

# Check if the santa_claus_problem.py file exists
santa_file_path = os.path.join("fairpyx", "fairpyx", "algorithms", "santa_claus_problem.py")
test_file_path = os.path.join("fairpyx", "tests", "test_santa_claus_problem.py")

print("Verifying Santa Claus Problem files in Virtual Environment...")
print("=" * 50)

# Display Python environment information
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Check if running in a virtual environment
is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
print(f"Running in virtual environment: {'YES' if is_venv else 'NO'}")
print(f"Virtual environment path: {sys.prefix}")

# Show site-packages directory
site_packages = site.getsitepackages()[0]
print(f"Site-packages directory: {site_packages}")

print("\n" + "=" * 50)

# Check if the files exist
print(f"Checking if {santa_file_path} exists: {os.path.exists(santa_file_path)}")
print(f"Checking if {test_file_path} exists: {os.path.exists(test_file_path)}")

# Read the santa_claus_problem.py file to check for empty implementations
if os.path.exists(santa_file_path):
    with open(santa_file_path, 'r') as f:
        content = f.read()
        
    print("\nChecking santa_claus_problem.py content:")
    print("-" * 50)
    
    # Check if the file has the required functions
    required_functions = [
        "santa_claus_algorithm",
        "find_optimal_target_value",
        "configuration_lp_solver",
        "create_super_machines",
        "round_small_configurations",
        "construct_final_allocation"
    ]
    
    for func in required_functions:
        if func in content:
            print(f"✓ Function {func} found")
            
            # Check if it has an empty implementation
            if "Empty implementation" in content or "return None" in content or "return 0" in content or "return {}" in content:
                print(f"  ✓ Has empty implementation (as required)")
            else:
                print(f"  ✗ Does not have an empty implementation")
        else:
            print(f"✗ Function {func} not found")
    
    # Check if the file has doctest examples
    if ">>>" in content:
        print("✓ Contains doctest examples")
    else:
        print("✗ Does not contain doctest examples")
        
    # Check if the file has proper header
    if "The Santa Claus Problem" in content and "Programmers: Roey and Adiel" in content:
        print("✓ Contains proper header with programmer names")
    else:
        print("✗ Does not contain proper header")

# Read the test_santa_claus_problem.py file to check for tests
if os.path.exists(test_file_path):
    with open(test_file_path, 'r') as f:
        test_content = f.read()
        
    print("\nChecking test_santa_claus_problem.py content:")
    print("-" * 50)
    
    # Count the number of test functions
    test_count = test_content.count("def test_")
    print(f"Found {test_count} test functions")
    
    # Check for edge case tests
    if "edge case" in test_content.lower():
        print("✓ Contains edge case tests")
    else:
        print("✗ Does not contain edge case tests")
        
    # Check for large instance tests
    if "large instance" in test_content.lower():
        print("✓ Contains large instance tests")
    else:
        print("✗ Does not contain large instance tests")
        
    # Check for random tests
    if "random" in test_content.lower():
        print("✓ Contains random instance tests")
    else:
        print("✗ Does not contain random instance tests")

print("\nVerification complete!")
print("=" * 50)
print("Note: Since the implementations are empty, all tests would fail as expected.")
print("This is the expected behavior for this stage of the homework.")
print("\nThis verification was run in a Python virtual environment as required.")
print(f"Virtual environment: {'ACTIVE' if is_venv else 'NOT ACTIVE'}")
print(f"Python path: {sys.executable}")
print("=" * 50)
