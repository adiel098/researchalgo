import sys
import os

# Add the fairpyx directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'fairpyx')))

# Try to import and run the tests
try:
    import pytest
    print("Running Santa Claus Problem tests...")
    exit_code = pytest.main(['fairpyx/tests/test_santa_claus_problem.py', '-v'])
    print(f"Tests completed with exit code: {exit_code}")
except Exception as e:
    print(f"Error running tests: {e}")
    
    # If pytest fails, try to run the test file directly to see the failures
    print("\nAttempting to run tests directly...")
    try:
        from fairpyx.tests import test_santa_claus_problem
        print("Tests imported successfully. Expected failures:")
        
        # List the test functions to show they would fail with empty implementations
        test_functions = [name for name in dir(test_santa_claus_problem) if name.startswith('test_')]
        for func_name in test_functions:
            print(f"- {func_name}: Will fail (empty implementation)")
            
    except Exception as inner_e:
        print(f"Error importing tests directly: {inner_e}")
