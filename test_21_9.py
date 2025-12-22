"""Quick test of exercise 21_9"""
import sys
print("Starting test...")
print(f"Python version: {sys.version}")

try:
    print("Importing numpy...")
    import numpy as np
    print("✓ NumPy OK")
    
    print("Importing matplotlib...")
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    print("✓ Matplotlib OK")
    
    print("Importing tqdm...")
    from tqdm import tqdm
    print("✓ tqdm OK")
    
    print("\nAll imports successful!")
    print("="*50)
    print("QUICK REINFORCE TEST")
    print("="*50)
    
    # Quick test
    for i in tqdm(range(10), desc="Test"):
        pass
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
