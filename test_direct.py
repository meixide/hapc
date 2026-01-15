"""Direct test of hapc_core module."""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python" / "hapc"))
sys.path.insert(0, str(project_root / "build"))
sys.path.insert(0, str(project_root / "python"))

print("Python paths:")
for p in sys.path[:5]:
    print(f"  {p}")

try:
    import hapc_core
    print("\n✓ hapc_core module imported successfully!")
    print(f"  Module: {hapc_core}")
    print(f"  Location: {hapc_core.__file__}")
    
    # List available functions
    funcs = [x for x in dir(hapc_core) if not x.startswith('_')]
    print(f"\n  Available functions/classes: {funcs}")
    
except ImportError as e:
    print(f"\n✗ Failed to import hapc_core: {e}")
    print("\nSearching for hapc_core files...")
    import os
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if "hapc_core" in file and file.endswith(('.so', '.pyd', '.dylib')):
                print(f"  Found: {os.path.join(root, file)}")
