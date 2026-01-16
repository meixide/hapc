"""Script to check if hapc_core extension is available after installation."""

import sys
from pathlib import Path

print("Checking hapc_core extension...")
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")

# Try to find hapc package
try:
    import hapc
    pkg_dir = Path(hapc.__file__).parent
    print(f"\n✓ hapc package found at: {pkg_dir}")
except ImportError as e:
    print(f"\n✗ Cannot import hapc: {e}")
    sys.exit(1)

# List files in package directory
print(f"\nFiles in package directory:")
for f in sorted(pkg_dir.iterdir()):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size} bytes)")
    else:
        print(f"  {f.name}/ (directory)")

# Try to import hapc_core directly
print(f"\nTrying to import hapc_core:")
try:
    import importlib
    # Try different import strategies
    try:
        hapc_core = importlib.import_module('hapc.hapc_core')
        print(f"  ✓ Successfully imported as hapc.hapc_core")
        print(f"  Location: {getattr(hapc_core, '__file__', 'builtin')}")
    except ImportError:
        try:
            hapc_core = importlib.import_module('hapc_core')
            print(f"  ✓ Successfully imported as hapc_core")
            print(f"  Location: {getattr(hapc_core, '__file__', 'builtin')}")
        except ImportError as e2:
            print(f"  ✗ Failed: {e2}")
except Exception as e:
    print(f"  ✗ Error: {e}")
