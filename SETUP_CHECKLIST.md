# ✅ Pre-Compiled Wheels - Quick Checklist

## What's Done ✅

- [x] GitHub Actions workflow created (`.github/workflows/build-and-publish.yml`)
- [x] Workflow configured for Python 3.8-3.12 on Linux, macOS, Windows
- [x] setup.py and pyproject.toml updated
- [x] Version bumped to 0.1.1
- [x] Documentation created (3 guides)
- [x] All changes committed and pushed to GitHub

## What YOU Need to Do (5 minutes)

### 1️⃣ Generate PyPI API Token

Visit: https://pypi.org/account/

```
1. Login to your account (create if needed)
2. Click "Account Settings"
3. Click "API tokens"
4. Click "Add API Token"
   - Name: "GitHub Actions - hapc"
   - Scope: "Entire account"
5. Click "Add token"
6. ⚠️ COPY the token immediately - it won't show again!
   It looks like: pypi-AgEIcHlwaS5vcmc...
```

### 2️⃣ Add Token to GitHub Secrets

Visit: https://github.com/meixide/hapc/settings/secrets/actions

```
1. Click "New repository secret"
2. Name: PYPI_API_TOKEN
3. Value: [paste token from step 1]
4. Click "Add secret"
```

✅ **Done!** Wheels can now be published

## Quick Test (Optional)

### Test Build Without Publishing

```bash
cd /Users/cgmeixide/Projects/hapc

# Trigger workflow manually
# Go to: https://github.com/meixide/hapc/actions
# Select: "Build and Publish Wheels"
# Click: "Run workflow" → "Run workflow"

# This builds wheels but won't publish (no tag)
# Perfect for testing!
```

### Actual Release

When ready to release v0.1.2:

```bash
# Update version
sed -i '' 's/0.1.1/0.1.2/g' pyproject.toml python/hapc/__init__.py

# Commit
git add pyproject.toml python/hapc/__init__.py
git commit -m "release: version 0.1.2"
git push origin main

# Create tag (this triggers the workflow!)
git tag v0.1.2
git push origin v0.1.2

# ✅ Workflow automatically:
#    → Builds 15 wheels + source
#    → Publishes to PyPI
#    → Creates GitHub release
```

## After Release

### Verify on PyPI

Visit: https://pypi.org/project/hapc/

You should see:
- New version (0.1.2)
- 15 wheel files (.whl) for all combinations
- Source distribution (.tar.gz)
- **Windows wheels!** No compiler needed!

### Test Installation

```bash
# Anyone can now install (including your friend on Windows)
pip install hapc

# ✅ Instant installation, no build errors!
```

## Key Files

| File | Purpose |
|------|---------|
| `.github/workflows/build-and-publish.yml` | Automated build & publish |
| `WHEELS_READY.md` | Summary of what's done |
| `WHEEL_DEPLOYMENT_GUIDE.md` | Detailed instructions |
| `PYPI_PUBLISH_GUIDE.md` | PyPI token setup guide |

## Workflow Triggers

The workflow will **automatically build and publish when**:
- You push a tag like `v0.1.2`

The workflow can **manually run** (just builds, no publish):
- Go to Actions tab → "Build and Publish Wheels" → "Run workflow"

## Expected Result

**Before (0.1.1 source distribution):**
```
ERROR: Building wheel for hapc (pyproject.toml) did not run successfully.
× Building wheel for hapc (pyproject.toml) did not run successfully.
│ exit code: 1
╰─> CMake Error: CMAKE_C_COMPILER not set
```

**After (0.1.2+ with pre-compiled wheels):**
```bash
pip install hapc
Successfully installed hapc-0.1.2

python -c "import hapc; print('✓ Ready to use!')"
✓ Ready to use!
```

---

## Summary

**To start building and publishing wheels:**

1. Add PyPI token to GitHub (5 minutes)
2. Create version tag (1 minute)
3. Workflow does the rest automatically! ✨

**Links:**
- PyPI Account: https://pypi.org/account/
- GitHub Secrets: https://github.com/meixide/hapc/settings/secrets/actions
- Workflow Runs: https://github.com/meixide/hapc/actions
- Package Page: https://pypi.org/project/hapc/

**Result:** Your friend can now `pip install hapc` on Windows without any compiler! 🎉
