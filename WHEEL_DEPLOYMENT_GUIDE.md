# Pre-Compiled Wheels Deployment Guide

## 🎯 What We've Set Up

You now have a fully automated GitHub Actions workflow that:

✅ **Builds wheels automatically** for every version tag pushed to GitHub  
✅ **Supports 5 Python versions**: 3.8, 3.9, 3.10, 3.11, 3.12  
✅ **Supports 3 operating systems**: Linux, macOS, Windows  
✅ **Publishes to PyPI** with a single command  
✅ **Creates GitHub Releases** with all artifacts  
✅ **No compiler needed** for end users on Windows!

## 📋 Quick Start for Future Releases

### Release a New Version (e.g., 0.1.2)

```bash
# 1. Update version numbers
#    Edit: pyproject.toml and python/hapc/__init__.py
#    Change: 0.1.1 → 0.1.2

# 2. Commit
git add pyproject.toml python/hapc/__init__.py
git commit -m "release: version 0.1.2"
git push origin main

# 3. Create and push tag (THIS TRIGGERS THE WORKFLOW!)
git tag v0.1.2
git push origin v0.1.2

# 4. Watch the magic happen:
#    → Go to https://github.com/meixide/hapc/actions
#    → See "Build and Publish Wheels" workflow run
#    → 15 wheels + 1 source dist built
#    → All published to PyPI automatically!

# 5. Verify
#    → Check https://pypi.org/project/hapc/
#    → Should show new version with Windows .whl files
```

## 🔐 Current Status: Setup Almost Complete

### ✅ Already Done
- GitHub Actions workflow created and pushed
- setup.py and pyproject.toml configured
- Version updated to 0.1.1 in all files

### ⚠️ Still Needed: PyPI API Token

**ONE-TIME SETUP** (5 minutes):

1. Go to https://pypi.org/account/
2. Log in (create account if needed)
3. Click **Account Settings**
4. Click **API tokens** (or go to https://pypi.org/account/token/)
5. Click **Add API Token**
   - Name: `GitHub Actions - hapc` (for reference)
   - Scope: **Entire account**
   - Click **Add token**
6. **COPY the token** (looks like `pypi-AgEIcHlwaS5vcmc...`)
   - ⚠️ **SAVE IT NOW** - PyPI won't show it again!

7. Add to GitHub:
   - Go to https://github.com/meixide/hapc/settings/secrets/actions
   - Click **New repository secret**
   - Name: `PYPI_API_TOKEN`
   - Value: Paste token from step 6
   - Click **Add secret**

**That's it!** The workflow can now publish to PyPI.

## 🚀 Test It Now

### Option 1: Release v0.1.1-updated (test with new workflow)

```bash
# Bump to test version
sed -i '' 's/0.1.1/0.1.1-updated/g' pyproject.toml python/hapc/__init__.py

git add pyproject.toml python/hapc/__init__.py
git commit -m "test: release with new wheel workflow"
git push origin main

git tag v0.1.1-updated
git push origin v0.1.1-updated
```

Then watch: https://github.com/meixide/hapc/actions

### Option 2: Trigger Manually (no git tag needed)

1. Go to https://github.com/meixide/hapc/actions
2. Click **"Build and Publish Wheels"** (left sidebar)
3. Click **Run workflow** → **Run workflow**
4. Workflow triggers, but won't publish (needs tag to publish)

This is great for testing the build process!

## 📊 Expected Output

When workflow completes successfully:

```
✅ Build wheels on ubuntu-latest (Python 3.8, 3.9, 3.10, 3.11, 3.12)
   ├─ hapc-0.1.2-cp38-cp38-linux_x86_64.whl
   ├─ hapc-0.1.2-cp39-cp39-linux_x86_64.whl
   ├─ hapc-0.1.2-cp310-cp310-linux_x86_64.whl
   ├─ hapc-0.1.2-cp311-cp311-linux_x86_64.whl
   └─ hapc-0.1.2-cp312-cp312-linux_x86_64.whl

✅ Build wheels on macos-latest (Python 3.8, 3.9, 3.10, 3.11, 3.12)
   ├─ hapc-0.1.2-cp38-cp38-macosx_10_9_x86_64.whl
   ├─ hapc-0.1.2-cp39-cp39-macosx_10_9_x86_64.whl
   ├─ hapc-0.1.2-cp310-cp310-macosx_10_9_x86_64.whl
   ├─ hapc-0.1.2-cp311-cp311-macosx_10_9_x86_64.whl
   └─ hapc-0.1.2-cp312-cp312-macosx_10_9_x86_64.whl

✅ Build wheels on windows-latest (Python 3.8, 3.9, 3.10, 3.11, 3.12)
   ├─ hapc-0.1.2-cp38-cp38-win_amd64.whl
   ├─ hapc-0.1.2-cp39-cp39-win_amd64.whl
   ├─ hapc-0.1.2-cp310-cp310-win_amd64.whl
   ├─ hapc-0.1.2-cp311-cp311-win_amd64.whl
   └─ hapc-0.1.2-cp312-cp312-win_amd64.whl

✅ Source distribution
   └─ hapc-0.1.2.tar.gz

✅ Published to PyPI
   → https://pypi.org/project/hapc/0.1.2/

✅ GitHub Release created
   → https://github.com/meixide/hapc/releases/tag/v0.1.2
```

**Total: 16 artifacts published!**

## 🎁 What Your Friend Gets

Once wheels are on PyPI:

```bash
# Windows (or any platform - NO COMPILER NEEDED!)
pip install hapc

# That's it! No build errors, instant installation.
# They can import and use immediately:
python -c "import hapc; print(hapc.__version__)"
```

**Before (0.1.1 from PyPI):**
```
ERROR: Building wheel for hapc (pyproject.toml) did not run successfully.
× Building wheel for hapc (pyproject.toml) did not run successfully.
│ exit code: 1
╰─> CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
```

**After (0.1.2+ with pre-compiled wheels):**
```
Successfully installed hapc-0.1.2
✓ Ready to use!
```

## 🔧 Troubleshooting

### Workflow fails to build
- Check logs: https://github.com/meixide/hapc/actions
- Common: Missing Eigen3 on Linux
- Solution: Workflow already installs it via apt

### Workflow won't publish
- PyPI token not set? Add it to GitHub secrets
- Token expired? Generate new one and update secret
- Version already on PyPI? Bump version number

### Wheels not showing on PyPI
- Check: https://pypi.org/project/hapc/#history
- Wait 5 minutes for cache refresh
- Look for your version number and platform wheels

## 📚 Files Changed

- `.github/workflows/build-and-publish.yml` - GitHub Actions workflow
- `setup.py` - Updated author/URL info
- `pyproject.toml` - Fixed deprecated license format
- `python/hapc/__init__.py` - Version bumped to 0.1.1
- `PYPI_PUBLISH_GUIDE.md` - Detailed publishing guide

## ✨ Next Steps

1. **Add PyPI API Token** (5 minutes) → See section above
2. **Test the workflow** (optional) → Manual trigger or test tag
3. **Release v0.1.2** → Full workflow cycle
4. **Share with your friend** → They can now `pip install hapc` on Windows!

---

**Result**: Your friend will be able to install HAPC on Windows without any compiler setup! 🎉

Questions? Check the logs at: https://github.com/meixide/hapc/actions
