# PyPI Wheel Publishing Setup

## Step 1: Generate PyPI API Token

1. Go to https://pypi.org/account/
2. Log in with your PyPI account (or create one at https://pypi.org/account/register/)
3. Navigate to **Account Settings** → **API Tokens**
4. Click **Add API Token**
5. Give it a name like "GitHub Actions - hapc"
6. Scope: **Entire account** (or just "hapc" project if available)
7. Copy the token (looks like: `pypi-AgEIcHlwaS5vcmc...`)

## Step 2: Add Secret to GitHub

1. Go to your GitHub repository: https://github.com/meixide/hapc
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `PYPI_API_TOKEN`
5. Value: Paste the token from Step 1
6. Click **Add secret**

## Step 3: Trigger Wheel Builds and Publication

### Option A: Create a Git Tag (Recommended)

```bash
# Make sure you're on main branch with latest changes committed
git checkout main
git pull origin main

# Create and push a version tag
git tag v0.1.1
git push origin v0.1.1
```

The workflow will automatically:
1. ✅ Build wheels for Python 3.8-3.12 on Ubuntu, macOS, and Windows
2. ✅ Build source distribution (.tar.gz)
3. ✅ Publish all wheels and source to PyPI
4. ✅ Create a GitHub Release with artifacts

### Option B: Manual Trigger

1. Go to **Actions** tab in GitHub
2. Select **"Build and Publish Wheels"** workflow
3. Click **Run workflow** dropdown
4. Click **Run workflow** button

## Step 4: Verify Publication

After the workflow completes:

```bash
# Check PyPI page
open https://pypi.org/project/hapc/

# Install from PyPI
pip install --upgrade hapc

# Verify installation
python -c "import hapc; print(hapc.__version__)"
```

## Workflow Details

The GitHub Actions workflow:
- **Triggers on**: Git tags matching `v*` (e.g., `v0.1.1`, `v0.2.0`)
- **Builds**: Python 3.8, 3.9, 3.10, 3.11, 3.12 on Linux, macOS, Windows
- **Creates**: 15 wheel files (.whl) + 1 source distribution (.tar.gz)
- **Publishes to**: PyPI with `twine upload`
- **Skips existing**: Already-uploaded wheels (safe to re-run)
- **Creates GitHub Release**: With all artifacts attached

## Troubleshooting

### Build Fails on Windows
- CMake/MSVC installed? Check: `cmake --version && cl.exe`
- Visual Studio Build Tools required: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Build Fails on Linux
- Eigen3 and cmake installed? Check: `apt-cache policy cmake libeigen3-dev`

### PyPI Upload Fails
- Token expired? Generate new one at https://pypi.org/account/token/
- Project name correct? Should match `name = "hapc"` in pyproject.toml
- Version already exists? Increment version and re-tag

### Checking Workflow Status
- GitHub Actions tab: https://github.com/meixide/hapc/actions
- Look for "Build and Publish Wheels" workflow
- Click on a run to see detailed logs

## Future Version Releases

Every time you want to release a new version:

```bash
# 1. Update version in pyproject.toml and __init__.py
#    e.g., 0.1.1 → 0.1.2

# 2. Commit changes
git add pyproject.toml python/hapc/__init__.py
git commit -m "release: version 0.1.2"

# 3. Create and push tag
git tag v0.1.2
git push origin main
git push origin v0.1.2

# 4. Workflow automatically builds and publishes!
# 5. Verify on https://pypi.org/project/hapc/
```

---

**Result**: Your friend can now install with `pip install hapc` on Windows without needing a C++ compiler! 🎉
