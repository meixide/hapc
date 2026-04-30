# Convenience targets for the hapc Python package.
#
# Usage:
#   make test                # full Python test suite (skips R integration)
#   make figure              # regenerate ate_hapc_diagnostics_demo.png
#   make release VERSION=0.3.1 [MSG="..."]
#                            # bump version, commit, tag, push (triggers PyPI)
#   make smoke-pypi VERSION=0.3.1
#                            # fresh venv, pip install hapc==<VERSION>, smoke test

.PHONY: help test figure release smoke-pypi clean

PYTHON ?= python3
VERSION ?=
MSG ?=

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*## /\t- /'
	@echo ""
	@echo "Interactive release steps:  bash scripts/publish_hapc_release.sh help"

test: ## Run pytest (skips R integration)
	$(PYTHON) -m pytest -q --ignore=tests/test_r_vs_python_alpha.py

figure: ## Regenerate ate_hapc_diagnostics_demo.png
	$(PYTHON) tests/test_ate_hapc_diagnostics_example.py

release: ## Bump version + tag + push (set VERSION=X.Y.Z [MSG="..."])
	@if [ -z "$(VERSION)" ]; then echo 'usage: make release VERSION=X.Y.Z [MSG="..."]'; exit 2; fi
	@if [ -n "$(MSG)" ]; then \
		bash scripts/release.sh $(VERSION) -m "$(MSG)"; \
	else \
		bash scripts/release.sh $(VERSION); \
	fi

smoke-pypi: ## Fresh venv install of `hapc==$(VERSION)` from PyPI + smoke test
	@if [ -z "$(VERSION)" ]; then echo 'usage: make smoke-pypi VERSION=X.Y.Z'; exit 2; fi
	$(PYTHON) -m venv /tmp/hapc-$(VERSION)
	. /tmp/hapc-$(VERSION)/bin/activate && \
		pip install --upgrade pip "hapc==$(VERSION)" && \
		python -c "import hapc, numpy as np; \
print('hapc', hapc.__version__); \
from hapc import ate_hapc; rng = np.random.default_rng(0); n = 100; \
W = rng.standard_normal((n,3)); A = (W[:,0] + rng.standard_normal(n) > 0).astype(float); \
Y = W[:,1] + 0.3*A + rng.standard_normal(n)*0.2; \
res = ate_hapc(W, Y, A, max_degree=1, npcs=15, nfolds=3, norm='2', \
log_lambda_prop_min=-4, log_lambda_prop_max=-2, grid_length_prop=4, \
log_lambda_out_min=-3, log_lambda_out_max=-1, grid_length_out=4); \
print('ate_hapc OK ->', res)"

clean: ## Remove build artifacts
	rm -rf build/ dist/ wheels-temp/ *.egg-info python/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
