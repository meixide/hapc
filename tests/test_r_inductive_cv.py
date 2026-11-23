"""Runs the R testthat file ``tests/r/test_inductive_cv.R`` (inductive
gaussian CV of ``cv.hapc``) through ``Rscript``.

Skipped if ``Rscript`` or ``testthat`` is missing, or if the installed R
``hapc`` predates the ``inductive`` argument of ``cv.hapc()``.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript is required for the R inductive CV tests.",
)

TEST_FILE = Path(__file__).parent / "r" / "test_inductive_cv.R"

R_DRIVER = """
for (pkg in c("testthat", "hapc")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat("__SKIP__", pkg, "not installed"); quit(save = "no", status = 0)
    }
}
if (!("inductive" %in% names(formals(hapc::cv.hapc)))) {
    cat("__SKIP__ installed R hapc predates the inductive CV")
    quit(save = "no", status = 0)
}
res <- as.data.frame(testthat::test_file(commandArgs(TRUE)[1],
                                         reporter = "summary"))
quit(save = "no", status = as.integer(any(res$failed > 0) || any(res$error)))
"""


def test_r_inductive_cv():
    out = subprocess.run(
        ["Rscript", "-e", R_DRIVER, str(TEST_FILE)],
        capture_output=True, text=True, timeout=600,
    )
    if "__SKIP__" in out.stdout:
        pytest.skip(out.stdout.split("__SKIP__", 1)[1].strip())
    assert out.returncode == 0, out.stdout + out.stderr
