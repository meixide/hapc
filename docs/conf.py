"""Sphinx configuration for the hapc API documentation.

Builds the Python API reference from the package docstrings (NumPy style, via
``napoleon``).  The compiled pybind11 extension ``hapc.hapc_core`` is *not*
built on the docs host (Read the Docs); it is mocked so autodoc can import the
pure-Python layer and read its docstrings without a C++ toolchain.
"""

import os
import re
import sys

# Make the source package importable from a checkout without installing it.
sys.path.insert(0, os.path.abspath("../python"))


def _read_version() -> str:
    init = os.path.join(os.path.dirname(__file__), "..", "python", "hapc",
                        "__init__.py")
    with open(init, encoding="utf-8") as fh:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', fh.read())
    return m.group(1) if m else "0.0.0"


# -- Project information ------------------------------------------------------
project = "hapc"
author = "Carlos García Meixide"
copyright = f"2024, {author}"
release = _read_version()
version = release

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# The pybind11 extension is unavailable on the docs host: mock it so importing
# `hapc` (which does `from . import hapc_core`) succeeds for autodoc.
autodoc_mock_imports = ["hapc.hapc_core"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = f"hapc {release}"
