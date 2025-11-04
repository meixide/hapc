# HAPC

`hapc` is an R package wrapping a fully C++ backend. 

## Installation

From the parent directory, build and install the package using:

```bash
R CMD build hapc
R CMD INSTALL hapc_0.1.0.tar.gz
```

Alternatively, install directly from GitHub:

```r
# install.packages("remotes")
remotes::install_github("meixide/hapc")
```

## Usage

Load the package in your R session:

```r
library(hapc)
```

## Useful Development Commands

During development, the following commands are helpful for maintaining and testing the package:

```r
devtools::document()    # Generate documentation from roxygen2 comments
devtools::build()       # Build the package
devtools::check()       # Run package checks and tests
```

```r
# 1. Unload the package completely
try(detach("package:hapc", unload = TRUE), silent = TRUE)

# 2. Remove the installed package
remove.packages("hapc")

# 3. Clean build artifacts
pkgbuild::clean_dll()

# 4. Restart R session (critical!)
.rs.restartR()

# After restart:
# 5. Reinstall
setwd("~/Projects/hapc")
devtools::install()

# Or load for development
devtools::load_all()
```

## Package Management

To remove or detach the package:

```r
remove.packages("hapc")                          # Remove the package
detach("package:hapc", unload = TRUE)            # Unload from current session
```

## Getting Started

After installation, explore the package functions and documentation to get started with your analysis workflows.
