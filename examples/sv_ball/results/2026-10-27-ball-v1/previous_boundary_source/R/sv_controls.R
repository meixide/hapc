# Validate before entering native SV solvers (C++ exceptions must not escape R).
.hapc_check_sv_inputs <- function(X, Y, max_iter, tol, step_factor) {
  if (nrow(X) < 1L || ncol(X) < 1L || length(Y) != nrow(X) ||
      any(!is.finite(X)) || any(!is.finite(Y)))
    stop("SV solver requires finite X and Y with matching dimensions", call. = FALSE)
  if (length(max_iter) != 1L || is.na(max_iter) || max_iter < 0L ||
      length(tol) != 1L || !is.finite(tol) || tol <= 0 ||
      length(step_factor) != 1L || !is.finite(step_factor) ||
      step_factor <= 0 || step_factor >= 1)
    stop("SV solver requires max_iter>=0, tol>0 and 0<step_factor<1", call. = FALSE)
}
