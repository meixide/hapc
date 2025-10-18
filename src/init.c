#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// Declarations of your native functions
extern SEXP pchal_cv_call(SEXP, SEXP, SEXP, SEXP,
                          SEXP, SEXP, SEXP, SEXP,
                          SEXP, SEXP, SEXP, SEXP);
extern SEXP fasthal_cv_call(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP kernel_call(SEXP);
extern SEXP kernel_cross_call(SEXP, SEXP);
extern SEXP fast_pchal_call(SEXP, SEXP, SEXP, SEXP);
extern SEXP pcghal_call(SEXP, SEXP, SEXP, SEXP,
                             SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP pchal_des(SEXP, SEXP, SEXP);
extern SEXP ridge_call(SEXP, SEXP, SEXP);


// Registration table
static const R_CallMethodDef CallEntries[] = {
  {"pchal_cv_call", (DL_FUNC) &pchal_cv_call, 12},
  {"fasthal_cv_call", (DL_FUNC) &fasthal_cv_call, 6},
  {"kernel_call", (DL_FUNC) &kernel_call, 1},
  {"kernel_cross_call", (DL_FUNC) &kernel_cross_call, 2},
  {"fast_pchal_call", (DL_FUNC) &fast_pchal_call, 4},
  {"pcghal_call", (DL_FUNC) &pcghal_call, 9},
  {"pchal_des", (DL_FUNC) &pchal_des, 3},
  {"ridge_call", (DL_FUNC) &ridge_call, 3},
  {NULL, NULL, 0}
};

// Initialization
void R_init_hapc(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}