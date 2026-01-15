"""
Demo showing Python should use R's cv.hapc for cross-validation.

The C++ fasthal_cv_call is integrated into R's cv.hapc function.
For Python users, call it from R:

```r
library(hapc)
rescv <- cv.hapc(X, Y, npcs=n, log_lambda_min=-6, log_lambda_max=0, 
                 norm="2", predict=Xnew, center=TRUE)
plot(rescv$lambdas, rescv$mses)
```

Or use Python to prepare data and call R via rpy2.
"""

import numpy as np

# Example: Use rpy2 to call R's cv.hapc from Python
try:
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    
    hapc = importr('hapc')
    
    # Prepare data
    n = 100
    p = 1
    X = np.random.uniform(-1, 1, (n, p))
    Y = 2 * np.sin(8 * np.pi * (X[:, 0]**2)) / X[:, 0] + 10 + np.random.normal(0, 2, n)
    Xnew = np.linspace(-1, 1, 100).reshape(-1, 1)
    
    # Call R's cv.hapc
    rescv = hapc.cv_hapc(X, Y, max_degree=2, npcs=n, 
                         log_lambda_min=-6, log_lambda_max=0,
                         norm="2", predict=Xnew, center=True)
    
    print(f"Best lambda: {rescv.rx2('best_lambda')[0]}")
    print(f"Predictions: {rescv.rx2('predictions')}")
    
except ImportError:
    print("""
    rpy2 not installed. To use cv.hapc from Python, install rpy2:
    pip install rpy2
    
    Or simply use R directly:
    library(hapc)
    rescv <- cv.hapc(X, Y, npcs=n, norm="2", predict=Xnew)
    """)
