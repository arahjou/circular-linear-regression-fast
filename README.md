---
Author: Dr. Ali Rahjouei
Email: ali.rahjouei@gmail.com
Date: 28.01.2026
---

# Optimization of Circular-Linear Regression Algorithm

## Technical Report: Modifications to `lm.circular.cl` from the R `circular` Package

**Date:** January 2026\
**Purpose:** Documentation of algorithm modifications for academic
publication

------------------------------------------------------------------------

## 1. Introduction

The original `lm.circular.cl` function from the R `circular` package
(Agostinelli & Lund, 2017) implements circular-linear regression using
an iterative weighted least squares algorithm. During application to
circadian rhythm data analysis, we identified convergence issues that
prevented the algorithm from terminating. This document describes the
diagnostic process and subsequent optimizations implemented to resolve
these issues.

### 1.1 Original Source

-   **Function:** `lm.circular.cl`
-   **Package:** circular (version 0.5-1)
-   **Original Authors:** Claudio Agostinelli and Ulric Lund
-   **License:** GPL-2

------------------------------------------------------------------------

## 2. Problem Identification

### 2.1 Observed Behavior

When applying the original function to our dataset (n = 4,847
observations), the algorithm failed to converge within reasonable time.
Diagnostic testing with verbose output revealed:

```         
Iteration 1: Log-Likelihood = 5314.556
Iteration 2: Log-Likelihood = 5315.112
Iteration 3: Log-Likelihood = 5315.112
Iteration 4: Log-Likelihood = 5315.112
...
Iteration 500: Log-Likelihood = 5315.112  diff = 1.17e-06
```

### 2.2 Root Cause Analysis

The algorithm uses a single convergence criterion based on the maximum
absolute change in beta coefficients:

``` r
while (diff > tol) {
    # ... iteration code ...
    diff <- max(abs(betaNew - betaPrev))
}
```

With the default tolerance (`tol = 1e-10`), the algorithm entered a
**limit cycle** where:

1.  The log-likelihood had fully converged (no change after iteration 3)
2.  Beta coefficients oscillated at \~1.17×10⁻⁶ due to floating-point
    arithmetic
3.  This oscillation exceeded the tolerance, preventing termination
4.  The algorithm would run indefinitely without a maximum iteration
    limit

------------------------------------------------------------------------

## 3. Modifications Implemented

### 3.1 Summary of Changes

| Modification | Original | Optimized | Rationale |
|----|----|----|----|
| Maximum iterations | None | 500 (configurable) | Prevents infinite loops |
| Convergence criteria | Beta only | Beta OR Log-likelihood | Detects practical convergence |
| Log-likelihood tolerance | N/A | 1×10⁻⁶ (configurable) | Accounts for floating-point limits |
| Matrix operations | Full diagonal matrices | Vectorized operations | Improved computational efficiency |
| Inner loop solver | `lm()` function | Direct `solve()` | Reduced overhead |
| Convergence reporting | None | Detailed output | Transparency and diagnostics |

### 3.2 Detailed Code Changes

#### 3.2.1 Addition of Maximum Iteration Limit

**Original code:**

``` r
while (diff > tol) {
    iter <- iter + 1
    # ... iteration code ...
}
```

**Modified code:**

``` r
while ((diff > tol) && (ll_diff > ll_tol) && (iter < max_iter)) {
    iter <- iter + 1
    # ... iteration code ...
}
```

**Rationale:** The original algorithm had no safeguard against infinite
loops. Adding `max_iter` ensures termination even in pathological cases.

------------------------------------------------------------------------

#### 3.2.2 Dual Convergence Criteria

**Original code:**

``` r
diff <- max(abs(betaNew - betaPrev))
# Only criterion: diff > tol
```

**Modified code:**

``` r
diff <- max(abs(beta - beta_old))
ll_diff <- abs(log_lik - log_lik_prev)

# Convergence when EITHER criterion is met:
# 1. Beta coefficients stabilize (diff <= tol)
# 2. Log-likelihood stabilizes (ll_diff <= ll_tol)
```

**Rationale:** In iterative optimization, practical convergence occurs
when the objective function (log-likelihood) stops improving. Due to
floating-point arithmetic limitations, parameter estimates may continue
to show small oscillations even after the likelihood has converged. The
dual criterion recognizes convergence in either case.

------------------------------------------------------------------------

#### 3.2.3 Vectorized Matrix Operations

**Original code:**

``` r
A <- diag(k*A1(k), nrow=n)
g.p <- diag(apply(x, 1, function(row, betaPrev) 
    2/(1+(t(betaPrev)%*%row)^2), betaPrev=betaPrev), nrow=n)
D <- g.p %*% x
```

**Modified code:**

``` r
# Vectorized: no n×n diagonal matrices needed
g_prime <- 2 / (1 + xb^2)  # Vector of length n
D <- g_prime * x           # Element-wise multiplication broadcasts
```

**Rationale:** - Original creates two n×n diagonal matrices (memory:
O(n²)) - Modified uses vectorized operations (memory: O(n)) - For n =
4,847, this reduces memory from \~188 MB to \~39 KB per matrix - The
`apply()` function with row-wise operations is replaced by vectorized
arithmetic

------------------------------------------------------------------------

#### 3.2.4 Direct Linear System Solution

**Original code:**

``` r
betaNew <- lm(t(D)%*%(u+A%*%D%*%betaPrev) ~ t(D)%*%A%*%D - 1)$coefficients
```

**Modified code:**

``` r
DtD <- crossprod(D)
working_response <- u / weights + as.vector(D %*% beta)
Dt_response <- crossprod(D, working_response)

beta <- tryCatch(
    as.vector(solve(DtD, Dt_response)),
    error = function(e) {
        as.vector(qr.solve(qr(DtD), Dt_response))
    }
)
```

**Rationale:** - Calling `lm()` inside a loop incurs overhead (formula
parsing, model frame construction) - Direct matrix solution via
`solve()` is more efficient - QR decomposition fallback handles
near-singular cases - `crossprod()` is faster than `t(X) %*% X`

------------------------------------------------------------------------

#### 3.2.5 Enhanced Output and Diagnostics

**Added fields to output:**

``` r
list(
    # ... original fields ...
    converged = converged,           # Logical: did algorithm converge?
    converge_reason = converge_reason, # "beta", "log-likelihood", or "max_iter"
    iterations = iter                # Number of iterations performed
)
```

**Rationale:** Transparency in reporting convergence status is essential
for: - Model validation - Debugging - Publication reproducibility

------------------------------------------------------------------------

## 4. Mathematical Equivalence

The modifications preserve the mathematical model and estimation
procedure:

### 4.1 Model Specification (Unchanged)

The circular-linear regression model relates a circular response θ to
linear predictors:

$$\theta_i = \mu + 2 \arctan(\mathbf{x}_i^\top \boldsymbol{\beta}) + \epsilon_i$$

where $\epsilon_i$ follows a von Mises distribution with concentration
parameter $\kappa$.

### 4.2 Estimation Algorithm (Unchanged)

The iterative reweighted least squares algorithm remains identical:

1.  Initialize $\boldsymbol{\beta}^{(0)}$
2.  Compute residuals and update $\mu$, $\kappa$
3.  Compute working weights and working response
4.  Solve weighted least squares for $\boldsymbol{\beta}^{(t+1)}$
5.  Check convergence; if not converged, return to step 2

### 4.3 Verification

To verify equivalence, we compared results with relaxed tolerance in the
original function:

| Parameter             | Original (`tol=1e-5`) | Modified |
|-----------------------|-----------------------|----------|
| $\beta_0$ (Intercept) | -0.5167               | -0.5167  |
| $\beta_1$ (bs(age)1)  | 0.2316                | 0.2316   |
| $\beta_2$ (bs(age)2)  | 0.2447                | 0.2447   |
| $\beta_3$ (bs(age)3)  | 0.1608                | 0.1608   |
| $\beta_4$ (sexF)      | 0.0145                | 0.0145   |
| $\mu$                 | -1.568                | -1.568   |
| $\kappa$              | 6.927                 | 6.927    |
| Log-Likelihood        | 5315.112              | 5315.112 |

Results are identical to numerical precision.

------------------------------------------------------------------------

## 5. Performance Comparison

### 5.1 Convergence Speed

| Metric                    | Original                 | Modified       |
|---------------------------|--------------------------|----------------|
| Iterations to converge    | \>500 (did not converge) | 4              |
| Convergence criterion met | None                     | Log-likelihood |
| Final beta_diff           | 1.17×10⁻⁶                | 9.78×10⁻⁶      |
| Final ll_diff             | 8.75×10⁻⁸                | 1.18×10⁻⁷      |

### 5.2 Computational Efficiency

For n = 4,847 observations and p = 5 predictors:

| Operation            | Original | Modified | Improvement |
|----------------------|----------|----------|-------------|
| Memory per iteration | \~188 MB | \~0.2 MB | \~940×      |
| Time per iteration   | \~0.15 s | \~0.02 s | \~7.5×      |

------------------------------------------------------------------------

## 6. Usage

### 6.1 Function Signature

``` r
lm_circular_cl_fast(
    y,                    # Circular response variable
    x,                    # Design matrix
    init = NULL,          # Initial values for beta (required)
    verbose = FALSE,      # Print iteration details
    tol = 1e-10,          # Tolerance for beta convergence
    max_iter = 500,       # Maximum iterations
    ll_tol = 1e-6,        # Tolerance for log-likelihood convergence
    control.circular = list()  # Circular data properties
)
```

### 6.2 Example

``` r
library(circular)
library(splines)
source("lm_circular_cl_fast.R")

# Prepare circular response
y_circular <- circular(time_data * 2 * pi / 24, units = "radians")

# Design matrix
X <- model.matrix(~ bs(age) + sex, data = mydata)

# Initial values
init_values <- rep(0, ncol(X))

# Fit model
result <- lm_circular_cl_fast(
    y = y_circular,
    x = X,
    init = init_values,
    verbose = TRUE
)

print(result)
```

------------------------------------------------------------------------

## 7. Limitations and Recommendations

### 7.1 Limitations

1.  **Initial values:** The algorithm requires reasonable starting
    values. Poor initialization may lead to local optima or slow
    convergence.

2.  **Model assumptions:** The modifications do not change the
    underlying von Mises error distribution assumption.

3.  **Large datasets:** While more efficient, very large datasets (n \>
    100,000) may still require substantial computation time.

### 7.2 Recommendations

1.  Use `verbose = TRUE` for initial model fitting to monitor
    convergence.

2.  Compare results with the original function (using relaxed tolerance)
    to verify consistency.

3.  Report convergence diagnostics in publications:

    -   Number of iterations
    -   Convergence criterion met
    -   Final log-likelihood

------------------------------------------------------------------------

## 8. References

Agostinelli, C., & Lund, U. (2017). R package 'circular': Circular
Statistics (version 0.4-93).
<https://CRAN.R-project.org/package=circular>

Fisher, N. I. (1993). *Statistical Analysis of Circular Data*. Cambridge
University Press.

Jammalamadaka, S. R., & SenGupta, A. (2001). *Topics in Circular
Statistics*. World Scientific.

------------------------------------------------------------------------

## 9. Code Availability

The optimized function `lm_circular_cl_fast.R` is available as
supplementary material. The code is released under GPL-2 license,
consistent with the original `circular` package.

------------------------------------------------------------------------

## Appendix A: Complete Modified Function

See accompanying file: `lm_circular_cl_fast.R`

## Appendix B: Comparison of Original and Modified Code

### B.1 Original Core Algorithm (from circular package)

``` r
LmCircularclRad <- function(y, x, init, verbose, tol) {
   n <- length(y)
   y <- y%%(2*pi)
   betaPrev <- init  
   S <- sum(sin(y-2*atan(x%*%betaPrev)))/n
   C <- sum(cos(y-2*atan(x%*%betaPrev)))/n
   R <- sqrt(S^2 + C^2)
   mu <- atan2(S,C)
   k  <- A1inv(R)
   diff <- tol+1
   iter <- 0
   while (diff > tol){
      iter <- iter + 1
      u <- k*sin(y - mu - 2*atan(x%*%betaPrev))
      A <- diag(k*A1(k), nrow=n)
      g.p <- diag(apply(x, 1, function(row, betaPrev) 
          2/(1+(t(betaPrev)%*%row)^2), betaPrev=betaPrev), nrow=n)
      D <- g.p%*%x
      betaNew <- lm(t(D)%*%(u+A%*%D%*%betaPrev) ~ t(D)%*%A%*%D - 1)$coefficients
      diff <- max(abs(betaNew - betaPrev))
      betaPrev <- betaNew
        
      S <- sum(sin(y-2*atan(x%*%betaPrev)))/n
      C <- sum(cos(y-2*atan(x%*%betaPrev)))/n
      R <- sqrt(S^2 + C^2)
      mu <- atan2(S,C)
      k  <- A1inv(R)
        
      if (verbose){
         log.lik <- -n*log(besselI(x = k, nu = 0, expon.scaled = FALSE)) + 
             k*sum(cos(y-mu-2*atan(x%*%betaNew)))
         cat("Iteration ", iter, ":    Log-Likelihood = ", log.lik, "\n")
      }
   }
   # ... remainder of function
}
```

### B.2 Modified Core Algorithm

``` r
lm_circular_cl_core <- function(y, x, init, verbose, tol, max_iter, ll_tol) {
  
  n <- length(y)
  p <- NCOL(x)
  y <- y %% (2 * pi)
  
  beta <- init
  x <- as.matrix(x)
  
  # Helper functions defined internally
  A1 <- function(k) {
    if (k < 1e-6) return(k / 2)
    besselI(k, 1) / besselI(k, 0)
  }
  
  A1inv <- function(R) {
    if (R < 0.53) {
      return(2 * R + R^3 + 5 * R^5 / 6)
    } else if (R < 0.85) {
      return(-0.4 + 1.39 * R + 0.43 / (1 - R))
    } else {
      return(1 / (R^3 - 4 * R^2 + 3 * R))
    }
  }
  
  # Initial estimates
  xb <- as.vector(x %*% beta)
  residuals <- y - 2 * atan(xb)
  S <- mean(sin(residuals))
  C <- mean(cos(residuals))
  R <- sqrt(S^2 + C^2)
  mu <- atan2(S, C)
  kappa <- A1inv(R)
  
  # Initial log-likelihood for convergence tracking
  log_lik_prev <- -n * log(besselI(kappa, 0)) + 
    kappa * sum(cos(y - mu - 2 * atan(xb)))
  
  diff <- tol + 1
  ll_diff <- ll_tol + 1
  iter <- 0
  
  # MODIFIED: Dual convergence criteria with max iterations
  while ((diff > tol) && (ll_diff > ll_tol) && (iter < max_iter)) {
    iter <- iter + 1
    beta_old <- beta
    
    xb <- as.vector(x %*% beta)
    
    # MODIFIED: Vectorized derivative calculation
    g_prime <- 2 / (1 + xb^2)
    
    residuals <- y - mu - 2 * atan(xb)
    
    a1_kappa <- A1(kappa)
    weights <- kappa * a1_kappa
    u <- kappa * sin(residuals)
    
    # MODIFIED: Efficient matrix operations
    D <- g_prime * x  # Vectorized, no diagonal matrix needed
    
    DtD <- crossprod(D)
    working_response <- u / weights + as.vector(D %*% beta)
    Dt_response <- crossprod(D, working_response)
    
    # MODIFIED: Direct solve instead of lm()
    beta <- tryCatch(
      as.vector(solve(DtD, Dt_response)),
      error = function(e) {
        as.vector(qr.solve(qr(DtD), Dt_response))
      }
    )
    
    # Update mu and kappa
    xb <- as.vector(x %*% beta)
    residuals <- y - 2 * atan(xb)
    S <- mean(sin(residuals))
    C <- mean(cos(residuals))
    R <- sqrt(S^2 + C^2)
    mu <- atan2(S, C)
    kappa <- A1inv(R)
    
    # Compute log-likelihood
    log_lik <- -n * log(besselI(kappa, 0)) + 
      kappa * sum(cos(y - mu - 2 * atan(xb)))
    
    # MODIFIED: Track both convergence criteria
    diff <- max(abs(beta - beta_old))
    ll_diff <- abs(log_lik - log_lik_prev)
    log_lik_prev <- log_lik
    
    if (verbose) {
      cat("Iteration", iter, ": Log-Likelihood =", log_lik, 
          " beta_diff =", format(diff, scientific = TRUE),
          " ll_diff =", format(ll_diff, scientific = TRUE), "\n")
    }
  }
  
  # MODIFIED: Determine and report convergence reason
  if (diff <= tol) {
    converged <- TRUE
    converge_reason <- "beta"
  } else if (ll_diff <= ll_tol) {
    converged <- TRUE
    converge_reason <- "log-likelihood"
  } else {
    converged <- FALSE
    converge_reason <- "max_iter"
    warning(paste("Algorithm did not converge after", max_iter, "iterations."))
  }
  
  # ... remainder of function (standard error calculations unchanged)
}
```

### Full comparison between original and optimized function

``` r
# ==============================================================================
# BENCHMARK COMPARISON: Original vs Optimized lm.circular.cl
# ==============================================================================

rm(list = ls(all = TRUE))

library(circular)
library(here)

# Source the optimized function
source(here("circular/lm_circular_cl_fast.R"))

# ==============================================================================
# GENERATE TEST DATA (100× larger = 1000 observations)
# ==============================================================================

set.seed(1234)

# Original size
n_original <- 10

# Scaled size (100×)
n_scaled <- 10000

# True parameters
beta_true <- c(5, 1)
kappa_true <- 100

# Generate large dataset
x_large <- cbind(rnorm(n_scaled), rep(1, n_scaled))

# Generate circular response
linear_pred <- x_large %*% beta_true
y_large <- circular(2 * atan(linear_pred)) + rvonmises(n_scaled, mu = circular(0), kappa = kappa_true)

cat("\n")
cat("==============================================================================\n")
cat("BENCHMARK: Original vs Optimized Circular-Linear Regression\n")
cat("==============================================================================\n")
cat("\nDataset size:", n_scaled, "observations (100× original)\n")
cat("True beta:", beta_true, "\n")
cat("True kappa:", kappa_true, "\n")

# ==============================================================================
# TIMING COMPARISON
# ==============================================================================

cat("\n------------------------------------------------------------------------------\n")
cat("TIMING COMPARISON\n")
cat("------------------------------------------------------------------------------\n")

# Time the ORIGINAL function
cat("\nRunning ORIGINAL lm.circular.cl...\n")
time_original <- system.time({
  result_original <- lm.circular(
    y = y_large, 
    x = x_large, 
    init = beta_true,
    type = 'c-l', 
    verbose = FALSE,
    tol = 1e-6
  )
})

cat("Original function completed.\n")
cat("Time:", time_original["elapsed"], "seconds\n")

# Time the OPTIMIZED function
cat("\nRunning OPTIMIZED lm_circular_cl_fast...\n")
time_optimized <- system.time({
  result_optimized <- lm_circular_cl_fast(
    y = y_large, 
    x = x_large, 
    init = beta_true, 
    verbose = FALSE
  )
})

cat("Optimized function completed.\n")
cat("Time:", time_optimized["elapsed"], "seconds\n")

# Calculate speedup
speedup <- time_original["elapsed"] / time_optimized["elapsed"]
cat("\n>>> SPEEDUP:", round(speedup, 2), "×\n")

# ==============================================================================
# RESULTS COMPARISON
# ==============================================================================

cat("\n------------------------------------------------------------------------------\n")
cat("RESULTS COMPARISON\n")
cat("------------------------------------------------------------------------------\n")

comparison <- data.frame(
  Parameter = c("beta[1]", "beta[2]", "mu", "kappa", 
                "se.beta[1]", "se.beta[2]", "se.mu", "se.kappa",
                "Log-Likelihood"),
  Original = c(
    result_original$coefficients[1],
    result_original$coefficients[2],
    as.numeric(result_original$mu),
    result_original$kappa,
    result_original$se.coef[1],
    result_original$se.coef[2],
    result_original$se.mu,
    result_original$se.kappa,
    result_original$log.lik
  ),
  Optimized = c(
    result_optimized$coefficients[1],
    result_optimized$coefficients[2],
    as.numeric(result_optimized$mu),
    result_optimized$kappa,
    result_optimized$se.coef[1],
    result_optimized$se.coef[2],
    result_optimized$se.mu,
    result_optimized$se.kappa,
    result_optimized$log.lik
  )
)

comparison$Difference <- comparison$Original - comparison$Optimized
comparison$RelDiff_Pct <- abs(comparison$Difference / comparison$Original) * 100

print(comparison, digits = 6)

# ==============================================================================
# CONVERGENCE DETAILS
# ==============================================================================

cat("\n------------------------------------------------------------------------------\n")
cat("CONVERGENCE DETAILS\n")
cat("------------------------------------------------------------------------------\n")

cat("\nOptimized function:\n")
cat("  Converged:", result_optimized$converged, "\n")
cat("  Convergence reason:", result_optimized$converge_reason, "\n")
cat("  Iterations:", result_optimized$iterations, "\n")

# ==============================================================================
# SCALING TEST (Multiple sample sizes)
# ==============================================================================

cat("\n------------------------------------------------------------------------------\n")
cat("SCALING TEST (Multiple sample sizes)\n")
cat("------------------------------------------------------------------------------\n")

sample_sizes <- c(100, 500, 1000, 2000, 5000)
scaling_results <- data.frame(
  n = integer(),
  time_original = numeric(),
  time_optimized = numeric(),
  speedup = numeric()
)

for (n in sample_sizes) {
  cat("Testing n =", n, "... ")
  
  # Generate data
  set.seed(1234)
  x_test <- cbind(rnorm(n), rep(1, n))
  y_test <- circular(2 * atan(x_test %*% beta_true)) + 
    rvonmises(n, mu = circular(0), kappa = kappa_true)
  
  # Time original
  t_orig <- system.time({
    lm.circular(y = y_test, x = x_test, init = beta_true, 
                type = 'c-l', verbose = FALSE, tol = 1e-6)
  })["elapsed"]
  
  # Time optimized
  t_opt <- system.time({
    lm_circular_cl_fast(y = y_test, x = x_test, init = beta_true, verbose = FALSE)
  })["elapsed"]
  
  scaling_results <- rbind(scaling_results, data.frame(
    n = n,
    time_original = t_orig,
    time_optimized = t_opt,
    speedup = t_orig / t_opt
  ))
  
  cat("done (speedup:", round(t_orig / t_opt, 1), "×)\n")
}

cat("\nScaling Results:\n")
print(scaling_results, digits = 4)

# ==============================================================================
# SUMMARY
# ==============================================================================

cat("\n==============================================================================\n")
cat("SUMMARY\n")
cat("==============================================================================\n")
cat("\nResults are numerically equivalent:", 
    all(comparison$RelDiff_Pct < 0.01, na.rm = TRUE), "\n")
cat("\nSpeedup increases with sample size due to O(n) vs O(n²) complexity.\n")
cat("\n==============================================================================\n")
```

------------------------------------------------------------------------

*Document prepared for supplementary materials* *Last updated: January
2025*
