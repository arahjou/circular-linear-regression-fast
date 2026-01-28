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