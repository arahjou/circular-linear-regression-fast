#############################################################
#                                                           #
#   lm_circular_cl_fast - Optimized Circular-Linear         #
#   Regression with dual convergence criteria               #
#                                                           #
#   Based on original by Claudio Agostinelli                #
#   Optimized version with fixes for convergence issues     #
#                                                           #
#############################################################

lm_circular_cl_fast <- function(y, x, init = NULL, verbose = FALSE, 
                                tol = 1e-10, max_iter = 500,
                                ll_tol = 1e-6,
                                control.circular = list()) {
  
  
  # Handle missing values
  ok <- complete.cases(x, y)
  if (NCOL(x) == 1) {
    x <- x[ok]
  } else {
    x <- x[ok, , drop = FALSE]
  }
  y <- y[ok]
  
  n <- length(y)
  if (n == 0) {
    warning("No observations (at least after removing missing values)")
    return(NULL)
  }
  
  if (is.null(init)) {
    stop("'init' is missing with no default")
  }
  
  if (is.vector(x)) {
    x <- matrix(x, ncol = 1)
  }
  
  if (NCOL(x) != length(init)) {
    stop("'init' must have the same number of elements as the columns of 'x'")
  }
  
  # Handle circular data properties
  if (requireNamespace("circular", quietly = TRUE) && circular::is.circular(y)) {
    datacircularp <- circular::circularp(y)
  } else {
    datacircularp <- list(type = "angles", units = "radians", template = "none",
                          modulo = "asis", zero = 0, rotation = "counter")
  }
  
  dc <- control.circular
  dc$type <- dc$type %||% datacircularp$type
  dc$units <- dc$units %||% datacircularp$units
  dc$template <- dc$template %||% datacircularp$template
  dc$modulo <- dc$modulo %||% datacircularp$modulo
  dc$zero <- dc$zero %||% datacircularp$zero
  dc$rotation <- dc$rotation %||% datacircularp$rotation
  
  # Convert to radians
  if (requireNamespace("circular", quietly = TRUE) && circular::is.circular(y)) {
    y <- circular::conversion.circular(y, units = "radians", zero = 0, 
                                       rotation = "counter", modulo = "2pi")
    attr(y, "circularp") <- attr(y, "class") <- NULL
  }
  
  # Call optimized core function
  result <- lm_circular_cl_core(y, x, init, verbose, tol, max_iter, ll_tol)
  
  if (is.null(result)) {
    return(NULL)
  }
  
  # Convert output back
  if (requireNamespace("circular", quietly = TRUE)) {
    result$mu <- circular::conversion.circular(
      circular::circular(result$mu), 
      dc$units, dc$type, dc$template, dc$modulo, dc$zero, dc$rotation
    )
  }
  if (dc$units == "degrees") {
    result$se.mu <- result$se.mu * 180 / pi
  }
  
  result$call <- match.call()
  class(result) <- "lm.circular.cl"
  return(result)
}

#############################################################
# Optimized core algorithm with dual convergence criteria
#############################################################

lm_circular_cl_core <- function(y, x, init, verbose, tol, max_iter, ll_tol) {
  
  n <- length(y)
  p <- NCOL(x)
  y <- y %% (2 * pi)
  
  beta <- init
  x <- as.matrix(x)
  
  # Helper function: A1 (ratio of Bessel functions I1/I0)
  A1 <- function(k) {
    if (k < 1e-6) return(k / 2)
    besselI(k, 1) / besselI(k, 0)
  }
  
  # Helper function: Inverse of A1 (approximation)
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
  
  # Initial log-likelihood
  log_lik_prev <- -n * log(besselI(kappa, 0)) + 
    kappa * sum(cos(y - mu - 2 * atan(xb)))
  
  diff <- tol + 1
  ll_diff <- ll_tol + 1
  iter <- 0
  
  # Converge when EITHER beta stabilizes OR log-likelihood stabilizes
  while ((diff > tol) && (ll_diff > ll_tol) && (iter < max_iter)) {
    iter <- iter + 1
    beta_old <- beta
    
    # Compute linear predictor
    xb <- as.vector(x %*% beta)
    
    # Vectorized derivative: g'(xb) = 2 / (1 + xb^2)
    g_prime <- 2 / (1 + xb^2)
    
    # Residuals
    residuals <- y - mu - 2 * atan(xb)
    
    # Working response and weights (vectorized)
    a1_kappa <- A1(kappa)
    weights <- kappa * a1_kappa
    u <- kappa * sin(residuals)
    
    # Weighted design matrix (D = diag(g_prime) %*% x, computed efficiently)
    D <- g_prime * x
    
    # Solve weighted least squares
    DtD <- crossprod(D)
    working_response <- u / weights + as.vector(D %*% beta)
    Dt_response <- crossprod(D, working_response)
    
    # Solve for new beta with error handling
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
    
    # Compute convergence criteria
    diff <- max(abs(beta - beta_old))
    ll_diff <- abs(log_lik - log_lik_prev)
    log_lik_prev <- log_lik
    
    if (verbose) {
      cat("Iteration", iter, ": Log-Likelihood =", log_lik, 
          " beta_diff =", format(diff, scientific = TRUE),
          " ll_diff =", format(ll_diff, scientific = TRUE), "\n")
    }
  }
  
  # Determine convergence reason
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
  
  # Final calculations
  xb <- as.vector(x %*% beta)
  g_prime <- 2 / (1 + xb^2)
  D <- g_prime * x
  a1_kappa <- A1(kappa)
  
  log_lik <- -n * log(besselI(kappa, 0)) + 
    kappa * sum(cos(y - mu - 2 * atan(xb)))
  
  # Covariance matrix of beta
  DtD <- crossprod(D)
  cov_beta <- tryCatch(
    solve(DtD) / (kappa * a1_kappa),
    error = function(e) matrix(NA, p, p)
  )
  se_beta <- sqrt(pmax(diag(cov_beta), 0))
  
  # Standard errors for kappa and mu
  se_kappa <- sqrt(1 / (n * (1 - a1_kappa^2 - a1_kappa / kappa)))
  se_mu <- 1 / sqrt((n - p) * kappa * a1_kappa)
  
  # Test statistics
  t_values <- abs(beta / se_beta)
  p_values <- 2 * (1 - pnorm(t_values))
  
  list(
    x = x,
    y = y,
    mu = mu,
    se.mu = se_mu,
    kappa = kappa,
    se.kappa = se_kappa,
    coefficients = beta,
    cov.coef = cov_beta,
    se.coef = se_beta,
    log.lik = log_lik,
    t.values = t_values,
    p.values = p_values,
    converged = converged,
    converge_reason = converge_reason,
    iterations = iter
  )
}

#############################################################
# Print method for lm.circular.cl objects
#############################################################

print.lm.circular.cl <- function(x, digits = max(3, getOption("digits") - 3), 
                                 signif.stars = getOption("show.signif.stars"), ...) {
  cat("\nCall:\n", deparse(x$call), "\n\n", sep = "") 
  
  result.matrix <- cbind(x$coefficients, x$se.coef, x$t.values, x$p.values)
  dimnames(result.matrix) <- list(
    if (!is.null(colnames(x$x))) colnames(x$x) else paste0("X", 1:ncol(x$x)),
    c("Estimate", "Std. Error", "t value", "Pr(>|t|)")
  )
  
  cat("\n Circular-Linear Regression \n")
  cat("\n Coefficients:\n")
  printCoefmat(result.matrix, digits = digits, signif.stars = signif.stars, ...)
  cat("\n")
  cat(" Log-Likelihood: ", format(x$log.lik, digits = digits), "\n")
  cat("\n Summary:\n")
  cat("  mu: ", format(x$mu, digits = digits), " (", format(x$se.mu, digits = digits), ")\n", sep = "")
  cat("  kappa: ", format(x$kappa, digits = digits), " (", format(x$se.kappa, digits = digits), ")\n", sep = "")
  
  if (!is.null(x$converged)) {
    cat("\n Convergence: ", ifelse(x$converged, "Yes", "No"), sep = "")
    if (!is.null(x$converge_reason)) {
      cat(" (", x$converge_reason, ")", sep = "")
    }
    if (!is.null(x$iterations)) {
      cat(" in ", x$iterations, " iterations", sep = "")
    }
    cat("\n")
  }
  
  cat("\n p-values are approximated using normal distribution\n\n")
  invisible(x)
}

#############################################################
# Null coalescing operator (if not already defined)
#############################################################

if (!exists("%||%")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

