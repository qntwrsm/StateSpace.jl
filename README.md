# StateSpace

This is a package for state filtering, smoothing, and parameter estimation in state space models.

Provides methods for a state space model such as filtering (Kalman filter), smoothing (Kalman smoother), forecasting, likelihood evaluation, and estimation of hyperparameters (Maximum Likelihood, Expectation-Maximization (EM), and Expectation-Conditional Maximization (ECM), w/ and w/o penalization).

Currently only supports filtering, smoothing, and estimation for linear Gaussian state space models.

# Filtering and Smoothing

The following filter methods are supported `:univariate`, `:collapsed`, `:multivariate`, and `:woodbury`, which correspond to the following filter types

- `UnivariateFilter`: Filter using the univariate treatment for a linear Gaussian state space model.
- `MultivariateFilter`: Standard multivariate filter for a linear Gaussian state space model.
- `WoodburyFilter`: Same as `MultivariateFilter`, but uses the Woodbury identity to compute the inverse.

The smoother type is

- `Smoother`: General state smoothing, which accepts both multivariate and univariate filters and handles arbitrary state autocovariance smoothing.

# Estimation

# Documentation

# Installation

[![Build Status](https://github.com/qntwrsm/StateSpace.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/qntwrsm/StateSpace.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/qntwrsm/StateSpace.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/qntwrsm/StateSpace.jl)
