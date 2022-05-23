#=
StateSpace.jl

    Provides methods for a linear Gaussian State Space model such as filtering
    (Kalman filter), smoothing (Kalman smoother), forecasting, likelihood
    evaluation, and estimation of hyperparameters (Maximum Likelihood, 
	Expectation-Maximization (EM), and Expectation-Conditional Maximization 
	(ECM), w/ and w/o penalization)

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2021/12/06
=#

module StateSpace

using 	LinearAlgebra, 
		Statistics, 
		Optim, 
		ProximalMethods, 
		FiniteDifferences,
        MultivariateStats

export 
# Structs
	# filters and smoother 
	MultivariateFilter, 
	WoodburyFilter, 
	UnivariateFilter,
	Smoother,
	# system
	LinearTimeInvariant,
	LinearTimeVariant,
	# penalization
	NoPen,
	Lasso, 
	GroupLasso,
	SparseGroupLasso,
	# models
	Independent, Idiosyncratic, SpatialErrorModel, SpatialMovingAverageModel,
	NoConstant, Constant, Exogeneous,
	DynamicFactorModel,
    DynamicNelsonSiegelModel,
# State space system
    create_system,
    init_system!,
    fix_system!,
# Filter
	kalman_filter!,
# Smoother
	kalman_smoother!,
	kalman_smoother_cov!,
# Forecast
	forecast, forecast!,
# Log Likelihood
	loglik,
# Estimation
	maximum_likelihood!,
	em!,
	ecm!
	
# Types
"""
    KalmanFilter

Abstract type for Kalman filters.
"""
abstract type KalmanFilter end

"""
    KalmanSmoother

Abstract type for Kalman smoothers.
"""
abstract type KalmanSmoother end

"""
    StateSpaceModel

Abstract type for state space models.
"""
abstract type StateSpaceModel end

# Include programs
# System
include("system.jl")

# Filters
include("filters/multivariate_filter.jl")
include("filters/univariate_filter.jl")

# Smoother
include("smoother.jl")

# Forecast
include("forecast.jl")

# Log-likelihood
include("loglik.jl")

# Estimation
include("estim/estim.jl")
# ML
include("estim/ml/ml.jl")
# EM
include("estim/em/em.jl")
include("estim/em/mstep.jl")
include("estim/em/estep.jl")

# Models
include("models/state_space.jl")
# error models
include("models/error_model.jl")
# mean models
include("models/mean_model.jl")
# Dynamic factor models
include("models/dynamic_factor_model.jl")
include("models/dynamic_nelson_siegel_model.jl")

end