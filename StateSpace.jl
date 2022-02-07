#=
StateSpace.jl

    Provides methods for a linear Gaussian State Space model such as filtering
    (Kalman filter), smoothing (Kalman smoother), forecasting, likelihood
    evaluation, and estimation of hyperparameters (Maximum Likelihood, EM, and 
	penalized EM)

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2021/12/06
=#

module StateSpace

using LinearAlgebra

export 
# Structs
	MultivariateFilter, 
	WoodburyFilter, 
	UnivariateFilter,
	Smoother,
	SysMat,
# Filter
	kalman_filter!,
# Smoother
	kalman_smoother!,
	kalman_smoother_cov!,
# Forecast
	forecast, forecast!,
# Log Likelihood
	loglik
	
# Structs
# KalmanFilter
struct KalmanFilter end

# State Space model system matrices 
struct SysMat{TZ, TT, TH, TQ, ta, TP}
	Z::TZ	# system matrix Z
	T::TT	# system matrix T
	H::TH	# system matrix H
	Q::TQ	# system matrix Q
	a1::TA	# initial state
	P1::TP	# initial state variance
end

# Include programs
# Filters
include("filters/multivariate_filter.jl")
include("filters/univariate_filter.jl")

# Smoother
include("smoother.jl")

# Forecast
include("forecast.jl")

# Log-likelihood
include("loglik.jl")

end