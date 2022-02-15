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

using LinearAlgebra, Statistics, ForwardDiff, Optim

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
	
# Type
abstract type KalmanFilter end
abstract type Penalization end

# Struct
# State space model system
struct StateSpaceSystem{TZ, TT, Td, Tc, TH, TQ, Ta, TP}
	Z::TZ	# system matrix Z
	T::TT	# system matrix T
	d::Td	# mean adjustment observation
	c::Tc	# mean adjustment state
	H::TH	# system matrix H
	Q::TQ	# system matrix Q
	a1::Ta	# initial state
	P1::TP	# initial state variance
end
# Constructor
function StateSpaceSystem{TZ, TT, Td, Tc, TH, TQ, Ta, TP}(n::Integer, p::Integer) where {TZ, TT, Td, Tc, TH, TQ, Ta, TP}
    # Initialize system components
    Z= similar(TZ, n, p)
    T= TT <: Diagonal ? Diagonal{eltype(TT)}(undef, p) : similar(TT, p, p)
	d= similar(Td, n)
	c= similar(Tc, p)
    H= TH <: Diagonal ? Diagonal{eltype(TH)}(undef, n) : similar(TH, n, n)
    Q= TQ <: Diagonal ? Diagonal{eltype(TQ)}(undef, p) : similar(TQ, p, p)
	a1= similar(Ta, p)
	P1= TP <: Diagonal ? Diagonal{eltype(TP)}(undef, p) : similar(TP, p, p)

    return StateSpaceSystem(Z, T, d, c, H, Q, a1, P1)
end

# Penalization
struct Lasso{Tγ, Tw} <: Penalization
	γ::Tγ		# penalization strength
	weights::Tw	# weights
end
struct GroupLasso{Tγ, Tw, Tg} <: Penalization
	γ::Tγ		# penalization strength
	weights::Tw	# weights
	groups::Tg	# group structure
end
struct SparseGroupLasso{Tγ, Tw, Tg} <: Penalization
	γ::Tγ			# penalization strength
	α::Tγ			# mixing parameter
	weights_l1::Tw	# weights ℓ₁-norm
	weights_l2::Tw	# weights ℓ₂-norm
	groups::Tg		# group structure
end

# Include programs
# Models
include("models/state_space.jl")
# Dynamic factor models
include("models/dynamic_factor.jl")
include("models/sparse_dynamic_factor.jl")
include("models/dynamic_factor_spatial_error.jl")
include("models/sparse_dynamic_factor_spatial_error.jl")

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
# EM
include("estim/em/em.jl")
include("estim/em/mstep.jl")
include("estim/em/estep.jl")
# ML

end