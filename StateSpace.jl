#=
StateSpace.jl

    Provides methods for a linear Gaussian State Space model such as filtering
    (Kalman filter), smoothing (Kalman smoother), forecasting, likelihood
    evaluation, and maximum likelihood estimation

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2021/12/06
=#

module StateSpace

using LinearAlgebra

export 
# Structs
	Filter, 
	FilterWb, 
	Smoother,
	SysMat,
# Filter
	kalmanfilter, kalmanfilter!,
	kalmanfilter_wb, kalmanfilter_wb!,
	kalmanfilter_eq, kalmanfilter_eq!,
# Smoother
	kalmansmoother, kalmansmoother!,
	kalmansmoother_cov, kalmansmoother_cov!,
	kalmansmoother_eq, kalmansmoother_eq!,
	kalmansmoother_cov_eq, kalmansmoother_cov_eq!,
# Forecast
	forecast, forecast!,
# Log Likelihood
	loglik,
	loglik_eq
	
# Structs
# Kalman filter
struct Filter{Aa<:AbstractArray, AP<:AbstractArray, AF<:AbstractArray}
	a::Aa				# filtered state
	P::AP				# filtered state variance
	v::Matrix{Float64}  # forecast error
	F::AF				# forecast error variance
	K::Array{Float64,3}	# Kalman gain
end
# Kalman filter based on Woodbury Identity
struct FilterWb{Aa<:AbstractArray, AP<:AbstractArray, AF<:AbstractArray}
	a::Aa				# filtered state
	P::AP				# filtered state variance
	v::Matrix{Float64}	# forecast error
	Fi::AF				# inverse forecast error variance
	K::Array{Float64,3}	# Kalman gain
end
# Kalman smoother
struct Smoother{A<:AbstractArray}
	α::Matrix{Float64}	# smoothed state
	V::A				# smoothed state covariances
end
# State Space model system matrices 
struct SysMat{MT<:AbstractMatrix, MH<:AbstractMatrix, MQ<:AbstractMatrix, MP<:AbstractMatrix}
	Z::Matrix{Float64}	# system matrix Z
	T::MT				# system matrix T
	H::MH				# system matrix H
	Q::MQ				# system matrix Q
	a1::Vector{Float64}	# initial state
	P1::MP				# initial state variance
end

# Include programs
include("filter.jl")
include("smoother.jl")
include("forecast.jl")
include("loglik.jl")

end