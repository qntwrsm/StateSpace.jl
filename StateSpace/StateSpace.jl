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
	
# Types
struct Filter{Aa<:AbstractArray, AP<:AbstractArray, AF<:AbstractArray}
	a::Aa
	P::AP
	v::Matrix{Float64}
	F::AF
	K::Array{Float64,3}
end

struct FilterWb{Aa<:AbstractArray, AP<:AbstractArray, AF<:AbstractArray}
	a::Aa
	P::AP
	v::Matrix{Float64}
	Fi::AF
	K::Array{Float64,3}
end

struct Smoother{A<:AbstractArray}
	α::Matrix{Float64}
	V::A
end
	
struct SysMat{MT<:AbstractMatrix, MH<:AbstractMatrix, MQ<:AbstractMatrix, MP<:AbstractMatrix}
	Z::Matrix{Float64}
	T::MT
	H::MH
	Q::MQ
	a1::Vector{Float64}
	P1::MP
end

# Include programs
include("filter.jl")
include("smoother.jl")
include("forecast.jl")
include("loglik.jl")

end