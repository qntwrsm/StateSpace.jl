#=
dynamic_factor_spatial_error.jl

    Dynamic factor model with special errors specification in state space 
    formulation

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/08
=#

"""
    DynamicFactorSpatialError <: StateSpaceModel

Constructor for a dynamic factor model with spatial errors instance of the state
space model type with hyperparameters `Λ`, `ϕ`, `Σε`, `Ση`, `ρ`, and `W`.
"""
mutable struct DynamicFactorSpatialError{Ty, Tr, Tg, TΛ, Tϕ, TΣε, TΣη, Tρ, TW} <: StateSpaceModel
    y::Ty   # data
    r::Tr   # number of factors
    g::Tg   # number of spatial groups
    Λ::TΛ   # Loading matrix
    ϕ::Tϕ   # autoregressive parameters
    Σε::TΣε # observation equation variance
    Ση::TΣη # factor equation variance
    ρ::Tρ   # spatial dependence
    W::TW   # spatial weight matrix 
end

function DynamicFactorSpatialError(y::AbstractMatrix, r::Integer, W::AbstractMatrix)
    # Get dims
    n= size(y,1)
    g= size(W,1)

    # Initialize model components
    Λ= similar(y, n, r)
    ϕ= Diagonal(similar(y, r))
    Σε= Diagonal(similar(y, n))
    Ση= Diagonal(similar(y, r))
    ρ= similar(y, g)

    return DynamicFactorSpatialError(y, r, g, Λ, ϕ, Σε, Ση, ρ, W)
end

function get_parameters!(ψ::AbstractVector, model::DynamicFactorSpatialError)
    # Get dims
    (n,r)= size(model.Λ)

    # Store values
    ψ[1:n*r].= vec(model.Λ)

    if model.ϕ isa Diagonal
        ψ[n*r+1:(n+1)*r].= model.ϕ.diag
    else
        ψ[n*r+1:(n+r)*r].= vec(model.ϕ)
    end

    if model.Σϵ isa Diagonal
        ψ[n*r+1:(n+1)*r].= model.ϕ.diag
    else

    end

end

function get_system!(sys::StateSpaceSystem, model::DynamicFactorSpatialError)
    
    return nothing
end

function store_parameters!(model::DynamicFactorSpatialError, ψ::AbstractVector)
    
    return nothing
end

function store_system!(model::DynamicFactorSpatialError, sys::StateSpaceSystem)
    
    return nothing
end
