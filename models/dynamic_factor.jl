#=
dynamic_factor.jl

    Dynamic factor model specification in state space formulation

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/07
=#

"""
    DynamicFactor <: StateSpaceModel

Constructor for a dynamic factor model instance of the state space model type
with hyperparameters `Λ`, `ϕ`, `Σε`, and `Ση`.
"""
struct DynamicFactor{Ty, Tr, TΛ, Tϕ, TΣε, TΣη} <: StateSpaceModel
    y::Ty   # data
    r::Tr   # number of factors
    Λ::TΛ   # Loading matrix
    ϕ::Tϕ   # autoregressive parameters
    Σε::TΣε # observation equation variance
    Ση::TΣη # factor equation variance
end

function DynamicFactor(y::AbstractMatrix, r::Integer)
    # Get dims
    n= size(y,1)

    # Initialize model components
    Λ= similar(y, n, r)
    ϕ= Diagonal(similar(y, r))
    Σε= Diagonal(similar(y, n))
    Ση= Diagonal(similar(y, r))

    return DynamicFactor(y, r, Λ, ϕ, Σε, Ση)
end

function get_parameters!(ψ::AbstractVector, model::DynamicFactor)
    # Get dims
    (n,r)= size(model.Λ)

    idx= 0  # index counter

    # Store values
    # Λ
    ψ[1:n*r].= vec(model.Λ)
    idx+= n*r

    # ϕ
    ψ[idx+1:idx+r].= model.ϕ.diag
    idx+= r

    # Σε
    if model.Σϵ isa Diagonal
        ψ[idx+1:idx+n].= model.Σε.diag
        idx+= n
    else
        ψ[idx+1:idx+n^2].= vec(model.Σε)
        idx+= n^2
    end

    # Ση
    ψ[idx+1:idx+r].= model.Ση.diag
    idx+= r

    return nothing
end

function get_system!(sys::StateSpaceSystem, model::DynamicFactor)
    T= eltype(model.y)

    # Store values
    sys.Z.= model.Λ
    sys.T.diag.= model.ϕ.diag
    sys.d.= zero(T)
    sys.c.= zero(T)
    sys.H.= model.Σε
    sys.Q.diag.= model.Ση.diag

    return nothing
end

function store_parameters!(model::DynamicFactor, ψ::AbstractVector)
    # Get dims
    (n,r)= size(model.Λ)

    idx= 0  # index counter

    # Store values
    # Λ
    vec(model.Λ).= view(ψ,1:n*r)
    idx+= n*r

    # ϕ
    model.ϕ.diag.= view(ψ,idx+1:idx+r)
    idx+= r

    # Σε
    if model.Σϵ isa Diagonal
        model.Σε.diag.= view(ψ,idx+1:idx+n)
        idx+= n
    else
        vec(model.Σε).= view(ψ,idx+1:idx+n^2)
        idx+= n^2
    end

    # Ση
    model.Ση.diag.= view(ψ,idx+1:idx+r)
    idx+= r

    return nothing
end

function store_system!(model::DynamicFactor, sys::StateSpaceSystem)
    # Store values
    model.Λ.= sys.Z
    model.ϕ.diag.= sys.T.diag
    model.Σε.= sys.H
    model.Ση.diag.= sys.Q.diag

    return nothing
end

function init!(model::DynamicFactor, fixed::NamedTuple)
    # Get dim
    n= size(model.y, 1)

    # Model
    init_model!(model, fixed)

    # State space system
    Te= eltype(model.y)
    Tv= Vector{Te}
    Tm= Matrix{Te}
    Td= Diagonal{Te}
    sys= StateSpaceSystem{Tm, Td, Tv, Tv, typeof(model.Σε), Td, Tv, Tm}(n, model.r)
    init_system!(sys, model)

    return sys
end

function init_model!(model::DynamicFactor, fixed::NamedTuple)
    # Get dim
    T= size(model.y,2)

    # Principal component analysis
    (pc, loadings)= pca(model.y, model.r)

    # Λ
    model.Λ.= haskey(fixed, :Λ) ? fixed.Λ : loadings
    # ϕ
    y= view(pc,:,2:T)
    X= view(pc,:,1:T-1)
    ϕ_diag= vec(sum(y .* X, dims=2) ./ sum(abs2, X, dims=2))
    model.ϕ.= haskey(fixed, :ϕ) ? fixed.ϕ : Diagonal(ϕ_diag)
    # Σε
    model.Σε.= haskey(fixed, :Σε) ? fixed.Σε : Diagonal(vec(var(model.y, dims=2)))
    # Ση
    model.Ση.= haskey(fixed, :Ση) ? fixed.Ση : Diagonal(vec(var(pc, dims=2)))

    return nothing
end

function init_system!(sys::StateSpaceSystem, model::DynamicFactor)
    T= eltype(model.y)
    # a
    sys.a1.= zero(T)
    # P
    sys.P1.= zero(T)
    @inbounds @fastmath for i in 1:model.r
        sys.P1[i,i]= model.Ση.diag[i] * inv(one(T) - model.ϕ.diag[i]^2)
    end
    
    return nothing
end
