#=
sparse_dynamic_factor_spatial_error.jl

    Sparse dynamic factor model with spatial errors specification in state space 
    formulation.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/10
=#

"""
    SparseDynamicFactorSpatialError <: DynamicFactorSpatialError

Constructor for a sparse dynamic factor model with spatial errors instance of
the state space model type with hyperparameters `Λ`, `ϕ`, `Σε`, `Ση`, `ρ`, and
`W` and penalization components `pen_Λ` and `pen_ρ`.
"""
struct SparseDynamicFactorSpatialError{Ty, Tr, TΛ, Tϕ, TΣε, TΩε, TΣξ, TΩξ, TΣη, Tρ, TW, TG, TpΛ, Tpρ} <: DynamicFactorSpatialError
    y::Ty       # data
    r::Tr       # number of factors
    Λ::TΛ       # Loading matrix
    ϕ::Tϕ       # autoregressive parameters
    Σε::TΣε     # observation equation variance
    Ωε::TΩε     # observation equation precision
    Σξ::TΣξ     # observation equation error variance
    Ωξ::TΩξ     # observation equation error precision
    Ση::TΣη     # factor equation variance
    ρ::Tρ       # spatial dependence
    W::TW       # spatial weight matrix
    G::TG       # inverse spatial lag polynomial      
    groups::Tg  # group structure
    pen_Λ::TpΛ  # penalization variables
    pen_ρ::Tpρ  # penalization variables
end
function SparseDynamicFactorSpatialError(y::AbstractMatrix, r::Integer, 
                                        groups::AbstractVector, W::AbstractMatrix,
                                        pen_Λ::Penalization, pen_ρ::Penalization)
    # Get dims
    n= size(y,1)
    g= length(groups)

    # Initialize model components
    Λ= similar(y, n, r)
    ϕ= Diagonal(similar(y, r))
    Σε= Symmetric(similar(y, n, n))
    Ωε= similar(Σε)
    Σξ= haskey(constraints, :Σξ) ? Diagonal(similar(y, n)) : Symmetric(similar(y, n, n))
    Ωξ= similar(Σξ)
    Ση= Diagonal(similar(y, r))
    ρ= similar(y, g)
    G= similar(y, n, n)

    return SparseDynamicFactorSpatialError(y, r, Λ, ϕ, Σε, Ωε, Σξ, Ωξ, Ση, ρ, W, G, groups, pen_Λ, pen_ρ)
end

# Methods
function init_model!(model::SparseDynamicFactorSpatialError, fixed::NamedTuple, constraints::NamedTuple)
    # Get dim
    T= size(model.y,2)
    g= length(model.ρ)

    # Principal component analysis
    (pc, loadings)= spca(model.y, model.r, model.pen_Λ.γ)

    # errors
    ε= copy(y)
    mul!(ε, loadings, pc, -1., 1.)

    # Λ
    model.Λ.= haskey(fixed, :Λ) ? fixed.Λ : loadings
    # ϕ
    y= view(pc,:,2:T)
    X= view(pc,:,1:T-1)
    ϕ_diag= vec(sum(X .* y, dims=2) ./ sum(abs2, X, dims=2))
    model.ϕ.= haskey(fixed, :ϕ) ? fixed.ϕ : Diagonal(ϕ_diag)
    # ρ
    ρ_max= max(1/maximum(sum(model.W, dims=1)), 1/maximum(sum(model.W, dims=2)))
    Wε= model.W*ε
    idx= 0
    @inbounds @fastmath for i in 1:g
        rng= max(idx,1):idx+model.groups[i]
        ρ_tmp= dot(view(Wε,rng,:), view(ε,rng,:)) / sum(abs2, view(Wε,rng,:))
        model.ρ[i]= max(ρ_tmp, .9 * ρ_max)
        idx+= model.groups[i]
    end
    # Σξ
    idx= 0
    @inbounds @fastmath for i in 1:g
        rng= max(idx,1):idx+model.groups[i]
        model.G[rng,:].= model.ρ[i] .* view(model.W,rng,:)
        idx+= model.groups[i]
    end
    ξ= transpose(G)*ε
    if haskey(constraints, :Σξ)
        model.Σξ.= haskey(fixed, :Σξ) ? fixed.Σξ : Diagonal(vec(var(ξ, dims=2)))
    else
        model.Σξ.= haskey(fixed, :Σξ) ? fixed.Σξ : Symmetric(diagm(vec(var(ξ, dims=2))))
    end
    # Ση
    model.Ση.= haskey(fixed, :Ση) ? fixed.Ση : Diagonal(vec(var(pc, dims=2)))

    return nothing
end

# Estimation
"""
    update_ρ!(model, state, smoother)

Update spatial dependence parameters `ρ` with sparsity imposed through
penalization, storing the result in `model`.

#### Arguments
  - `model::SparseDynamicFactorSpatialError`: state space model
  - `state:::EMState`                       : state variables
  - `smoother::Smoother`                    : Kalman smoother output
"""
function update_ρ!(model::SparseDynamicFactorSpatialError, state::EMState, smoother::Smoother)
    # Precision matrix
    model.Ωξ.= model.Σξ
    LinearAlgebra.inv!(cholesky!(model.Ωξ))

    # Infer type
    Tρ= eltype(model.ρ) 

    # Closure of objective function and proximal operator
    f(x::AbstractVector)= f_ρ(x, model, state, smoother)
    prox_op!(x::AbstractVector, λ::Real)= prox!(x, λ, model.pen_ρ)

    # gradient
    ∇f!(∇f::AbstractVector, x::AbstractVector)= ForwardDiff.gradient!(∇f, f, x)

    # Transform parameters
    model.ρ.= logit.(model.ρ, one(Tρ), inv(2*one(Tρ)))

    # Penalized estimation via proximal gradient method
    prox_grad!(model.ρ, f, ∇f!, prox_op!; style="nesterov")

    # Transform parameters back
    model.ρ.= logistic.(model.ρ, one(Tρ), 2*one(Tρ))

    return nothing
end

function update_Λ!(model::SparseDynamicFactorSpatialError, state::EMState, smoother::Smoother)
    # inverse of G
    idx= 0
    @inbounds @fastmath for i in 1:g
        rng= max(idx,1):idx+model.groups[i]
        model.G[rng,:].= model.ρ[i] .* view(model.W,rng,:)
        idx+= model.groups[i]
    end
    @inbounds @fastmath for i in axes(model.G,1)
        model.G[i,i]+= one(eltype(model.G)) 
    end

    # Precision matrix
    @inbounds @fastmath for j in axes(model.Σξ,2)
        G_j= view(model.G,:,j)
        for i in 1:j
            G_i= view(model.G,:,i)
            model.Ωε.data[i,j]= dot(G_i, model.Ωξ, G_j)
        end
    end

    # Closure of functions
    f(x::AbstractVector)= f_Λ(x, model, state, smoother)
    ∇f!(∇f::AbstractVector, x::AbstractVector)=  ∇f_Λ!(∇f, x, model, state, smoother)
    prox_op!(x::AbstractVector, λ::Real)= prox!(x, λ, model.pen_Λ)

    # Penalized estimation via proximal gradient method
    prox_grad!(vec(model.Λ), f, ∇f!, prox_op!; style="nesterov")
    
    return nothing
end