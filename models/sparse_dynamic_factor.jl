#=
sparse_dynamic_factor.jl

    Sparse dynamic factor model specification in state space formulation, where 
    sparsity is imposed on the loading matrix.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/10
=#

"""
    SparseDynamicFactor <: DynamicFactor

Constructor for a sparse dynamic factor model instance of the state space model
type with hyperparameters `Λ`, `ϕ`, `Σε`, and `Ση` and penalization components
`pen`.
"""
struct SparseDynamicFactor{Ty, Tr, TΛ, Tϕ, TΣε, TΩε, TΣη, Tp} <: DynamicFactor
    y::Ty   # data
    r::Tr   # number of factors
    Λ::TΛ   # Loading matrix
    ϕ::Tϕ   # autoregressive parameters
    Σε::TΣε # observation equation variance
    Ωε::TΩε # observation equation precision
    Ση::TΣη # factor equation variance
    pen::Tp # penalization variables
end
# Constructor
function SparseDynamicFactor(y::AbstractMatrix, r::Integer, 
                            constraints::NamedTuple, pen::Penalization)
    # Get dims
    n= size(y,1)

    # Initialize model components
    Λ= similar(y, n, r)
    ϕ= Diagonal(similar(y, r))
    Σε= haskey(constraints, :Σε) ? Diagonal(similar(y, n)) : Symmetric(similar(y, n, n))
    Ωε= similar(Σε, n, n)
    Ση= Diagonal(similar(y, r))

    return SparseDynamicFactor(y, r, Λ, ϕ, Σε, Ωε, Ση, pen)
end

# Methods
# Initialization
function init_model!(model::SparseDynamicFactor, fixed::NamedTuple, constraints::NamedTuple)
    # Get dim
    T= size(model.y,2)

    # Sparse principal component analysis
    (pc, loadings)= spca(model.y, model.r, model.pen.γ)

    # Λ
    model.Λ.= haskey(fixed, :Λ) ? fixed.Λ : loadings
    # ϕ
    y= view(pc,:,2:T)
    X= view(pc,:,1:T-1)
    ϕ_diag= vec(sum(X .* y, dims=2) ./ sum(abs2, X, dims=2))
    model.ϕ.= haskey(fixed, :ϕ) ? fixed.ϕ : Diagonal(ϕ_diag)
    # Σε
    if haskey(constraints, :Σε)
        model.Σε.= haskey(fixed, :Σε) ? fixed.Σε : Diagonal(vec(var(model.y, dims=2)))
    else
        model.Σε.= haskey(fixed, :Σε) ? fixed.Σε : Symmetric(diagm(vec(var(model.y, dims=2))))
    end
    # Ση
    model.Ση.= haskey(fixed, :Ση) ? fixed.Ση : Diagonal(vec(var(pc, dims=2)))

    return nothing
end

# Estimation
"""
    f_Λ(λ, model, state, smoother)

Compute objective function value `f` w.r.t. loading matrix ``Λ`` (E-step joint
average negative log-likelihood w.r.t. ``Λ``).

#### Arguments
  - `λ::AbstractVector`         : vectorized loading matrix
  - `model::SparseDynamicFactor`: state space model
  - `state::EMState`            : state variables
  - `smoother::Smoother`        : Kalman smoother output

#### Returns
  - `f::Real`   : objective function value
"""
function f_Λ(λ::AbstractVector, model::DynamicFactor, state::EMState, smoother::Smoother)
    # Get dim
    T= size(model.y,2)

    # objective function
    f= zero(eltype(λ))
    Λ= reshape(λ, size(model.Λ))
    @inbounds @fastmath for j in axes(Λ,2)
        Λ_j= view(Λ,:,j)
        for k in axes(Λ,2)
            Λ_k= view(Λ,:,k)
            f+= .5 * state.V_0[k,j] * dot(Λ_k, Ωε, Λ_j)
        end
        α_j= vec(view(smoother.α,j,:))
        for i in axes(Λ,1)
            Ωε_i= view(model.Ωε,:,i)
            f-= Λ[i,j] * dot(Ωε_i, model.y, α_j)
        end
    end

    return f * inv(T)
end

"""
    ∇f_Λ!(∇f, λ, model, state, smoother)

Compute gradient `∇f` w.r.t. loading matrix ``Λ`` (gradient of E-step joint
average negative log-likelihood w.r.t. ``Λ``), storing the result in `∇f`.

#### Arguments
  - `λ::AbstractVector`         : vectorized loading matrix
  - `model::SparseDynamicFactor`: state space model
  - `state::EMState`            : state variables
  - `smoother::Smoother`        : Kalman smoother output

#### Returns
  - `∇f::AbstractVector`    : gradient 
"""
function ∇f_Λ!(∇f::AbstractVector, λ::AbstractVector, model::DynamicFactor, 
                state::EMState, smoother::Smoother)
    # Get dim
    T= size(model.y,2)
    
    # gradient
    k= 0
    Λ= reshape(λ, size(model.Λ))    # reshape to matrix
    @inbounds @fastmath for j in axes(Λ,2)
        α_j= vec(view(smoother.α,j,:))
        Λ_j= view(Λ,:,j)
        for i in axes(Λ,1)
            k+= 1
            Ωε_i= view(model.Ωε,:,i)
            ∇f[k]= -dot(Ωε_i, model.y, α_j)
            a= dot(Ωε_i, Λ_j)
            for l in axes(Λ,2)
                ∇f[k]+= state.V_0[l,j] * a
            end
        end
    end
    # scale
    lmul!(inv(T), ∇f)

    return nothing
end

"""
    prox!(x, λ, pen)

Compute proximal operatior for type `pen` at point `x` with scale `λ`, storing
the result in `x`.

#### Arguments
  - `x::AbstractVector` : input
  - `λ::Real`           : scaling parameter
  - `pen::Penalization` : penalization variables
"""
prox!(x::AbstractVector, λ::Real, pen::Lasso)= x.= soft_thresh.(x, λ .* pen.γ .* vec(pen.weights))

function prox!(x::AbstractVector, λ::Real, pen::GroupLasso)
    idx= 0
    for g in 1:length(pen.groups)
        rng= max(idx,1):idx+pen.groups[g]
        x_g= view(x,rng)
        block_soft_thresh!(x_g, λ * pen.γ * pen.weights[g])
        idx+= pen.groups[g]
    end 

    return nothing
end 

function prox!(x::AbstractVector, λ::Real, pen::SparseGroupLasso)
    idx= 0 
    x.= soft_thresh.(x, λ .* pen.γ .* pen.α .* vec(pen.weights_l1))
    for g in 1:length(pen.groups)
        rng= max(idx,1):idx+pen.groups[g]
        x_g= view(x,rng)
        block_soft_thresh!(x_g, λ * pen.γ * (one(α) - α) * pen.weights_l2[g])
        idx+= pen.groups[g]
    end

    return nothing
end 

"""
    update_Λ!(model, state, smoother)

Update loading matrix parameters `Λ` with sparsity imposed through penalization,
storing the result in `model`.

#### Arguments
  - `model::SparseDynamicFactor`: state space model
  - `state:::EMState`           : state variables
  - `smoother::Smoother`        : Kalman smoother output
"""
function update_Λ!(model::SparseDynamicFactor, state::EMState, smoother::Smoother)
    # Precision matrix
    model.Ωε.= model.Σε
    LinearAlgebra.inv!(cholesky!(model.Ωε))

    # Closure of functions
    f(x::AbstractVector)= f_Λ(x, model, state, smoother)
    ∇f!(∇f::AbstractVector, x::AbstractVector)=  ∇f_Λ!(∇f, x, model, state, smoother)
    prox_op!(x::AbstractVector, λ::Real)= prox!(x, λ, model.pen)

    # Penalized estimation via proximal gradient method
    prox_grad!(vec(model.Λ), f, ∇f!, prox_op!; style="nesterov")
    
    return nothing
end