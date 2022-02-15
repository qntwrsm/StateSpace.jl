#=
dynamic_factor_spatial_error.jl

    Dynamic factor model with spatial errors specification in state space 
    formulation.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/08
=#

"""
    DynamicFactorSpatialError <: DynamicFactor

Constructor for a dynamic factor model with spatial errors instance of the state
space model type with hyperparameters `Λ`, `ϕ`, `Σξ`, `Ση`, `ρ`, and `W`.
"""
struct DynamicFactorSpatialError{Ty, Tr, TΛ, Tϕ, TΣε, TΩε, TΣξ, TΩξ, TΣη, Tρ, TW, TG, Tg} <: DynamicFactor
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
end
function DynamicFactorSpatialError(y::AbstractMatrix, r::Integer, groups::AbstractVector, W::AbstractMatrix)
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

    return DynamicFactorSpatialError(y, r, Λ, ϕ, Σε, Ωε, Σξ, Ωξ, Ση, ρ, W, G, groups)
end

# Methods
function number_parameters(model::DynamicFactorSpatialError)
    # Get dims
    (n,r)= size(model.Λ)
    g= length(model.ρ)

    n_params= n*r + r + r + (model.Σε isa Diagonal ? n : n*(n+1)÷2) + g

    return n_params
end

function get_parameters!(ψ::AbstractVector, model::DynamicFactorSpatialError)
    # Get dims
    (n,r)= size(model.Λ)
    g= length(model.ρ)

    idx= 0  # index counter

    # Store values
    # Λ
    ψ[1:n*r].= vec(model.Λ)
    idx+= n*r

    # ϕ
    ψ[idx+1:idx+r].= model.ϕ.diag
    idx+= r

    # Σξ
    if model.Σξ isa Diagonal
        ψ[idx+1:idx+n].= model.Σξ.diag
        idx+= n
    else
        k= 0
        @inbounds @fastmath for j in 1:n
            for i in 1:j
                k+= 1
                ψ[idx+k]= model.Σξ[i,j]
            end
        end
        idx+= n*(n+1)÷2
    end

    # Ση
    ψ[idx+1:idx+r].= model.Ση.diag
    idx+= r

    # ρ
    ψ[idx+1:idx+g].= model.ρ

    return nothing
end

function get_system!(sys::StateSpaceSystem, model::DynamicFactorSpatialError)
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

function store_parameters!(model::DynamicFactorSpatialError, ψ::AbstractVector)
    # Get dims
    (n,r)= size(model.Λ)
    g= length(model.ρ)

    idx= 0  # index counter

    # Store values
    # Λ
    vec(model.Λ).= view(ψ,1:n*r)
    idx+= n*r

    # ϕ
    model.ϕ.diag.= view(ψ,idx+1:idx+r)
    idx+= r

    # Σξ
    if model.Σξ isa Diagonal
        model.Σξ.diag.= view(ψ,idx+1:idx+n)
        idx+= n
    else
        k= 0
        @inbounds @fastmath for j in 1:n
            for i in 1:j
                k+= 1
                model.Σξ.data[i,j]= ψ[idx+k]
            end
        end
        idx+= n*(n+1)÷2
    end

    # Ση
    model.Ση.diag.= view(ψ,idx+1:idx+r)
    idx+= r

    # ρ
    model.ρ.= view(ψ,idx+1:idx+g)

    return nothing
end

function init_model!(model::DynamicFactorSpatialError, fixed::NamedTuple, constraints::NamedTuple)
    # Get dim
    T= size(model.y,2)
    g= length(model.ρ)

    # Principal component analysis
    (pc, loadings)= pca(model.y, model.r)

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
    logistic(x; offset=0.0, scale=1.0)

Evaluate the logistic function at point `x`, with optionally an `offset` and
`scale`, mapping `x` ``∈ [-∞, +∞]`` to [-`offset`, `scale`-`offset`].

#### Arguments
  - `x::Real`       : input
  - `offset::Real`  : offset
  - `scale::Real`   : scale

#### Returns
  - `y::Real`   : output
"""
logistic(x::Real; offset::Real=0.0, scale::Real=1.0)= scale * inv(one(x) + exp(-x)) - offset

"""
    logit(x; offset=0.0, scale=1.0)

Evaluate the logit function (inverse logistic) at point `x`, with optionally an
`offset` and `scale`, mapping `x` ``∈`` [-`offset`, `scale`-`offset`] to ``[-∞,
+∞]``.

#### Arguments
  - `x::Real`       : input
  - `offset::Real`  : offset
  - `scale::Real`   : scale

#### Returns
  - `y::Real`   : output
"""
logit(x::Real; offset::Real=0.0, scale::Real=1.0)= log(scale * (x + offset) * inv(one(x) - scale * (x + offset)))

"""
    f_ρ(ρ, model, state, smoother)

Compute objective function value `f` w.r.t. loading matrix ``ρ`` (E-step joint
average negative log-likelihood w.r.t. ``ρ``).

#### Arguments
  - `ρ::AbstractVector`                 : spatial dependence
  - `model::DynamicFactorSpatialError`  : state space model
  - `state::EMState`                    : state variables
  - `smoother::Smoother`                : Kalman smoother output

#### Returns
  - `f::Real`   : objective function value
"""
function f_ρ(ρ::AbstractVector, model::DynamicFactorSpatialError, state::EMState, 
                smoother::Smoother)
    # Get dims
    g= length(ρ)

    # inverse of G
    idx= 0
    @inbounds @fastmath for i in 1:g
        rng= max(idx,1):idx+model.groups[i]
        model.G[rng,:].= logistic(ρ[i]; offset=one(ρ[i]), scale=2*one(ρ[i])) .* view(model.W,rng,:)
        idx+= model.groups[i]
    end
    @inbounds @fastmath for i in axes(model.G,1)
        model.G[i,i]+= one(eltype(model.G)) 
    end

    # Precision matrix
    @inbounds @fastmath for j in axes(model.Ωε,2)
        G_j= view(model.G,:,j)
        for i in 1:j
            G_i= view(model.G,:,i)
            model.Ωε.data[i,j]= dot(G_i, model.Ωξ, G_j)
        end
    end

    # objective function
    f= zero(eltype(ρ))
    @inbounds @fastmath for j in axes(model.Λ,2)
        Λ_j= view(model.Λ,:,j)
        for k in axes(model.Λ,2)
            Λ_k= view(model.Λ,:,k)
            f+= .5 * state.V_0[k,j] * dot(Λ_k, model.Ωε, Λ_j)
        end
        α_j= vec(view(smoother.α,j,:))
        for i in axes(model.Λ,1)
            Ωε_i= view(model.Ωε,:,i)
            f-= Λ[i,j] * dot(Ωε_i, model.y, α_j)
        end
    end
    @inbounds @fastmath for i in axes(model.y,1)
        y_i= vec(view(model.y,i,:))
        Ωε_i= view(model.Ωε,:,i)
        f+= .5 * dot(Ωε_i, model.y, y_i)
    end

    f= f * inv(T) - logdet(lu!(model.G))

    return f
end

"""
    update_ρ!(model, state, smoother)

Update spatial dependence parameters `ρ`, storing the result in `model`.

#### Arguments
  - `model::DynamicFactorSpatialError`  : state space model
  - `state:::EMState`                   : state variables
  - `smoother::Smoother`                : Kalman smoother output
"""
function update_ρ!(model::DynamicFactorSpatialError, state::EMState, smoother::Smoother)
    # Precision matrix
    model.Ωξ.= model.Σξ
    LinearAlgebra.inv!(cholesky!(model.Ωξ))

    # Get dims
    g= length(model.ρ)

    # Infer type
    Tρ= eltype(model.ρ) 

    # Closure of objective function
    f(x::AbstractVector)= f_ρ(x, model, state, smoother)

    # Transform parameters
    model.ρ.= logit.(model.ρ, one(Tρ), inv(2*one(Tρ)))

    # Small dimensions: BFGS
    if g < 10 
        res= optimize(f, model.ρ, BFGS(); autodiff=:forward)
    # Large dimensions: L-BFGS
    else 
        res= optimize(f, model.ρ, LBFGS(); autodiff=:forward)
    end

    # Store results
    model.ρ.= logistic.(Optim.minimizer(res), one(Tρ), 2*one(Tρ))

    return nothing
end

"""
    update_Σξ!(model, state, smoother)

Update observation equation error variance parameters `Σξ`, storing the result
in `model`.

#### Arguments
  - `model::DynamicFactorSpatialError`  : state space model
  - `state:::EMState`                   : state variables
  - `smoother::Smoother`                : Kalman smoother output
"""
update_Σξ!(model::DynamicFactorSpatialError, state::EMState, smoother::Smoother)= update_Σε!(model, state, smoother)

function update_model!(model::DynamicFactorSpatialError, state::EMState, 
                        smoother::Smoother, fixed::NamedTuple)
    # Update hyperparameters
    haskey(fixed, :ϕ) ? nothing : update_ϕ!(model, state) 
    haskey(fixed, :Ση) ? nothing : update_Ση!(model, state, smoother)
    haskey(fixed, :Σξ) ? nothing : update_Σξ!(model, state, smoother)
    haskey(fixed, :ρ) ? nothing : update_ρ!(model, state, smoother)
    haskey(fixed, :Λ) ? nothing : update_Λ!(model, state, smoother)

    # Update G
    @inbounds @fastmath for i in 1:g
        rng= max(idx,1):idx+model.groups[i]
        model.G[rng,:].= ρ[i] .* view(model.W,rng,:)
        idx+= model.groups[i]
    end
    @inbounds @fastmath for i in axes(model.G,1)
        model.G[i,i]+= one(eltype(model.G)) 
    end
    LinearAlgebra.inv!(lu!(model.G))

    # Update Σε
    @inbounds @fastmath for j in axes(model.Σε,2)
        G_j= view(model.G,:,j)
        for i in 1:j
            G_i= view(model.G,:,i)
            model.Σε.data[i,j]= dot(G_i, model.Σξ, G_j)
        end
    end

    return nothing
end
