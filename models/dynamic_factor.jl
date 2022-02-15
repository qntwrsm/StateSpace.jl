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
# Constructor
function DynamicFactor(y::AbstractMatrix, r::Integer, constraints::NamedTuple)
    # Get dims
    n= size(y,1)

    # Initialize model components
    Λ= similar(y, n, r)
    ϕ= Diagonal(similar(y, r))
    Σε= haskey(constraints, :Σε) ? Diagonal(similar(y, n)) : Symmetric(similar(y, n, n))
    Ση= Diagonal(similar(y, r))

    return DynamicFactor(y, r, Λ, ϕ, Σε, Ση)
end

# Methods
# State space system and hyperparameters
function number_parameters(model::DynamicFactor)
    # Get dims
    (n,r)= size(model.Λ)

    n_params= n*r + r + r + (model.Σε isa Diagonal ? n : n*(n+1)÷2)

    return n_params
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
        k= 0
        @inbounds @fastmath for j in 1:n
            for i in 1:j
                k+= 1
                ψ[idx+k]= model.Σε[i,j]
            end
        end
        idx+= n*(n+1)÷2
    end

    # Ση
    ψ[idx+1:idx+r].= model.Ση.diag

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
    if model.Σε isa Diagonal
        model.Σε.diag.= view(ψ,idx+1:idx+n)
        idx+= n
    else
        k= 0
        @inbounds @fastmath for j in 1:n
            for i in 1:j
                k+= 1
                model.Σε.data[i,j]= ψ[idx+k]
            end
        end
        idx+= n*(n+1)÷2
    end

    # Ση
    model.Ση.diag.= view(ψ,idx+1:idx+r)

    return nothing
end

# Initialization
function init!(model::DynamicFactor, fixed::NamedTuple, constraints::NamedTuple)
    # Get dim
    n= size(model.y, 1)

    # Model
    init_model!(model, fixed, constraints)

    # State space system
    Te= eltype(model.y)
    Tv= Vector{Te}
    Tm= Matrix{Te}
    Td= Diagonal{Te}
    sys= StateSpaceSystem{Tm, Td, Tv, Tv, typeof(model.Σε), Td, Tv, Tm}(n, model.r)
    init_system!(sys, model)

    return sys
end

function init_model!(model::DynamicFactor, fixed::NamedTuple, constraints::NamedTuple)
    # Get dim
    T= size(model.y,2)

    # Principal component analysis
    (pc, loadings)= pca(model.y, model.r)

    # Λ
    model.Λ.= haskey(fixed, :Λ) ? fixed.Λ : loadings
    # ϕ
    y= view(pc,:,2:T)
    X= view(pc,:,1:T-1)
    ϕ_diag= vec(sum(X .* y, dims=2) ./ sum(abs2, X, dims=2))
    model.ϕ.diag.= haskey(fixed, :ϕ) ? fixed.ϕ.diag : ϕ_diag
    # Σε
    if haskey(constraints, :Σε)
        model.Σε.diag.= haskey(fixed, :Σε) ? fixed.Σε.diag : vec(var(model.y, dims=2))
    else
        model.Σε.= haskey(fixed, :Σε) ? fixed.Σε : Symmetric(diagm(vec(var(model.y, dims=2))))
    end
    # Ση
    model.Ση.diag.= haskey(fixed, :Ση) ? fixed.Ση.diag : vec(var(pc, dims=2))

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

# Estimation
"""
    update_ϕ!(model, state)

Update autoregressive parameters `ϕ`, storing the result in `model`.

#### Arguments
  - `model::DynamicFactor`  : state space model
  - `state:::EMState`       : state variables
"""
function update_ϕ!(model::DynamicFactor, state::EMState)
    @inbounds @fastmath for i in axes(model.ϕ,1)
        model.ϕ.diag[i]= state.V_01[i,i] * inv(state.V_1[i,i])
    end

    return nothing
end

"""
    update_Ση!(model, state, smoother)

Update factor equation error variance parameters `Ση`, storing the result in
`model`.

#### Arguments
  - `model::DynamicFactor`  : state space model
  - `state:::EMState`       : state variables
  - `smoother::Smoother`    : Kalman smoother output
"""
function update_Ση!(model::DynamicFactor, state::EMState, smoother::Smoother)
    T= size(model.y,2)
    @inbounds @fastmath for i in axes(model.Ση,1)
        model.Ση.diag[i]= inv(T - 1) * ( state.V_0[i,i] - 
                                        smoother.V[i,i,1,1] - 
                                        smoother.α[i,1]^2 - 
                                        state.V_01[i,i]^2 * inv(state.V_1[i,i]) ) 
    end

    return nothing
end

"""
    update_Σε!(model, state, smoother)

Update observation equation error variance parameters `Σε`, storing the result
in `model`.

#### Arguments
  - `model::DynamicFactor`  : state space model
  - `state:::EMState`       : state variables
  - `smoother::Smoother`    : Kalman smoother output
"""
function update_Σε!(model::DynamicFactor, state::EMState, smoother::Smoother)
    T= size(model.y,2)
    # Diagonal
    if model.Σε isa Diagonal
        @inbounds @fastmath for i in axes(model.Σε,1)
            y_i= vec(view(model.y,i,:))
            Λ_i= vec(view(model.Λ,i,:))
            model.Σε.diag[i]= inv(T) * ( sum(abs2, y_i) - 
                                        2 * dot(Λ_i, smoother.α, y_i) +
                                        dot(Λ_i, state.V_0, Λ_i) )
        end
    # Symmetric
    else
        @inbounds @fastmath for j in axes(model.Σε,2)
            y_j= vec(view(model.y,j,:))
            Λ_j= vec(view(model.Λ,j,:))
            for i in 1:j
                y_i= vec(view(model.y,i,:))
                Λ_i= vec(view(model.Λ,i,:))
                model.Σε.data.[i,j]= inv(T) * ( dot(y_i, y_j) - 
                                                dot(Λ_i, smoother.α, y_j) - 
                                                dot(Λ_j, smoother.α, y_i) +
                                                dot(Λ_i, state.V_0, Λ_j) )
            end
        end
    end

    return nothing
end

"""
    update_Λ!(model, state, smoother)

Update loading matrix parameters `Λ`, storing the result in `model`.

#### Arguments
  - `model::DynamicFactor`  : state space model
  - `state:::EMState`       : state variables
  - `smoother::Smoother`    : Kalman smoother output
"""
function update_Λ!(model::DynamicFactor, state::EMState, smoother::Smoother)
    # Generalized Ridge regression
    mul!(model.Λ, y, transpose(smoother.α))
    fac= cholesky!(Hermitian(state.V_0))
    rdiv!(model.Λ, fac)
    
    return nothing
end

function update_model!(model::DynamicFactor, state::EMState, smoother::Smoother, 
                        fixed::NamedTuple)
    # Update hyperparameters
    haskey(fixed, :ϕ) ? nothing : update_ϕ!(model, state) 
    haskey(fixed, :Ση) ? nothing : update_Ση!(model, state, smoother)
    haskey(fixed, :Σε) ? nothing : update_Σε!(model, state, smoother)
    haskey(fixed, :Λ) ? nothing : update_Λ!(model, state, smoother)

    return nothing
end