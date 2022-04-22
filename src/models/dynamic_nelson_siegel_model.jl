#=
dynamic_nelson_siegel_model.jl

    Dynamic Nelson Siegel model specification in state space formulation

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/03/28
=#

"""
    DynamicNelsonSiegelModel <: StateSpaceModel

Constructor for a dynamic Nelson Siegel model instance of the state space model type
with hyperparameters `λ`, `ϕ`, and error specifications
"""
mutable struct DynamicNelsonSiegelModel{Ty, Tτ, Tλ, Tϕ, Teobs, Tefac} <: StateSpaceModel
    y::Ty               # data
    τ::Tτ               # maturities
    λ::Tλ               # decay
    ϕ::Tϕ               # autoregressive parameters
    error_obs::Teobs    # obs. eq. error specification
    error_factor::Tefac # factor eq. error specification
end
# Constructors
function DynamicNelsonSiegelModel(y::AbstractMatrix, τ::AbstractVector, 
                                    error_obs::AbstractErrorModel, error_factor::AbstractErrorModel)    
    # hyper paremeters
    λ= .0609
    ϕ= similar(y, 3, 3)

    return DynamicNelsonSiegelModel(y, τ, λ, ϕ, error_obs, error_factor)
end

# Methods
cov(model::DynamicNelsonSiegelModel)= cov(model.error_obs)
prec(model::DynamicNelsonSiegelModel)= prec(model.error_obs)
resid(model::DynamicNelsonSiegelModel)= resid(model.error_obs)
maturities(model::DynamicNelsonSiegelModel)= model.τ
decay(model::DynamicNelsonSiegelModel)= model.λ

function loadings!(Λ::AbstractMatrix, model::DynamicNelsonSiegelModel)
    λ= decay(model)
    τ= maturities(model)
    n= length(τ)

    Λ.= one(eltype(Λ))
    @inbounds @fastmath for i in 1:n
        Λ[i,2]= (1 - exp(-λ * τ[i])) * inv(λ * τ[i])
        Λ[i,3]= Λ[i,2] - exp(-λ * τ[i])
    end

    return nothing
end
function loadings(model::DynamicNelsonSiegelModel)
    # get dims
    n= length(maturities(model))

    Λ= Matrix{typeof(decay(model))}(undef, n, 3)
    loadings!(Λ, model)

    return Λ
end

# State space system and hyperparameters
nparams(model::DynamicNelsonSiegelModel)= 10 + nparams(model.error_obs) + nparams(model.error_factor)

function get_params!(ψ::AbstractVector, model::DynamicNelsonSiegelModel)
    idx= 0  # index counter

    # Store values
    # λ
    ψ[1]= model.λ
    idx+= 1

    # ϕ
    ψ[idx+1:idx+9].= vec(model.ϕ)
    idx+= 9

    # observation equation error model
    get_params!(view(ψ,idx+1:idx+nparams(model.error_obs)), model.error_obs)
    idx+= nparams(model.error_obs)

    # factor equation error model
    get_params!(view(ψ,idx+1:idx+nparams(model.error_fac)), model.error_factor)

    return nothing
end

function get_system!(sys::LinearTimeInvariant, model::DynamicNelsonSiegelModel, method::Symbol)
    Λ= loadings(model)  # loadings

    # Store values
    sys.T.= model.ϕ
    sys.Q.= cov(model.error_factor)
    if method === :univariate
        # Cholesky decomposition of H
        C= cholesky(cov(model))

        # Store values
        ldiv!(sys.y, C.L, model.y)
        ldiv!(sys.Z, C.L, Λ)
    elseif method === :collapsed
        # Collapsing transformation
        # Z'×H⁻¹×Z
        tmp_pn= transpose(Λ) * prec(model)
        tmp= tmp_pn * Λ
        # (Z'×H⁻¹×Z)⁻¹ 
        # perform pseudo inverse as Λ can contain zero columns
        pseudo= pinv(tmp)
        # pivoted Cholesky decomposition
        C= cholesky!(Hermitian(pseudo), Val(true), check=false)
        # transformation
        ip= invperm(C.p)
        U= C.U[:,ip]
        A= U * tmp_pn

        # Store values
        mul!(sys.y, A, model.y)
        sys.Z.= pinv(transpose(U))
    else
        # Store values
        sys.Z.= Λ
        sys.H.= cov(model)
    end

    return nothing
end

function store_params!(model::DynamicNelsonSiegelModel, ψ::AbstractVector)
    idx= 0  # index counter

    # Store values
    # λ
    model.λ= ψ[1]
    idx+= 1

    # ϕ
    vec(model.ϕ).= view(ψ,idx+1:idx+9)
    idx+= 9

    # observation equation error model
    store_params!(model.error_obs, view(ψ,idx+1:idx+nparams(model.error_obs)))
    idx+= nparams(model.error_obs)

    # factor equation error model
    store_params!(model.error_factor, view(ψ,idx+1:idx+nparams(model.error_factor)))

    return nothing
end

# Initialization
function init!(model::DynamicNelsonSiegelModel, init::NamedTuple, method::Symbol)
    # Model
    init_model!(model, init)

    # State space system
    create_system(model, method)
    fix_system!(sys, model, method)
    init_system!(sys, model)

    return sys
end

function init_factors(y::AbstractMatrix, Λ::AbstractMatrix)
    # Get dims
    T= size(y,2)

    # Result container
    β= similar(y, 3, T)

    # OLS
    A= transpose(Λ) * Λ
	B= A \ transpose(Λ)
    mul!(β, B, y)

    return β
end

"""
    init_ϕ!(ϕ, β)

Initialize autoregressive parameters `ϕ` using OLS with factors, storing the 
result in `ϕ`.

#### Arguments
  - `β::AbstractMatrix` : factors

#### Returns
  - `ϕ::AbstractMatrix` : autoregressive parameters
"""
function init_ϕ!(ϕ::AbstractMatrix, β::AbstractMatrix)
    # Get dims
    T= size(β,2)

    # OLS
    X= view(β,:,1:T-1)
    y= view(β,:,2:T)
    ϕ.= y / X

    return nothing
end

function init_model!(model::DynamicNelsonSiegelModel, init::NamedTuple)
    # get dims
    T= size(model.y,2) 

    # λ
    haskey(init, :λ) ? model.λ= init.λ : nothing
    # factors
    Λ= loadings(model)
    β= init_factors(model.y, Λ)

    # observation equation residuals
    resid(model).= model.y
    mul!(resid(model), Λ, β, -1., 1.)

    # ϕ
    haskey(init, :ϕ) ? model.ϕ.= init.ϕ : init_ϕ!(model.ϕ, β)

    # factor equation residuals
    resid(model.error_factor).= view(β,:,2:T)
    mul!(resid(model.error_factor), model.ϕ, view(β,:,1:T-1), -1., 1.)

    # observation equation error model
    init_error!(model.error_obs, init)

    # factor equation error model
    init_error!(model.error_factor, init)

    return nothing
end

function create_system(model::DynamicNelsonSiegelModel, method::Symbol)
    # get dims
    (n,T_len)= method === :collapsed ? (3, size(model.y,2)) : size(model.y)

    # types
    Te= eltype(model.y)
    Tv= Vector{Te}
    Tm= Matrix{Te}
    Td= Diagonal{Te}
    TQ= cov(model.error_factor) isa Symmetric ? Symmetric{Te} : Td
    if method === :univariate || method === :collapsed
        TH= Td
    else
        TH= cov(model) isa Symmetric ? Symmetric{Te} : Td
    end

    # State space system
    sys= LinearTimeInvariant{Tm, Tm, Tm, Tv, Tv, TH, TQ, Tv, Tm}(n, 3, T_len)

    return sys
end

function fix_system!(sys::LinearTimeInvariant, model::DynamicNelsonSiegelModel, method::Symbol)
    # infer type
    T= eltype(model.y)

    if method === :multivariate || method === :woodbury
        sys.y.= model.y
    end
    if method === :univariate || method === :collapsed
        sys.H.diag.= one(T)
    end
    sys.d.= zero(T)
    sys.c.= zero(T)
end

function init_system!(sys::StateSpaceSystem, model::DynamicNelsonSiegelModel)
    T= eltype(model.y)
    # a
    sys.a1.= zero(T)
    # P
    sys.P1.= zero(T)
    @inbounds @fastmath for i in 1:3
        sys.P1[i,i]= one(T)
    end
    
    return nothing
end

# Estimation
# Log-likelihood
function loglik(filter::KalmanFilter, sys::StateSpaceSystem, model::DynamicNelsonSiegelModel, method::Symbol)
    # Compute log-likelihood
    ll= loglik(filter)

    # Account for transformation
    if method === :univariate
        # get dims
        T_len= size(model.y,2)

        # Cholesky decomposition of H
        C= cholesky(cov(model))

        # Add Jacobian determinant term
        d,s= logabsdet(C.L)
        ll-= T_len * (d + log(s))
    elseif method === :collapsed
        # get dims
        (n,T_len)= size(model.y)

        Λ= loadings(model)  # loadings

        # Cholesky decomposition of H
        C= cholesky(cov(model))

        # Add Jacobian determinant term
        ll-= .5 * T_len * logdet(C)

        # Add projected out term
        ll-= .5 * (n - model.r) * T_len * log(2*π)
        e= similar(model.y, n)
        @inbounds @fastmath for t in 1:T_len
            e.= view(model.y,:,t)
            mul!(e, Λ, view(sys.y,:,t), -1., 1.)
            ll-= .5 * dot(e, prec(model), e)
        end
    end

    return ll
end

# Forecast
function forecast(model::DynamicNelsonSiegelModel, h::Integer)
    # number of time series
    n= size(model.y,1)

    # create forecast variables
    y_f= hcat(model.y, fill(NaN, n, h))

    # reinstantiate
    forecast_model= DynamicNelsonSiegelModel(y_f, model.τ, model.λ, model.ϕ, model.error_obs, model.error_factor)
    
    # State space system
    sys= create_system(forecast_model, :multivariate)
    fix_system!(sys, forecast_model, :multivariate)
    init_system!(sys, forecast_model)
    get_system!(sys, forecast_model, :multivariate)

    # forecasts
    f= forecast(sys, h)

    return f
end