#=
dynamic_factor_model.jl

    Dynamic factor model specification in state space formulation

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/07
=#

"""
    DynamicFactorModel <: StateSpaceModel

Constructor for a dynamic factor model instance of the state space model type
with hyperparameters `Λ` and `ϕ`, error specification, and mean
specification.
"""
struct DynamicFactorModel{Ty, Tr, Tm, TΛ, Tϕ, Te} <: StateSpaceModel
    y::Ty       # data
    r::Tr       # number of factors
    mean::Tm    # mean specification
    Λ::TΛ       # Loading matrix
    ϕ::Tϕ       # autoregressive parameters
    error::Te   # error specification
end
# Constructors
function DynamicFactorModel(
    y::AbstractMatrix, 
    r::Integer, 
    mean::AbstractMeanModel, 
    error::AbstractErrorModel
)
    # Get dims
    n= size(y,1)
    
    # hyper paremeters
    Λ= similar(y, n, r)
    ϕ= Diagonal(similar(y, r))

    return DynamicFactorModel(y, r, mean, Λ, ϕ, error)
end

# Methods
mean(model::DynamicFactorModel)= mean(model.mean)
mean!(model::DynamicFactorModel)= mean!(model.mean)
cov(model::DynamicFactorModel)= cov(model.error)
cov!(model::DynamicFactorModel)= cov!(model.error)
prec(model::DynamicFactorModel)= prec(model.error)
prec!(model::DynamicFactorModel)= prec!(model.error)
resid(model::DynamicFactorModel)= resid(model.error)

# State space system and hyperparameters
function nparams(model::DynamicFactorModel)
    # Get dims
    (n,r)= size(model.Λ)

    n_params= n*r + r + nparams(model.error) + nparams(model.mean)

    return n_params
end

function get_params!(ψ::AbstractVector, model::DynamicFactorModel)
    # Get dims
    (n,r)= size(model.Λ)
    idx= 0  # index counter

    # Store values
    # Λ
    ψ[1:n*r].= vec(model.Λ)
    idx+= n*r

    # ϕ
    ψ[idx+1:idx+r].= logit.(model.ϕ.diag, offset=1., scale=2)
    idx+= r

    # mean model
    get_params!(view(ψ,idx+1:idx+nparams(model.mean)), model.mean)
    idx+= nparams(model.mean)

    # error model
    get_params!(view(ψ,idx+1:idx+nparams(model.error)), model.error)

    return nothing
end

function get_system!(
    sys::LinearTimeInvariant, 
    model::DynamicFactorModel, 
    method::Symbol
)
    # Store values
    sys.T.= model.ϕ
    if method === :univariate
        # Cholesky decomposition of H
        C= cholesky(cov(model))

        # Store values
        ldiv!(sys.y, C.L, model.y)
        ldiv!(sys.Z, C.L, model.Λ)
        if !isa(model.mean, NoConstant)
            ldiv!(sys.d, C.L, mean(model))
        end
    elseif method === :collapsed
        # Collapsing transformation
        # Z'×H⁻¹×Z
        tmp_pn= transpose(model.Λ) * prec(model)
        tmp= tmp_pn * model.Λ
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
        if !isa(model.mean, NoConstant)
            mul!(sys.d, A, mean(model))
        end
    else
        # Store values
        sys.Z.= model.Λ
        sys.H.= cov(model)
        if !isa(model.mean, NoConstant)
            sys.d.= mean(model)
        end
    end

    return nothing
end

function get_system!(
    sys::LinearTimeVariant, 
    model::DynamicFactorModel, 
    method::Symbol
)
    # get dims
    T_len= size(model.y,2) 

    # Store values
    @inbounds for t in 1:T_len
        sys.T[t].= model.ϕ
    end
    if method === :univariate
        # Cholesky decomposition of H
        C= cholesky(cov(model))

        # Store values
        ldiv!(sys.y, C.L, model.y)
        ldiv!(sys.Z[1], C.L, model.Λ)
        @inbounds for t in 2:T_len
            sys.Z[t].= sys.Z[1]
        end
        if !isa(model.mean, NoConstant)
            ldiv!(sys.d, C.L, mean(model))
        end
    elseif method === :collapsed
        # Collapsing transformation
        # Z'×H⁻¹×Z
        tmp_pn= transpose(model.Λ) * prec(model)
        tmp= tmp_pn * model.Λ
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
        sys.Z[1].= pinv(transpose(U))
        @inbounds for t = 2:T_len
            sys.Z[t].= sys.Z[1]
        end
        if !isa(model.mean, NoConstant)
            mul!(sys.d, A, mean(model))
        end
    else
        # Store values
        @inbounds for t = 1:T_len
            sys.Z[t].= model.Λ
            sys.H[t].= cov(model)
        end
        if !isa(model.mean, NoConstant)
            sys.d.= mean(model)
        end
    end

    return nothing
end

function store_params!(model::DynamicFactorModel, ψ::AbstractVector)
    # Get dims
    (n,r)= size(model.Λ)
    idx= 0  # index counter

    # Store values
    # Λ
    vec(model.Λ).= view(ψ,1:n*r)
    idx+= n*r

    # ϕ
    model.ϕ.diag.= logistic.(view(ψ,idx+1:idx+r), offset=1., scale=2)
    idx+= r

    # mean model
    store_params!(model.mean, view(ψ,idx+1:idx+nparams(model.mean)))
    idx+= nparams(model.mean)

    # error model
    store_params!(model.error, view(ψ,idx+1:idx+nparams(model.error)))

    return nothing
end

# Initialization
function init!(model::DynamicFactorModel, init::NamedTuple, method::Symbol)
    # Model
    init_model!(model, init)

    # State space system
    sys= create_system(model, method)
    fix_system!(sys, model, method)
    init_system!(sys, model)

    return sys
end

"""
    init_ϕ!(ϕ, pc)

Initialize autoregressive parameters `ϕ` using OLS with principal components,
storing the result in `ϕ`.

#### Arguments
  - `pc::AbstractMatrix`: principal components

#### Returns
  - `ϕ::Diagonal`   : autoregressive parameters
"""
function init_ϕ!(ϕ::Diagonal, pc::AbstractMatrix)
    # Get dims
    (r,T_len)= size(pc)
    # type
    T= eltype(pc)

    # OLS
    @inbounds @fastmath for i = 1:r
        y= view(pc,i,2:T_len)
        x= view(pc,i,1:T_len-1)
        ϕ_i= dot(x, y) * inv(sum(abs2, x))

        # threshold
        ϕ.diag[i]= .99 * sign(ϕ_i) * min(abs(ϕ_i), one(T))
    end

    return nothing
end

function init_model!(model::DynamicFactorModel, init::NamedTuple)
    # Principal component analysis
    M= fit(PCA, model.y, maxoutdim=model.r, pratio=1.)
    loadings= projection(M)
    pc= transform(M, model.y)

    # residuals
    resid(model).= model.y
    mul!(resid(model), loadings, pc, -1., 1.)

    # Λ
    model.Λ.= haskey(init, :Λ) ? init.Λ : loadings
    # ϕ
    haskey(init, :ϕ) ? model.ϕ.= init.ϕ : init_ϕ!(model.ϕ, pc)

    # mean model
    init_mean!(model.mean, model.y, init)

    # update residuals
    if !isa(model.mean, NoConstant)
        resid(model).-= mean(model)
    end

    # error model
    init_error!(model.error, init)

    return nothing
end

function create_system(model::DynamicFactorModel, method::Symbol)
    # get dims
    (n,T_len)= method === :collapsed ? (model.r, size(model.y,2)) : size(model.y)
    
    # types
    Te= eltype(model.y)
    Tv= Vector{Te}
    Tm= Matrix{Te}
    Td= Diagonal{Te}
    if method === :univariate || method === :collapsed
        TH= Td
    else
        TH= cov(model) isa Symmetric ? Symmetric{Te} : Td
    end

    # State space system
    if model.mean isa Exogeneous
        sys= LinearTimeVariant{Tm, Tm, Td, Tm, Tm, TH, Td, Tv, Tm}(n, model.r, T_len)
    else 
        sys= LinearTimeInvariant{Tm, Tm, Td, Tv, Tv, TH, Td, Tv, Tm}(n, model.r, T_len)
    end

    return sys
end

function fix_system!(
    sys::LinearTimeInvariant, 
    model::DynamicFactorModel, 
    method::Symbol
)
    # infer type
    T= eltype(model.y)

    if method === :multivariate || method === :woodbury
        sys.y.= model.y
    end
    if method === :univariate || method === :collapsed
        sys.H.diag.= one(T)
    end
    sys.Q.diag.= one(T)
    if model.mean isa NoConstant
        sys.d.= zero(T)
    end
    sys.c.= zero(T)
end

function fix_system!(
    sys::LinearTimeVariant, 
    model::DynamicFactorModel, 
    method::Symbol
)
    #  get dims
    T_len= size(model.y,2)

    # infer type
    T= eltype(model.y)

    if method === :multivariate || method === :woodbury
        sys.y.= model.y
    end
    if method === :univariate || method === :collapsed
        @inbounds for t = 1:T_len
            sys.H[t].diag.= one(T)
        end
    end
    @inbounds for t = 1:T_len
        sys.Q[t].diag.= one(T)
    end
    if model.mean isa NoConstant
        sys.d.= zero(T)
    end
    sys.c.= zero(T)
end

function init_system!(sys::StateSpaceSystem, model::DynamicFactorModel)
    T= eltype(model.y)
    # a
    sys.a1.= zero(T)
    # P
    sys.P1.= zero(T)
    @inbounds @fastmath for i = 1:model.r
        sys.P1[i,i]= one(T)
    end
    
    return nothing
end

# Estimation
# Log-likelihood
function loglik(filter::KalmanFilter, model::DynamicFactorModel, method::Symbol)
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

        # Cholesky decomposition of H
        C= cholesky(cov(model))

        # Add Jacobian determinant term
        ll-= .5 * T_len * logdet(C)

        # Z'×H⁻¹×Z
        tmp_pn= transpose(model.Λ) * prec(model)
        tmp= tmp_pn * model.Λ
        # (Z'×H⁻¹×Z)⁻¹ 
        # perform pseudo inverse as Λ can contain zero columns
        pseudo= pinv(tmp)
        # Projection
        M= I - model.Λ * pseudo * tmp_pn

        # Add projected out term
        ll-= .5 * (n - model.r) * T_len * log(2*π)
        e= similar(model.y, n)
        @inbounds @fastmath for t = 1:T_len
            mul!(e, M, view(model.y,:,t))
            ll-= .5 * dot(e, prec(model), e)
        end
    end

    return ll
end

# EM
"""
    f_ϕ(ϕ, A, B)

Compute objective function value `f` w.r.t. autoregressive parameter ``ϕ``
(negative log-likelihood w.r.t. ``ϕ``).

#### Arguments
  - `ϕ::AbstractVector` : autoreggresive parameters
  - `A::AbstractMatrix` : linear coefficients
  - `B::AbstractMatrix` : quadratic coefficients

#### Returns
  - `f::Real`   : objective function value
"""
function f_ϕ(ϕ::AbstractVector, A::AbstractMatrix, B::AbstractMatrix)
    # type
    T= eltype(ϕ)

    # objective function
    f= zero(eltype(ϕ))
    @inbounds @fastmath for i ∈ axes(ϕ,1)
        ϕ_i= logistic(ϕ[i], offset=one(T), scale=2 * one(T))
        f+= .5 * ϕ_i^2 * B[i,i] - ϕ_i * A[i,i]
    end

    return f
end

"""
    ∇f_ϕ!(∇f, ϕ, A, B)

Compute gradient `∇f` of objective function value `f` w.r.t. autoregressive
parameter ``ϕ`` (gradient of negative log-likelihood w.r.t. ``ϕ``).

#### Arguments
  - `ϕ::AbstractVector` : autoreggresive parameters
  - `A::AbstractMatrix` : linear coefficients
  - `B::AbstractMatrix` : quadratic coefficients

#### Returns
  - `∇f::AbstractVector`: gradient
"""
function ∇f_ϕ!(
    ∇f::AbstractVector, 
    ϕ::AbstractVector, 
    A::AbstractMatrix, 
    B::AbstractMatrix
)
    # type
    T= eltype(ϕ)

    @inbounds @fastmath for i ∈ axes(ϕ,1)
        ϕ_i= logistic(ϕ[i], offset=one(T), scale=2 * one(T))
        jacob= 2 * logistic(ϕ[i]) * (one(T) - logistic(ϕ[i]))
        ∇f[i]= (ϕ_i * B[i,i] - A[i,i]) * jacob
    end

    return nothing
end

"""
    update_ϕ!(model, state)

Update autoregressive parameters `ϕ`, storing the result in `model`.

#### Arguments
  - `model::DynamicFactorModel` : state space model
  - `state:::EMOptimizerState`  : state variables
"""
function update_ϕ!(model::DynamicFactorModel, state::EMOptimizerState)
    # dims
    T_len= size(model.y,2)
    # type
    T= eltype(model.ϕ)

    # closure of objective function and gradient
    f(x::AbstractVector)= f_ϕ(x, state.V_01, state.V_1) * inv(T_len)
    ∇f!(∇f::AbstractVector, x::AbstractVector)= begin 
                                                    ∇f_ϕ!(∇f, x, state.V_01, state.V_1)
                                                    ldiv!(T_len, ∇f)
                                                end
    
    # Transform parameters
    model.ϕ.diag.= logit.(model.ϕ.diag, offset=one(T), scale=2 * one(T))

    
    # optimize
    options= Optim.Options(g_tol=1e-7, x_reltol=1e-4)
    res= optimize(f, ∇f!, model.ϕ.diag, LBFGS(), options)

    # store result
    model.ϕ.diag.= logistic.(Optim.minimizer(res), offset=one(T), scale=2 * one(T))

    return nothing
end

"""
    update_Λ!(model, state, smoother, pen)

Update loading matrix parameters `Λ`, storing the result in `model`.

#### Arguments
  - `model::DynamicFactorModel` : state space model
  - `state:::EMOptimizerState`  : state variables
  - `smoother::Smoother`        : Kalman smoother output
  - `pen::Penalization`         : penalization parameters
"""
function update_Λ!(
    model::DynamicFactorModel, 
    state::EMOptimizerState, 
    smoother::Smoother, 
    pen::Penalization
)
    # residuals
    resid(model).= model.y .- mean(model)
    # precision matrix
    Ω= prec(model)

    # Gram matrix
    α_1= view(smoother.α,:,1)
    state.V_0.+= view(smoother.V,:,:,1,1) .+ α_1 .* transpose(α_1)

    # linear coefficient b
    b= -inv(prod(size(model.y))) * vec(Ω * resid(model) * transpose(smoother.α))
    # I + A, with A quadratic coefficient
    tmp= inv(prod(size(model.y))) * kron(state.V_0, Ω)
    @inbounds @fastmath for i ∈ axes(tmp,1)
        tmp[i,i]+= one(eltype(tmp))
    end
    # Cholesky decomposition
    C= cholesky!(Hermitian(tmp))

    # Proximal operators
    prox_g!(x::AbstractVector, λ::Real)= prox!(x, λ, pen)
    prox_f!(x::AbstractVector, λ::Real)= shrinkage!(x, λ, C, b)

    # Penalized estimation via admm
    vec(model.Λ).= admm!(vec(model.Λ), prox_f!, prox_g!)

    return nothing
end

function update_Λ!(
    model::DynamicFactorModel, 
    state::EMOptimizerState, 
    smoother::Smoother, 
    pen::NoPen
)
    # residuals
    resid(model).= model.y .- mean(model)

    # Gram matrix
    α_1= view(smoother.α,:,1)
    state.V_0.+= view(smoother.V,:,:,1,1) .+ α_1 .* transpose(α_1)

    # Generalized Ridge regression
    mul!(model.Λ, resid(model), transpose(smoother.α))
    C= cholesky!(Hermitian(state.V_0))
    rdiv!(model.Λ, C)
    
    return nothing
end

"""
    resid_factors!(ε, model, smoother)

Calculate residuals based on factors only.

#### Arguments
  - `model::DynamicFactorModel` : state space model
  - `smoother::Smoother`        : Kalman smoother output

#### Returns
  - `ε::AbstractMatrix` : residuals
"""
function resid_factors!(
    ε::AbstractMatrix, 
    model::DynamicFactorModel, 
    smoother::Smoother
)    
    # Update residuals
    ε.= model.y
    mul!(ε, model.Λ, smoother.α, -1., 1.)

    return nothing
end

function update_model!(
    model::DynamicFactorModel, 
    state::EMOptimizerState, 
    smoother::Smoother, 
    pen::NamedTuple
)
    # Closure of function
    init_resid!(ε::AbstractMatrix)= resid_factors!(ε, model, smoother) 
    
    # Quadratic ridge component
    quad= model.Λ * state.V_sum * transpose(model.Λ)

    # Update hyper parameters
    update_ϕ!(model, state)
    update_error!(model.error, quad, pen.error)
    update_Λ!(model, state, smoother, pen.Λ)
    update_mean!(model.mean, init_resid!, model.error, pen.mean)

    # Update residuals
    ε= resid(model) # retrieve residual container
    ε.= model.y .- mean(model)
    mul!(ε, model.Λ, smoother.α, -1., 1.)

    return nothing
end

# Forecast
function reinstantiate(model::DynamicFactorModel, h::Integer)
    # number of time series
    n= size(model.y,1)

    # expand
    forecast_mean= forecast(model.mean, h)
    y_f= hcat(model.y, fill(NaN, n, h))

    return DynamicFactorModel(y_f, model.r, forecast_mean, model.Λ, model.ϕ, model.error)
end

function forecast(model::DynamicFactorModel, h::Integer)
    # reinstantiate
    forecast_model= reinstantiate(model, h)
    
    # State space system
    sys= create_system(forecast_model, :multivariate)
    fix_system!(sys, forecast_model, :multivariate)
    init_system!(sys, forecast_model)
    get_system!(sys, forecast_model, :multivariate)

    # forecasts
    f= forecast(sys, h)

    return f
end