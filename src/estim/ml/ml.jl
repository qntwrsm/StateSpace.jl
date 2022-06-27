#=
ml.jl

    Maximum Likelihood (ML) algorithm to estimate the hyper parameters of 
    a linear Gaussian State Space model.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/03/14
=#

"""
    objective(ψ, model, filter, sys, method)

Compute objective function value, i.e. average negative log-likelihood, `ll` of
the State Space model `model` with system matrices `sys` at parameter vector
`ψ`.

#### Arguments
  - `ψ::AbstractVector`     : parameters
  - `model::StateSpaceModel`: state space model
  - `filter::KalmanFilter`  : Kalman filter output
  - `sys::StateSpaceSystem` : state space system matrices
  - `method::Symbol`        : filtering method

#### Returns
  - `ll::Real`          : average negative log-likelihood value 
"""
function objective( ψ::AbstractVector, 
                    model::StateSpaceModel, 
                    filter::KalmanFilter, 
                    sys::StateSpaceSystem,
                    method::Symbol
                    )
    # Get dims
    (n,T)= size(model.y) 

    # Store parameters
    store_params!(model, ψ)

    # update mean, covariance, and precision
    mean!(model)
    cov!(model)
    prec!(model)

    # Get system
    init_system!(sys, model)
    get_system!(sys, model, method)

    # Run Kalman filter
    kalman_filter!(filter, sys)

    # Average negative log-likelihood
    ll= -inv(n * T) * loglik(filter, model, method)

    return ll
end

"""
    _maximum_likelihood!(model, init, method, pen, ϵ_abs, ϵ_rel, max_iter)

Internal dispatch function for maximum likelihood (ML) estimation of hyper
parameters of a linear Gaussian State Space model.

#### Arguments
  - `model::StateSpaceModel`: state space model
  - `init::NamedTuple`      : initial model parameters
  - `method::Symbol`        : filtering method
  - `pen::Penalization`     : penalization type
  - `ϵ_abs::Real`           : absolute tolerance
  - `ϵ_rel::Real`           : relative tolerance
  - `max_iter::Integer`     : max number of iterations

#### Returns
  - `ll::Real`          : log-likelihood value 
"""
function _maximum_likelihood!(  model::StateSpaceModel,
                                init::NamedTuple, 
                                method::Symbol,
                                pen::Penalization,
                                ϵ_abs::Real, 
                                ϵ_rel::Real, 
                                max_iter::Integer
                                )            
    # Initialize state space model and system
    sys= init!(model, init, method)

    # Get dims
    (n,T_len)= size(model.y)
    p= length(sys.a1)
    n_params= nparams(model)

    # Initialize parameters
    ψ= similar(model.y, n_params)
    get_params!(ψ, model)
    
    # Initialize filter
    T= eltype(model.y)  # type
    if method === :univariate 
        filter= UnivariateFilter(n, p, T_len, T)
    elseif method === :collapsed
        filter= UnivariateFilter(p, p, T_len, T)
    elseif method === :multivariate
        filter= MultivariateFilter(n, p, T_len, T)
    elseif method === :woodbury
        filter= WoodburyFilter(n, p, T_len, T)
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end
    
    # Closure of objective function and gradient
    f(x::AbstractVector)= objective(x, model, filter, sys, method)
    cache= FiniteDiff.GradientCache(similar(ψ), similar(ψ))
    ∇f!(∇f::AbstractVector, x::AbstractVector)= FiniteDiff.finite_difference_gradient!(∇f, f, x, cache)

    # Warm start for proximal operator
    y_prev= copy(ψ)

    # Proximal operators
    prox_g!(x::AbstractVector, λ::Real)= prox!(x, λ, pen)
    prox_f!(x::AbstractVector, λ::Real)= smooth!(x, λ, f, ∇f!, y_prev)

    # optimize
    ψ.= admm!(ψ, prox_f!, prox_g!, ϵ_abs=ϵ_abs, ϵ_rel=ϵ_rel, max_iter=max_iter)

    # Store results
    store_params!(model, ψ)
   
    return -(n * T_len) * f(ψ)
end

function _maximum_likelihood!(  model::StateSpaceModel,
                                init::NamedTuple, 
                                method::Symbol,
                                pen::NoPen,
                                ϵ_abs::Real, 
                                ϵ_rel::Real, 
                                max_iter::Integer
                                )            
    # Initialize state space model and system
    sys= init!(model, init, method)

    # Get dims
    (n,T_len)= size(model.y)
    p= length(sys.a1)
    n_params= nparams(model)

    # Initialize parameters
    ψ0= similar(model.y, n_params)
    get_params!(ψ0, model)
    
    # Initialize filter
    T= eltype(model.y)  # type
    if method === :univariate 
        filter= UnivariateFilter(n, p, T_len, T)
    elseif method === :collapsed
        filter= UnivariateFilter(p, p, T_len, T)
    elseif method === :multivariate
        filter= MultivariateFilter(n, p, T_len, T)
    elseif method === :woodbury
        filter= WoodburyFilter(n, p, T_len, T)
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end
    
    # Closure of objective function
    f(x::AbstractVector)= objective(x, model, filter, sys, method)
    cache= FiniteDiff.GradientCache(similar(ψ0), similar(ψ0))
    ∇f!(∇f::AbstractVector, x::AbstractVector)= FiniteDiff.finite_difference_gradient!(∇f, f, x, cache)

    # optimize
    options= Optim.Options(g_tol=ϵ_abs, x_reltol=ϵ_rel, iterations=max_iter)
    res= optimize(f, ∇f!, ψ0, LBFGS(), options)

    # Store results
    store_params!(model, Optim.minimizer(res))
   
    return -(n * T_len) * Optim.minimum(res)
end

"""
    maximum_likelihood!(model; init=NamedTuple(), method=:collapsed, pen=NoPen(), ϵ_abs=1e-7, ϵ_rel=1e-4, max_iter=1000)

Maximum Likelihood (ML) algorithm to estimate the hyper parameters of a linear
Gaussian State Space model as defined by `model`, results are stored in `model`.
Penalized ML estimation is allowed through `pen`. If ```pen` = NoPen()`` the
optimization routine is L-BFGS, when ```pen` ≂̸ NoPen()`` optimization is doen
through ADMM.

#### Arguments
  - `model::StateSpaceModel`: state space model
  - `init::NamedTuple`      : initial model parameters
  - `method::Symbol`        : filtering method
  - `pen::Penalization`     : penalization type
  - `ϵ_abs::Real`           : absolute tolerance
  - `ϵ_rel::Real`           : relative tolerance
  - `max_iter::Integer`     : max number of iterations

#### Returns
  - `ll::Real`          : log-likelihood value      
"""
function maximum_likelihood!(   model::StateSpaceModel; 
                                init::NamedTuple=NamedTuple(), 
                                method::Symbol=:collapsed,
                                pen::Penalization=NoPen(),
                                ϵ_abs::Real=1e-7, 
                                ϵ_rel::Real=1e-4, 
                                max_iter::Integer=1_000
                                )
    return _maximum_likelihood!(model, init, method, pen, ϵ_abs, ϵ_rel, max_iter)
end