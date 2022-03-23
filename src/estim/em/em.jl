#=
em.jl

    Expectation-Maximization (EM) algorithm to estimate the hyper parameters of 
    a linear Gaussian State Space model.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/09
=#

# Expectation-Maximization
struct EMState{Tv, Tm} <: EMOptimizerState
	ψ::Tv		    # current state
	ψ_prev::Tv	    # previous state
    ψ_prev_m::Tv    # M step update buffer
	Δ::Tv		    # change in state
    V_sum::Tm       # buffer for E step sum Vₜ
    V_0::Tm         # buffer for E step V₀   
    V_1::Tm         # buffer for E step V₋₁
    V_01::Tm        # buffer for E step V₀₋₁
end

# Expectation-Conditional Maximization
struct ECMState{Tv, Tm} <: EMOptimizerState
	ψ::Tv		    # current state
	ψ_prev::Tv	    # previous state
	Δ::Tv		    # change in state
    V_sum::Tm       # buffer for E step sum Vₜ
    V_0::Tm         # buffer for E step V₀   
    V_1::Tm         # buffer for E step V₋₁
    V_01::Tm        # buffer for E step V₀₋₁
end

"""
    em!(model, pen; init=NamedTuple(), ϵ_abs=1e-7, ϵ_rel=1e-3, max_iter=1000)

Expectation-Maximization (EM) algorithm to estimate the hyper parameters of a
linear Gaussian State Space model as defined by `model`, storing the results in
`model`.

#### Arguments
  - `model::StateSpaceModel`: state space model
  - `pen::NamedTuple`       : penalization parameters
  - `method::Symbol`        : filtering method
  - `init::NamedTuple`      : initial model parameters
  - `ϵ_abs::Real`           : absolute tolerance
  - `ϵ_rel::Real`           : relative tolerance
  - `max_iter::Integer`     : max number of iterations

#### Returns
  - `ll::Real`  : log-likelihood value      
"""
function em!(model::StateSpaceModel, pen::NamedTuple; 
            method::Symbol=:univariate, init::NamedTuple=NamedTuple(), 
            ϵ_abs::Real=1e-7, ϵ_rel::Real=1e-3, max_iter::Integer=1000)
    # Initialize state space model and system
    sys= init!(model, init, method)

    # Get dims
    (n,T)= size(model.y)
    p= length(sys.a1)
    n_params= nparams(model)

    # Initialize parameters
    ψ= similar(model.y, n_params)
    get_params!(ψ, model)

    # Initialize state
    state= EMState(ψ, similar(ψ), similar(ψ), similar(ψ), similar(ψ, p, p), 
                    similar(ψ, p, p), similar(ψ, p, p), similar(ψ, p, p))
    
    # Initialize state
    state= ECMState(ψ, similar(ψ), similar(ψ), similar(ψ, p, p), similar(ψ, p, p), 
                    similar(ψ, p, p), similar(ψ, p, p))

    # Initialize filter and smoother
    if method === :univariate || method === :collapsed
        filter= UnivariateFilter(similar(model.y, p, n+1, T), similar(model.y, p, p, n+1, T), 
                                similar(model.y, n, T), similar(model.y, n, T),
                                similar(model.y, p, n, T))
    elseif method === :multivariate
        filter= MultivariateFilter(similar(model.y, p, T), similar(model.y, p, p, T), 
                                similar(model.y, n, T), similar(model.y, n, n, T),
                                similar(model.y, p, n, T))
    elseif method === :woodbury
        filter= WoodburyFilter(similar(model.y, p, T), similar(model.y, p, p, T), 
                                similar(model.y, n, T), similar(model.y, n, n, T),
                                similar(model.y, p, n, T))
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end
    smoother= Smoother(similar(model.y, p, T), similar(model.y, p, p, 2, T), 
                                similar(model.y, p, p), similar(model.y, p), 
                                similar(model.y, p, p))

    # Initialize stopping flags
    abs_change= Inf
    rel_change= Inf
    # Initialize iteration counter
    iter= 1
    # EM algorithm
    while (abs_change > ϵ_abs && rel_change > ϵ_rel) && iter < max_iter
        # Store current parameters
        copyto!(state.ψ_prev, state.ψ)

        # E-step
        init_system!(sys, model)
        get_system!(sys, model, method)
        estep!(smoother, filter, sys)

        # M-step
        mstep!(state, model, smoother, pen)

        # Store change in state
        state.Δ.= state.ψ .- state.ψ_prev

        # Absolute change
        abs_change= maximum(abs, state.Δ)
        # Relative change
        rel_change= abs_change * inv(1 + maximum(abs, state.ψ))

        # Update iteration counter
        iter+=1
    end
   
    return loglik(filter, sys, model, method)
end

"""
    ecm!(model, pen; init=NamedTuple(), ϵ_abs=1e-7, ϵ_rel=1e-3, max_iter=1000)

Expectation-Conditional Maximization (ECM) algorithm to estimate the hyper
parameters of a linear Gaussian State Space model as defined by `model`, results
are stored in `model`.

#### Arguments
  - `model::StateSpaceModel`: state space model
  - `pen::NamedTuple`       : penalization parameters
  - `method::Symbol`        : filtering method
  - `init::NamedTuple`      : initial model parameters
  - `ϵ_abs::Real`           : absolute tolerance
  - `ϵ_rel::Real`           : relative tolerance
  - `max_iter::Integer`     : max number of iterations

#### Returns
  - `ll::Real`  : log-likelihood value      
"""
function ecm!(model::StateSpaceModel, pen::NamedTuple; 
                method::Symbol=:univariate, init::NamedTuple=NamedTuple(), 
                ϵ_abs::Real=1e-7, ϵ_rel::Real=1e-3, max_iter::Integer=1000)            
    # Initialize state space model and system
    sys= init!(model, init, method)

    # Get dims
    (n,T)= size(model.y)
    p= length(sys.a1)
    n_params= nparams(model)

    # Initialize parameters
    ψ= similar(model.y, n_params)
    get_params!(ψ, model)

    # Initialize state
    state= ECMState(ψ, similar(ψ), similar(ψ), similar(ψ, p, p), similar(ψ, p, p), 
                    similar(ψ, p, p), similar(ψ, p, p))

    # Initialize filter and smoother
    if method === :univariate 
        filter= UnivariateFilter(similar(model.y, p, n+1, T), similar(model.y, p, p, n+1, T), 
                                similar(model.y, n, T), similar(model.y, n, T),
                                similar(model.y, p, n, T))
    elseif method === :collapsed
        filter= UnivariateFilter(similar(model.y, p, p+1, T), similar(model.y, p, p, p+1, T), 
                                similar(model.y, p, T), similar(model.y, p, T),
                                similar(model.y, p, p, T))
    elseif method === :multivariate
        filter= MultivariateFilter(similar(model.y, p, T), similar(model.y, p, p, T), 
                                similar(model.y, n, T), similar(model.y, n, n, T),
                                similar(model.y, p, n, T))
    elseif method === :woodbury
        filter= WoodburyFilter(similar(model.y, p, T), similar(model.y, p, p, T), 
                                similar(model.y, n, T), similar(model.y, n, n, T),
                                similar(model.y, p, n, T))
    else
        throw(ArgumentError("Invalid method name $(method)"))
    end
    smoother= Smoother(similar(model.y, p, T), similar(model.y, p, p, 2, T), 
                                similar(model.y, p, p), similar(model.y, p), 
                                similar(model.y, p, p))

    # Initialize stopping flags
    abs_change= Inf
    rel_change= Inf
    # Initialize iteration counter
    iter= 1
    # EM algorithm
    while (abs_change > ϵ_abs && rel_change > ϵ_rel) && iter < max_iter
        # Store current parameters
        copyto!(state.ψ_prev, state.ψ)

        # E-step
        init_system!(sys, model)
        get_system!(sys, model, method)
        estep!(smoother, filter, sys)

        # Conditional M-step
        mstep!(state, model, smoother, pen)

        # Store change in state
        state.Δ.= state.ψ .- state.ψ_prev

        # Absolute change
        abs_change= maximum(abs, state.Δ)
        # Relative change
        rel_change= abs_change * inv(1 + maximum(abs, state.ψ))

        # Update iteration counter
        iter+=1
    end
   
    return loglik(filter, sys, model, method)
end