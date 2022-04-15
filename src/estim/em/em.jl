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
    em!(model, pen; init=NamedTuple(), ϵ=1e-4, max_iter=1000)

Expectation-Maximization (EM) algorithm to estimate the hyper parameters of a
linear Gaussian State Space model as defined by `model`, storing the results in
`model`.

#### Arguments
  - `model::StateSpaceModel`: state space model
  - `pen::NamedTuple`       : penalization parameters
  - `method::Symbol`        : filtering method
  - `init::NamedTuple`      : initial model parameters
  - `ϵ::Real`               : tolerance
  - `max_iter::Integer`     : max number of iterations

#### Returns
  - `ll::Real`  : log-likelihood value      
"""
function em!(model::StateSpaceModel, pen::NamedTuple; 
            method::Symbol=:univariate, init::NamedTuple=NamedTuple(), 
            ϵ::Real=1e-4, max_iter::Integer=1000)
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

    # Initialize relative change
    rel_change= Inf
    # Initialize iteration counter
    iter= 1
    # EM algorithm
    while rel_change > ϵ && iter < max_iter
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

        # Relative change
        rel_change= norm(state.Δ, Inf) * inv(1 + norm(state.ψ, Inf))

        # Update iteration counter
        iter+=1
    end
   
    return loglik(filter, sys, model, method)
end

"""
    ecm!(model, pen; init=NamedTuple(), ϵ=1e-4, max_iter=1000)

Expectation-Conditional Maximization (ECM) algorithm to estimate the hyper
parameters of a linear Gaussian State Space model as defined by `model`, results
are stored in `model`.

#### Arguments
  - `model::StateSpaceModel`: state space model
  - `pen::NamedTuple`       : penalization parameters
  - `method::Symbol`        : filtering method
  - `init::NamedTuple`      : initial model parameters
  - `ϵ::Real`               : tolerance
  - `max_iter::Integer`     : max number of iterations

#### Returns
  - `ll::Real`  : log-likelihood value      
"""
function ecm!(model::StateSpaceModel, pen::NamedTuple; 
                method::Symbol=:univariate, init::NamedTuple=NamedTuple(), 
                ϵ::Real=1e-4, max_iter::Integer=1000)            
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

    # Initialize relative change
    rel_change= Inf
    # Initialize iteration counter
    iter= 1
    # EM algorithm
    while rel_change > ϵ && iter < max_iter
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

        # Relative change
        rel_change= norm(state.Δ, Inf) * inv(1 + norm(state.ψ, Inf))

        # Update iteration counter
        iter+=1
    end
   
    return loglik(filter, sys, model, method)
end