#=
state_space.jl

    State space model abstract types and general fallback routines

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/08
=#

"""
    nparams(model)

Determine the number of hyper parameters.

#### Arguments
  - `model::StateSpaceModel`: state space model

#### Returns
  - `n_params::Integer` : number of hyper parameters
"""
function nparams end

"""
	get_params!(ψ, model)

Retrieve hyper parameters from the state space model and store them in `ψ`.

#### Arguments
  - `model::StateSpaceModel`: state space model

#### Returns
  - `ψ::AbstractVector`     : hyper parameters 
"""
function get_params! end

"""
    store_params!(model, ψ)

Store hyper parameters `ψ` from the state space model in `model`.

#### Arguments
  - `ψ::AbstractVector`     : hyper parameters

#### Returns
  - `model::StateSpaceModel`: state space model 
"""
function store_params! end

"""
    get_system!(sys, model, method)

Retrieve system matrices from the state space model and store them in `sys`.

#### Arguments
  - `model::StateSpaceModel`: state space model
  - `method::Symbol`		: filtering method        

#### Returns
  - `sys::StateSpaceSystem` : state space system matrices
"""
function get_system! end

"""
    init!(model, init, method)

Initialize the state space model hyper parameters as defined by `model` and the
initial conditions of the state space model in `sys`.

#### Arguments
  - `init::NamedTuple`	: initial hyper parameters
  - `method::Symbol`	: filtering method

#### Returns
  - `model::StateSpaceModel`: state space model
  - `sys::StateSpaceSystem` : state space system matrices
"""
function init! end

"""
    init_model!(model, init)

Initialize the state space model hyper parameters.

#### Arguments
  - `init::NamedTuple`	: initial hyper parameters

#### Returns
  - `model::StateSpaceModel`: state space model
"""
function init_model! end

"""
    fix_system!(sys, model, method)

Fix state space system components.

#### Arguments
  - `model::StateSpaceModel`: state space model
  - `method::Symbol`		: filtering method

#### Returns
  - `sys::StateSpaceSystem` : state space system matrices
"""
function fix_system! end

"""
    init_system!(sys, model)

Initialize state space system.

#### Arguments
  - `model::StateSpaceModel`: state space model

#### Returns
  - `sys::StateSpaceSystem` : state space system matrices
"""
function init_system! end

"""
	loglik(filter, sys, model, method)

Compute the log-likelihood for a linear Gaussian State Space model, given by
`model`, with Kalman filter output `filter`, based on filtering method
(`method`).

#### Arguments
  - `filter::KalmanFilter`	: Kalman filter output
  - `sys::StateSpaceSystem` : state space system matrices
  - `model::StateSpaceModel`: state space model
  - `method::Symbol`		: filtering method

#### Returns
  - `ll::Real`	: log-likelihood 
"""
function loglik end