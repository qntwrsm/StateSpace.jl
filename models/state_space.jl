#=
state_space.jl

    State space model abstract types and general fallback routines

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/08
=#

"""
    StateSpaceModel

Abstract type for state space models.
"""
abstract type StateSpaceModel end

"""
    get_parameters!(ψ, model)

Retrieve hyper parameters from the state space model and store them in `ψ`.

#### Arguments
  - `model::StateSpaceModel`: state space model

#### Returns
  - `ψ::AbstractVector`     : hyper parameters 
"""
get_parameters!(ψ::AbstractVector, model::StateSpaceModel)= nothing

"""
    store_parameters!(model, ψ)

Store hyper parameters `ψ` from the state space model in `model`.

#### Arguments
  - `ψ::AbstractVector`     : hyper parameters 

#### Returns
  - `model::StateSpaceModel`: state space model 
"""
store_parameters!(model::StateSpaceModel, ψ::AbstractVector)= nothing

"""
    get_system!(sys, model)

Retrieve system matrices from the state space model and store them in `sys`.

#### Arguments
  - `model::StateSpaceModel`: state space model

#### Returns
  - `sys::StateSpaceSystem` : state space system matrices
"""
get_system!(sys::StateSpaceSystem, model::StateSpaceModel)= nothing

"""
    store_system!(model, sys)

Store system matrices from the state space model in `model`.

#### Arguments
  - `sys::StateSpaceSystem` : state space system matrices

#### Returns
  - `model::StateSpaceModel`: state space model
"""
store_system!(model::StateSpaceModel, sys::StateSpaceSystem)= nothing 

"""
    init!(model, sys)

Initialize the state space model hyper parameters as defined by `model` and the
initial conditions of the state space model in `sys`.

#### Returns
  - `model::StateSpaceModel`: state space model
  - `sys::StateSpaceSystem` : state space system matrices
"""
init!(model::StateSpaceModel, sys::StateSpaceSystem)= nothing

"""
    init_model!(model)

Initialize the state space model hyper parameters.

#### Returns
  - `model::StateSpaceModel`: state space model
"""
init_model!(model::StateSpaceModel)= nothing

"""
    init_system!(sys, model)

Initialize the state space model hyper parameters.

#### Arguments
  - `model::StateSpaceModel`: state space model

#### Returns
  - `sys::StateSpaceSystem` : state space system matrices
"""
init_system!(sys::StateSpaceSystem, model::StateSpaceModel)= nothing