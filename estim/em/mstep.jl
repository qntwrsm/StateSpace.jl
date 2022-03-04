#=
mstep.jl

    Maximization step of the Expectation-Maximization (EM) algorithm to estimate 
    the hyper parameters of a linear Gaussian State Space model.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/09
=#

"""
    update_model!(model, state, smoother, fixed)

Conditional update of model components of the linear Gaussian state space model.

#### Arguments
  - `state::EMOptimizerState` 	: state variables
  - `smoother::Smoother`		: Kalman smoother output
  - `fixed::NamedTuple` 		: fixed hyper parameters

#### Returns
  - `model::StateSpaceModel`: state space model
"""
function update_model! end

"""
    mstep!(state, model, smoother, fixed,; ϵ_abs=1e-7, ϵ_rel=1e-3, max_iter=1000)

Update state and model using the maximization step of the EM algorithm.

#### Arguments
  - `state::EMState`        : state variables
  - `model::StateSpaceModel`: state space model
  - `smoother::Smoother`    : Kalman smoother output
  - `fixed::NamedTuple`     : fixed hyper parameters
  - `ϵ_abs::Real`           : absolute tolerance
  - `ϵ_rel::Real`           : relative tolerance
  - `max_iter::Integer`     : max number of iterations
"""
function mstep!(state::EMState, model::StateSpaceModel, smoother::Smoother,
                fixed::NamedTuple; 
                ϵ_abs::Real=1e-7, ϵ_rel::Real=1e-3, max_iter::Integer=1000)
    # Get dims
    T_len= size(model.y,1)

    # lag and lead smoothed states
    α_lag= view(smoother.α,:,1:T_len-1)
    α_lead= view(smoother.α,:,2:T_len)

    # Update buffer variables
    # sum
    sum!(state.V_sum, view(smoother.V, :, :, 1, :))
    state.V_0.= state.V_sum .- view(smoother.V,:,:,1,1)
    # variance
    @views mul!(state.V_0, α_lead, transpose(α_lead), 1., 1.)
    state.V_1.= state.V_sum .- view(smoother.V,:,:,1,T_len)
    # covariance
    @views mul!(state.V_1, α_lag, transpose(α_lag), 1., 1.)
    sum!(state.V_01, view(smoother.V, :, :, 2, 2:T_len))
    @views mul!(state.V_01, α_lead, transpose(α_lag), 1., 1.)
    
    # Initialize stopping flags
    abs_change= zero(eltype(state.ψ))
    rel_change= zero(eltype(state.ψ))
    # Initialize iteration counter
    iter= 1
    # M-step (maximization)
    while (abs_change < ϵ_abs || rel_change < ϵ_rel) && iter < max_iter
        # Store current parameters
        copyto!(state.ψ_prev_m, state.ψ)        

        # Update model
        update_model!(model, state, smoother, fixed)

        # Update state
        get_parameters!(state.ψ, model, fixed)

        # Store change in state
        @. state.Δ= state.ψ - state.ψ_prev_m

        # Absolute change
        abs_change= maximum(abs, state.Δ)
        # Relative change
        rel_change= abs_change * inv(1 + maximum(abs, state.ψ))

        # Update iteration counter
        iter+=1
    end

    return nothing
end

"""
    mstep!(state, model, smoother, fixed)

Update state and model using the conditional maximization step of the ECM
algorithm.

#### Arguments
  - `state::ECMState`       : state variables
  - `model::StateSpaceModel`: state space model
  - `smoother::Smoother`    : Kalman smoother output
  - `fixed::NamedTuple`     : fixed hyper parameters
"""
function mstep!(state::ECMState, model::StateSpaceModel, smoother::Smoother,
                fixed::NamedTuple)
    # Get dims
    T_len= size(model.y,1)

    # lag and lead smoothed states
    α_lag= view(smoother.α,:,1:T_len-1)
    α_lead= view(smoother.α,:,2:T_len)

    # Update buffer variables
    # sum
    sum!(state.V_sum, view(smoother.V, :, :, 1, :))
    # variance
    state.V_0.= state.V_sum .- view(smoother.V,:,:,1,1)
    @views mul!(state.V_0, α_lead, transpose(α_lead), 1., 1.)
    state.V_1.= state.V_sum .- view(smoother.V,:,:,1,T_len)
    # covariance
    @views mul!(state.V_1, α_lag, transpose(α_lag), 1., 1.)
    sum!(state.V_01, view(smoother.V, :, :, 2, 2:T_len))
    @views mul!(state.V_01, α_lead, transpose(α_lag), 1., 1.)
     
    # Update model
    update_model!(model, state, smoother, fixed)

    # Update state
    get_parameters!(state.ψ, model, fixed)

    return nothing
end