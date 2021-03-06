#=
estep.jl

    Expectation step of the Expectation-Maximization (EM) algorithm to estimate 
    the hyper parameters of a linear Gaussian State Space model.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/09
=#

"""
    estep!(smoother, filter, sys)

Perform expectation step of the joint log-likelihood of the linear Gaussian
State Space model as part of the EM algorithm.

#### Arguments
  - `sys::StateSpaceSystem` : state space system matrices

#### Returns
  - `filter::KalmanFilter`  : Kalman filter output
  - `smoother::Smoother`    : Kalman smoother output
"""
function estep!(smoother::Smoother, filter::KalmanFilter, sys::StateSpaceSystem)
	# Run filter
    kalman_filter!(filter, sys)
	# Run smoother
    kalman_smoother_cov!(smoother, filter, sys)

    return nothing
end