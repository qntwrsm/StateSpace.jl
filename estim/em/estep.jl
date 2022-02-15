#=
estep.jl

    Expectation step of the Expectation-Maximization (EM) algorithm to estimate 
    the hyper parameters of a linear Gaussian State Space model.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/09
=#

"""
    estep!(smoother, filter, y, sys)

Perform expectation step of the joint log-likelihood of the linear Gaussian
State Space model as part of the EM algorithm.

#### Arguments
  - `y::AbstractMatrix`		: data (n x T)
  - `sys::StateSpaceSystem` : state space system matrices

#### Returns
  - `filter::KalmanFilter`  : Kalman filter output
  - `smoother::Smoother`    : Kalman smoother output
"""
function estep!(smoother::Smoother, filter::KalmanFilter, y::AbstractMatrix, 
				sys::StateSpaceSystem)
	# Run filter
    kalman_filter!(filter, y, sys)
	# Run smoother
    kalman_smoother_cov!(smoother, filter, sys)

    return nothing
end