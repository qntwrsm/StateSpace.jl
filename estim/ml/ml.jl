#=
ml.jl

    Maximum Likelihood (ML) algorithm to estimate the hyper parameters of 
    a linear Gaussian State Space model.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/03/14
=#

function f(ψ::AbstractVector, model::StateSpaceModel, filter::KalmanFilter, sys::StateSpaceSystem)
    # Get dims
    (n,T)= size(model.y) 

    # Store parameters
    store_params!(model, ψ)

    # Get system
    init_system!(sys, model)
    get_system!(sys, model)

    # Run Kalman filter
    kalman_filter!(filter, model.y, sys)

    # Average negative log-likelihood
    ll= -inv(n * T) * loglik(filter)

    return ll
end

"""
    maximum_likelihood!(model; init=NamedTuple(), ϵ_abs=1e-7, ϵ_rel=1e-3, max_iter=1000)

Maximum Likelihood (ML) algorithm to estimate the hyper parameters of
a linear Gaussian State Space model as defined by `model`, results are stored in
`model`.

#### Arguments
  - `model::StateSpaceModel`: state space model
  - `init::NamedTuple`      : initial model parameters
  - `ϵ_abs::Real`           : absolute tolerance
  - `ϵ_rel::Real`           : relative tolerance
  - `max_iter::Integer`     : max number of iterations

#### Returns
  - `ll::Real`          : log-likelihood value      
"""
function maximum_likelihood!(model::StateSpaceModel; 
            init::NamedTuple=NamedTuple(), ϵ_abs::Real=1e-7, ϵ_rel::Real=1e-3, max_iter::Integer=1_000)            
    # Initialize state space model and system
    sys= init!(model, init)

    # Get dims
    (n,T)= size(model.y)
    p= length(sys.a1)
    n_params= nparams(model)

    # Initialize parameters
    ψ0= similar(model.y, n_params)
    get_parameters!(ψ0, model)
    
    # Initialize filter and smoother
    filter= MultivariateFilter(similar(model.y, p, T), similar(model.y, p, p, T), 
                            similar(model.y, n, T), similar(model.y, n, n, T), 
                            similar(model.y, p, n, T))
    
    # Closure of objective function
    f_cl(x::AbstractVector)= f(x, model, filter, sys)

    res= optimize(f_cl, ψ0, LBFGS())
   
    return minimum(res)
end