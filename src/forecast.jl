#=
forecast.jl

    Forecasting routines for linear Gaussian state space models

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2021/12/06
=#
# Struct
struct Forecast{Ty, TF, Ta, TP}
    y_hat::Ty   # forecasts
    F_hat::TF   # forecast error variance
    a_hat::Ta   # forecasted states
    P_hat::TP   # forecasted state variances
end
# Constructor
function Forecast(n::Integer, p::Integer, h::Integer, T::Type)
    # forecast components
    y_hat= Matrix{T}(undef, n, h)
    F_hat= Array{T,3}(undef, n, n, h)
    a_hat= Matrix{T}(undef, p, h)
    P_hat= Array{T,3}(undef, p, p, h)

    return Forecast(y_hat, F_hat, a_hat, P_hat)
end

"""
	forecast(sys, h)
	
Compute ``h``-step ahead forecasts for observations and states with
corresponding forecast error variances and forecasted state variances for a
state space model with system matrices `sys`.

#### Arguments
  - `sys::StateSpaceSystem`	: state space system matrices
  - `h::Integer`			: forecast horizon

#### Returns
  - `f::Forecast`   : forecasts and forecast error variances
"""
function forecast(sys::LinearTimeInvariant, h::Integer)
	# get dims
	(n,T_len)= size(sys.y)
    p= length(sys.a1)

    # Type
    T= eltype(sys.y)

    # Kalman filter
    filter= MultivariateFilter(n, p, T_len, T)
    kalman_filter!(filter, sys)

    # forecasts
    f= Forecast(n, p, h, T)
    @inbounds @fastmath for i = 1:h
        # Store filter forecast output
        f.F_hat[:,:,i]= view(filter.F,:,:,T_len-h+i)
        f.a_hat[:,i]= view(filter.a,:,T_len-h+i)
        f.P_hat[:,:,i]= view(filter.P,:,:,T_len-h+i)

        # forecast
        y_hat= view(f.y_hat, :, i)
        a_hat= view(f.a_hat, :, i)
        y_hat.= sys.d
        mul!(y_hat, sys.Z, a_hat, 1., 1.)
    end
	
	return f
end

function forecast(sys::LinearTimeVariant, h::Integer)
	# get dims
	(n,T_len)= size(sys.y)
    p= length(sys.a1)

    # Type
    T= eltype(sys.y)

    # Kalman filter
    filter= MultivariateFilter(n, p, T_len, T)
    kalman_filter!(filter, sys)

    # forecasts
    f= Forecast(n, p, h, T)
    @inbounds @fastmath for i = 1:h
        # Store filter forecast output
        f.F_hat[:,:,i]= view(filter.F,:,:,T_len-h+i)
        f.a_hat[:,i]= view(filter.a,:,T_len-h+i)
        f.P_hat[:,:,i]= view(filter.P,:,:,T_len-h+i)

        # forecast
        y_hat= view(f.y_hat, :, i)
        a_hat= view(f.a_hat, :, i)
        y_hat.= view(sys.d, :, T_len-h+i)
        mul!(y_hat, sys.Z[T_len-h+i], a_hat, 1., 1.)
    end
	
	return f
end