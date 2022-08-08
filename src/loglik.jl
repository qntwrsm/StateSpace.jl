#=
loglik.jl

    Log-likelihood calculation routines for a linear Gaussian State Space model

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2020/07/29
=#

"""
	loglik(filter)
	
Compute the log-likelihood for a linear Gaussian State Space model with Kalman
filter output `filter`.

#### Arguments
  - `filter::KalmanFilter`	: Kalman filter output

#### Returns
  - `ll::Real`	: log-likelihood
"""
function loglik(filter::MultivariateFilter)
    (n,T_len)= size(filter.v)
	
	# Initialize temp. containers
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_n= Vector{Float64}(undef, n)

    # Initialize log-likelihood
    ll= -log(2*π) * T_len * n

    # Log-likelihood
    @inbounds @fastmath for t = 1:T_len
		# Store views
        v_t= view(filter.v,:,t)
		F_t= view(filter.F,:,:,t)
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		C= cholesky!(Hermitian(tmp_nn))
		
		# log|Fₜ|
       	ll-= logdet(C)
		
		# Fₜ⁻¹ｘvₜ
        ldiv!(tmp_n, C, v_t)
		# vₜ'ｘFₜ⁻¹ｘvₜ
		ll-= dot(v_t, tmp_n)
    end

    return .5*ll
end

function loglik(filter::WoodburyFilter)
    (n,T_len)= size(filter.v)
	
	# Initialize temp. container
	tmp= Matrix{Float64}(undef, (n,n))

    # Initialize log-likelihood
    ll= -log(2*π) * T_len * n

    # Log-likelihood
    @inbounds @fastmath for t = 1:T_len
		# Store views
        v_t= view(filter.v,:,t)
		Fi_t= view(filter.Fi,:,:,t)
		
		# vₜ'ｘFₜ⁻¹ｘvₜ
		ll-= dot(v_t, Fi_t, v_t)
		
		# log|Fₜ⁻¹|
		copyto!(tmp, Fi_t)
       	ll+= logdet(cholesky!(Hermitian(tmp)))
    end

    return .5*ll
end

function loglik(filter::UnivariateFilter)
    (n,T_len)= size(filter.v)

    # Initialize log-likelihood
    ll= zero(Float64)

    # Log-likelihood
    @inbounds @fastmath for t = 1:T_len
		for i in 1:n
			F_it= filter.F[i,t]
			if F_it ≠ zero(F_it)
				ll-= log(2*π) + log(F_it) + filter.v[i,t]^2 * inv(F_it)
			end
		end
    end
		
	return .5*ll
end