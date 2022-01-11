#=
loglik.jl

    Log-likelihood calculation routines for a linear Gaussian State Space model

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2020/07/29
=#

"""
	loglik(f)
	
Compute the log-likelihood for a linear Gaussian State Space model with Kalman
filter output `f`.

#### Arguments
  - `f::Filter`	: Kalman filter output

#### Returns
  - `ll::Real`	: log-likelihood
"""
function loglik(f::Filter)
    (n,T_len)= size(f.v)
	
	# Initialize temp. containers
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_n= view(tmp_nn,:,1)

    # Initialize log-likelihood
    ll= -log(2.0*pi)*T_len*n

    # Log-likelihood
    @inbounds @fastmath for t in 1:T_len
		# Store views
        v_t= view(f.v,:,t)
		F_t= view(f.F,:,:,t)
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		fac= cholesky!(tmp_nn)
		
		# Fₜ⁻¹ｘvₜ
        ldiv!(tmp_n, fac, v_t)
		# vₜ'ｘFₜ⁻¹ｘvₜ
		ll-= dot(v_t, tmp_n)
		# log|Fₜ|
       	ll-= logdet(fac)
    end

    return .5*ll
end

"""
	loglik(f)
	
Compute the log-likelihood for a linear Gaussian State Space model with Kalman
filter output `f` based on Woodbury's Identity.

#### Arguments
  - `f::FilterWb`	: Kalman filter output

#### Returns
  - `ll::Real`		: log-likelihood
"""
function loglik(f::FilterWb)
    (n,T_len)= size(f.v)
	
	# Initialize temp. container
	tmp= Matrix{Float64}(undef, (n,n))

    # Initialize log-likelihood
    ll= -log(2.0*pi)*T_len*n

    # Log-likelihood
    @inbounds @fastmath for t in 1:T_len
		# Store views
        v_t= view(f.v,:,t)
		Fi_t= view(f.Fi,:,:,t)
		
		# vₜ'ｘFₜ⁻¹ｘvₜ
		ll-= dot(v_t, Fi_t, v_t)
		
		# Cholesky factorization of Fₜ⁻¹
		copyto!(tmp, Fi_t)
		fac= cholesky!(tmp)
		
		# log|Fₜ⁻¹|
       	ll+= logdet(fac)
    end

    return .5*ll
end

"""
	loglik_eq(f)
	
Compute the log-likelihood for a linear Gaussian State Space model with Kalman
filter output `f` based on the equation-by-equation or univariate version of
the filter.

#### Arguments
  - `f::Filter`	: Kalman filter output

#### Returns
  - `ll::Real`		: log-likelihood
"""
function loglik_eq(f::Filter)
    (n,T_len)= size(f.v)
	
	# Initialize temp. container
	tmp= Matrix{Float64}(undef, (n,n))

    # Initialize log-likelihood
    ll= zero(Float64)

    # Log-likelihood
    @inbounds @fastmath for t in 1:T_len
		for i in 1:n
			F_it= f.F[i,t]
			if F_it > zero(F_it)
				ll-= .5*(log(2.0*pi) + log(F_it) + f.v[i,t]^2*inv(F_it))
			end
		end
    end		
		
	return ll
end