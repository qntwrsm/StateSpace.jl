#=
univariate_filter.jl

    Kalman filtering routine for a linear Gaussian State Space model based on 
    univariate treatment of multivariate series

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/07
=#

# Struct
struct UnivariateFilter{Ta, TP, Tv, TF, TK} <: KalmanFilter
	a::Ta   # filtered state
	P::TP   # filtered state variance
	v::Tv   # forecast error
	F::TF	# forecast error variance
	K::TK	# Kalman gain
end

"""
	forward!(a_f, P_f, a_i, P_i, K_i, v, F)
	
Compute states ``a`` and corresponding variance ``P`` for series ``i+1`` using
univariate treatment, storing the result in `a_f` and `P_f`.

#### Arguments
  - `a_i::AbstractVector`	: predicted state for series ``i`` (p x 1)
  - `P_i::AbstractMatrix`	: predicted states variance for series ``i`` (p x p)
  - `K_i::AbstractVector`	: Kalman gain for series ``i`` (p x 1)
  - `v::Real`				: forecast error for series ``i``
  - `F::Real`				: forecast error variance for series ``i``

#### Returns
  - `a_f::AbstractVector`	: predicted state at ``t+1`` (p x 1)
  - `P_f::AbstractMatrix`	: predicted states variance at time ``t+1`` (p x p)
"""
function forward!(a_f::AbstractVector, P_f::AbstractMatrix, a_i::AbstractVector, 
						P_i::AbstractMatrix, K_i::AbstractVector, v::Real, F::Real)
	# aᵢ₊₁ = aᵢ + vｘKᵢ
	@. a_f= a_i + v*K_i
	# Pᵢ₊₁ = Pᵢ - KᵢｘKᵢ'/F
	@. P_f= P_i - inv(F)*K_i*transpose(K_i)
	
	return nothing
end

"""
	predict!(a_p, P_p, a_t, P_t, T, Q, c, tmp)
	
Predict states ``a`` and corresponding variance ``P`` at time ``t+1`` using
univariate treatment, storing the result in `a_p` and `P_p`.

#### Arguments
  - `a_t::AbstractVector`	: predicted state at time ``t`` (p x 1)
  - `P_t::AbstractMatrix`	: predicted states variance at time ``t`` (p x p)
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)
  - `Q::AbstractMatrix`		: system matrix ``Q`` (p x p)
  - `c::AbstractVector`		: mean adjustment (p x 1)
  - `tmp::AbstractMatrix`	: tmp storage array (p x p)

#### Returns
  - `a_p::AbstractVector`	: predicted state at time ``t+1`` (p x 1)
  - `P_p::AbstractMatrix`	: predicted states variance at time ``t+1`` (p x p)
"""
function predict!(a_p::AbstractVector, P_p::AbstractMatrix, a_t::AbstractVector, 
						P_t::AbstractMatrix, T::AbstractMatrix, Q::AbstractMatrix, 
						c::AbstractVector, tmp::AbstractMatrix)
	# Predict states
	mul!(a_p, T, a_t)
	a_p.+= c

	# Predict states variance
	# PₜｘT'
	mul!(tmp, P_t, transpose(T))
	# TｘPₜｘT'
	mul!(P_p, T, tmp)
	# Pₜ₊₁ = TｘPₜｘT' + Q
	@. P_p+= Q
	
	return nothing
end

function predict!(a_p::AbstractVector, P_p::AbstractMatrix, a_t::AbstractVector, 
						P_t::AbstractMatrix, T::AbstractMatrix, Q::Diagonal, 
						c::AbstractVector, tmp::AbstractMatrix)
	# Predict states
	mul!(a_p, T, a_t)
	a_p.+= c

	# Predict states variance
	# PₜｘT'
	mul!(tmp, P_t, transpose(T))
	# TｘPₜｘT'
	mul!(P_p, T, tmp)
	# Pₜ₊₁ = TｘPₜｘT' + Q
	@inbounds @fastmath for i in axes(Q,1)
		P_p[i,i]+= Q.diag[i]
	end
	
	return nothing
end

function predict!(a_p::AbstractVector, P_p::AbstractMatrix, a_t::AbstractVector, 
						P_t::AbstractMatrix, T::Diagonal, Q::AbstractMatrix, 
						c::AbstractVector, tmp::AbstractMatrix)
	# Predict states
	@. a_p= T.diag*a_t + c

	# Predict states variance
	@. P_p= T.diag*P_t*transpose(T.diag) + mat.Q
	
	return nothing
end

"""
	kalman_filter!(filter, y, mat) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variances ``P`` and ``F`` and Kalman gain ``K`` for a linear Gaussian State
Space model with system matrices `sys` and data `y` using the
equation-by-equation or univariate version of the Kalman filter, storing the
results in `filter`.

#### Arguments
  - `y::AbstractMatrix`		: data (n x T)
  - `sys::StateSpaceSystem`	: state space system matrices
    
#### Returns
  - `filter::UnivariateFilter`	: Kalman filter output
"""
function kalman_filter!(filter::UnivariateFilter, y::AbstractMatrix, sys::LinearTimeInvariant)
	# Get dims
	(n,T_len)= size(y)
	p= length(sys.a1)
	
    # Initialize temp. containers
    tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Initialize filter
	filter.a[:,1,1]= sys.a1
	filter.P[:,:,1,1]= sys.P1
	
	# Tranpose
	Zt= transpose(sys.Z)
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
		for i in 1:n
	        # Store views 
	        a_it= view(filter.a,:,i,t)
			a_f= view(filter.a,:,i+1,t)
			P_it= view(filter.P,:,:,i,t)
			P_f= view(filter.P,:,:,i+1,t)
	        K_it= view(filter.K,:,i,t)
			Z_i= view(Zt,:,i)
		
			# Forecast error
			filter.v[i,t]= y[i,t] - dot(Z_i, a_it) - sys.d[i]
		
			# Forecast error variance
			filter.F[i,t]= dot(Z_i, P_it, Z_i) + sys.H.diag[i]
		
			# Kalman gain
			mul!(K_it, P_it, Z_i, inv(filter.F[i,t]), .0)
		
			# Move states and variances forward
			forward!(a_f, P_f, a_it, P_it, K_it, filter.v[i,t], filter.F[i,t])
		end
		
		if t < T_len
			# Store views
			a_f= view(filter.a,:,n+1,t)
			P_f= view(filter.P,:,:,n+1,t)
	        a_p= view(filter.a,:,1,t+1)
			P_p= view(filter.P,:,:,1,t+1)
			
			# Predict states and variances
			predict!(a_p, P_p, a_f, P_f, sys.T, sys.Q, sys.c, tmp_p)
		end
	end
	
	return nothing
end

function kalman_filter!(filter::UnivariateFilter, y::AbstractMatrix, sys::LinearTimeVariant)
	# Get dims
	(n,T_len)= size(y)
	p= length(sys.a1)
	
    # Initialize temp. containers
    tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Initialize filter
	filter.a[:,1,1]= sys.a1
	filter.P[:,:,1,1]= sys.P1
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
		# store
		Z_t= sys.Z[t]
		H_t= sys.H[t]
		c_t= view(sys.c,:,t)
		for i in 1:n
	        # Store views 
	        a_it= view(filter.a,:,i,t)
			a_f= view(filter.a,:,i+1,t)
			P_it= view(filter.P,:,:,i,t)
			P_f= view(filter.P,:,:,i+1,t)
	        K_it= view(filter.K,:,i,t)
			Z_it= view(Z_t,i,:)
		
			# Forecast error
			filter.v[i,t]= y[i,t] - dot(Z_it, a_it) - sys.d[i,t]
		
			# Forecast error variance
			filter.F[i,t]= dot(Z_it, P_it, Z_it) + H_t.diag[i]
		
			# Kalman gain
			mul!(K_it, P_it, Z_it, inv(filter.F[i,t]), .0)
		
			# Move states and variances forward
			forward!(a_f, P_f, a_it, P_it, K_it, filter.v[i,t], filter.F[i,t])
		end
		
		if t < T_len
			# Store views
			a_f= view(filter.a,:,n+1,t)
			P_f= view(filter.P,:,:,n+1,t)
	        a_p= view(filter.a,:,1,t+1)
			P_p= view(filter.P,:,:,1,t+1)
			
			# Predict states and variances
			predict!(a_p, P_p, a_f, P_f, sys.T[t], sys.Q[t], c_t, tmp_p)
		end
	end
	
	return nothing
end