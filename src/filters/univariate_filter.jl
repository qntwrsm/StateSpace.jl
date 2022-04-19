#=
univariate_filter.jl

    Kalman filtering routine for a linear Gaussian State Space model based on 
    univariate treatment of multivariate series

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/07
=#

# Struct
struct UnivariateFilter{Ta, TP, Tv, TF, TK, Tm} <: KalmanFilter
	a::Ta   # filtered state
	P::TP   # filtered state variance
	v::Tv   # forecast error
	F::TF	# forecast error variance
	K::TK	# Kalman gain
    tmp::Tm # buffer (p x p)
end
# Constructor
function UnivariateFilter(n::Integer, p::Integer, T_len::Integer, T::Type)
    # filter output
    a= Array{T,3}(undef, p, n+1, T_len)
    P= Array{T,4}(undef, p, p, n+1, T_len)
    v= Matrix{T}(undef, n, T_len)
    F= Matrix{T}(undef, n, T_len)
    K= Array{T,3}(undef, p, n, T_len)

    # buffer
    tmp= Matrix{T}(undef, p, p)

    return UnivariateFilter(a, P, v, F, K, tmp)
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
	a_f.= a_i .+ v .* K_i

	# Pᵢ₊₁ = Pᵢ - KᵢｘKᵢ'×F
	P_f.= P_i .- F .* K_i .* transpose(K_i)
	
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
	P_p.+= Q
	
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
						P_t::AbstractMatrix, T::Diagonal, Q::Diagonal, 
						c::AbstractVector, tmp::AbstractMatrix)
	# Predict states
	a_p.= T.diag .* a_t .+ c

	# Predict states variance
	P_p.= T.diag .* P_t .* transpose(T.diag) .+ Q
	
	return nothing
end

"""
    update_filter!(filter, y, Z, d, σ², i, t)

Update Kalman filter components at time `t`, storing the results in `filter`.

#### Arguments
  - `y::Real`           : data
  - `Z::AbstractVector` : system matrix ``Z``
  - `d::Real`           : mean adjustment observation
  - `σ²::Real`          : observation equation variance
  - `i::Integer`        : series
  - `t::Integer`        : time

#### Returns
  - `filter::UnivariateFilter`  : Kalman filter components
"""
function update_filter!(filter::UnivariateFilter, 
                        y::Real, 
                        Z::AbstractVector, 
                        d::Real, 
                        σ²::Real, 
                        i::Integer,
                        t::Integer)
    # Store views 
    a= view(filter.a,:,i,t)
    a_f= view(filter.a,:,i+1,t)
    P= view(filter.P,:,:,i,t)
    P_f= view(filter.P,:,:,i+1,t)
    K= view(filter.K,:,i,t)

    # Forecast error
    filter.v[i,t]= y - dot(Z, a) - d

    # Forecast error variance
    filter.F[i,t]= dot(Z, P, Z) + σ²

    # Kalman gain
    mul!(K, P, Z, inv(filter.F[i,t]), .0)

    # Move states and variances forward
    forward!(a_f, P_f, a, P, K, filter.v[i,t], filter.F[i,t])

    return nothing
end

"""
    update_filter!(filter, Z, σ², i, t)

Update Kalman filter components at time `t` when observation ``y`` is missing,
i.e. NaN, storing the results in `filter`.

#### Arguments
  - `Z::AbstractVector` : system matrix ``Z``
  - `σ²::Real`          : observation equation variance
  - `i::Integer`        : series
  - `t::Integer`        : time

#### Returns
  - `filter::KalmanFilter`  : Kalman filter components
"""
function update_filter!(filter::UnivariateFilter, 
                        Z::AbstractVector, 
                        σ²::Real, 
                        i::Integer,
                        t::Integer)
    # Store views 
    a= view(filter.a,:,i,t)
    a_f= view(filter.a,:,i+1,t)
    P= view(filter.P,:,:,i,t)
    P_f= view(filter.P,:,:,i+1,t)
    K= view(filter.K,:,i,t)

    # Forecast error
    filter.v[i,t]= NaN

    # Forecast error variance
    filter.F[i,t]= dot(Z, P, Z) + σ²

    # Kalman gain
    K.= zero(eltype(K))

    # Move states and variances forward
    a_f.= a
    P_f.= P

    return nothing
end

"""
	kalman_filter!(filter, sys) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variances ``P`` and ``F`` and Kalman gain ``K`` for a linear Gaussian State
Space model with system matrices `sys` using the equation-by-equation or
univariate version of the Kalman filter, storing the results in `filter`.

#### Arguments
  - `sys::StateSpaceSystem`	: state space system matrices
    
#### Returns
  - `filter::UnivariateFilter`	: Kalman filter output
"""
function kalman_filter!(filter::UnivariateFilter, sys::LinearTimeInvariant)
	# Get dims
	(n,T_len)= size(sys.y)
	
	# Initialize filter
	filter.a[:,1,1]= sys.a1
	filter.P[:,:,1,1]= sys.P1
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
		for i in 1:n
	        # Store view
			Z_i= view(Z,i,:)

            # update filter
            if isnan(sys.y[i,t])
                update_filter!(filter, Z_i, sys.H.diag[i], i, t)
            else
                update_filter!(filter, sys.y[i,t], Z_i, sys.d[i], sys.H.diag[i], i, t)
            end
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

function kalman_filter!(filter::UnivariateFilter, sys::LinearTimeVariant)
	# Get dims
	(n,T_len)= size(sys.y)
	
	# Initialize filter
	filter.a[:,1,1]= sys.a1
	filter.P[:,:,1,1]= sys.P1
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
		# store
		Z_t= sys.Z[t]
		H_t= sys.H[t]
		
		for i in 1:n
	        # Store view
			Z_it= view(Z_t,i,:)
		
			# update filter
            if isnan(sys.y[i,t])
                update_filter!(filter, Z_it, H_t.diag[i], i, t)
            else
                update_filter!(filter, sys.y[i,t], Z_it, sys.d[i,t], H_t.diag[i], i, t)
            end
		end
		
		if t < T_len
			# Store views
			a_f= view(filter.a,:,n+1,t)
			P_f= view(filter.P,:,:,n+1,t)
	        a_p= view(filter.a,:,1,t+1)
			P_p= view(filter.P,:,:,1,t+1)
            c_t= view(sys.c,:,t)
			
			# Predict states and variances
			predict!(a_p, P_p, a_f, P_f, sys.T[t], sys.Q[t], c_t, tmp_p)
		end
	end
	
	return nothing
end