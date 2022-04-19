#=
multivariate_filter.jl

    Multivariate Kalman filtering routines for a linear Gaussian State Space 
	model

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2020/07/28
=#

# Structs
struct MultivariateFilter{Ta, TP, Tv, TF, TK, Tm} <: KalmanFilter
	a::Ta	    # filtered state
	P::TP	    # filtered state variance
	v::Tv  	    # forecast error
	F::TF	    # forecast error variance
	K::TK	    # Kalman gain
    tmp_pn::Tm  # buffer
    tmp_p::Tm   # buffer
    tmp_n::Tm   # buffer
end
# Constructor
function MultivariateFilter(n::Integer, p::Integer, T_len::Integer, T::Type)
    # filter output
    a= Matrix{T}(undef, p, T_len)
    P= Array{T,3}(undef, p, p, T_len)
    v= Matrix{T}(undef, n, T_len)
    F= Array{T,3}(undef, n, n, T_len)
    K= Array{T,3}(undef, p, n, T_len)

    # buffers
    tmp_pn= Matrix{T}(undef, p, n)
    tmp_p= Matrix{T}(undef, p, p)
    tmp_n= Matrix{T}(undef, n, n)

    return MultivariateFilter(a, P, v, F, K, tmp_pn, tmp_p, tmp_n)
end

struct WoodburyFilter{Ta, TP, Tv, TFi, TK, Tm} <: KalmanFilter
	a::Ta	    # filtered state
	P::TP	    # filtered state variance
	v::Tv	    # forecast error
	Fi::TFi	    # inverse forecast error variance
	K::TK	    # Kalman gain
    tmp_np::Tm  # buffer
    tmp_pn::Tm  # buffer
    tmp_p::Tm   # buffer
end
# Constructor
function WoodburyFilter(n::Integer, p::Integer, T_len::Integer, T::Type)
    # filter output
    a= Matrix{T}(undef, p, T_len)
    P= Array{T,3}(undef, p, p, T_len)
    v= Matrix{T}(undef, n, T_len)
    Fi= Array{T,3}(undef, n, n, T_len)
    K= Array{T,3}(undef, p, n, T_len)

    # buffers
    tmp_np= Matrix{T}(undef, n, p)
    tmp_pn= Matrix{T}(undef, p, n)
    tmp_p= Matrix{T}(undef, p, p)

    return WoodburyFilter(a, P, v, Fi, K, tmp_np, tmp_pn, tmp_p)
end

"""
	error!(v_t, y, Z, a_t, d)
	
Compute forecast error ``v`` at time ``t``, storing the result in `v_t`.

#### Arguments
  - `y::AbstractVector`		: data (n x 1)
  - `Z::AbstractMatrix`		: system matrix ``Z`` (n x p)
  - `a_t::AbstractVector`	: predicted state (p x 1)
  - `d::AbstractVector`		: mean adjustment (n x 1)

#### Returns
  - `v_t::AbstractVector`	: forecast error (n x 1)
"""
function error!(v_t::AbstractVector, y::AbstractVector, Z::AbstractMatrix, 
				a_t::AbstractVector, d::AbstractVector)
	# yₜ - d
	v_t.= y .- d
    # vₜ = yₜ - Z×aₜ - d
    mul!(v_t, Z, a_t, -1., 1.)
	
	return nothing
end

"""
	error_var!(F_t, P_t, Z, H, tmp)
	
Compute forecast error variance ``F`` at time ``t``, storing the result in
`F_t`.

#### Arguments
  - `P_t::AbstractMatrix`	: predicted state variance (p x p)
  - `Z::AbstractMatrix`		: system matrix ``Z`` (n x p)
  - `H::AbstractMatrix`		: system matrix ``H`` (n x n)
  - `tmp::AbstractMatrix`	: tmp storage array (p x n)

#### Returns
  - `F_t::AbstractMatrix`	: forecast error variance (n x n)
"""
function error_var!(F_t::AbstractMatrix, P_t::AbstractMatrix, Z::AbstractMatrix, 
					H::AbstractMatrix, tmp::AbstractMatrix)
    # PₜｘZ'
	mul!(tmp, P_t, transpose(Z))
    # ZｘPₜｘZ'
	mul!(F_t, Z, tmp)
	# Fₜ = ZｘPₜｘZ' + H
	F_t.+= H
	
	return nothing
end

function error_var!(F_t::AbstractMatrix, P_t::AbstractMatrix, Z::AbstractMatrix, 
					H::Diagonal, tmp::AbstractMatrix)	
    # PₜｘZ'
	mul!(tmp, P_t, transpose(Z))
    # ZｘPₜｘZ'
	mul!(F_t, Z, tmp)
	# Fₜ = ZｘPₜｘZ' + H
	@inbounds @fastmath for i in axes(H,1) 
		F_t[i,i]+= H.diag[i]
	end
	
	return nothing
end

"""
	error_prec!(Fi_t, P_t, Z, H, tmp)
	
Compute forecast error precision ``F⁻¹`` at time ``t`` using Woodbury's
identity, storing the result in `Fi_t`.

#### Arguments
  - `P_t::AbstractMatrix`	: predicted state variance (p x p)
  - `Z::AbstractMatrix`		: system matrix ``Z`` (n x p)
  - `Hi::AbstractMatrix`	: inverse system matrix ``H`` (n x n)
  - `tmp_np::AbstractMatrix`: tmp storage array (n x p)
  - `tmp_pn::AbstractMatrix`: tmp storage array (p x n)
  - `tmp_p::AbstractMatrix`	: tmp storage array (p x p)

#### Returns
  - `Fi_t::AbstractMatrix`	: forecast error precision (n x n)
"""
function error_prec!(Fi_t::AbstractMatrix, P_t::AbstractMatrix, Z::AbstractMatrix, 
					Hi::AbstractMatrix, tmp_np::AbstractMatrix, tmp_pn::AbstractMatrix, 
					tmp_p::AbstractMatrix)
	# Pₜ⁻¹
	copyto!(tmp_p, P_t)
	LinearAlgebra.inv!(cholesky!(Hermitian(tmp_p)))
	# H⁻¹×Z
	mul!(tmp_np, Hi, Z)
	# Pₜ⁻¹ + Z'×H⁻¹×Z
	mul!(tmp_p, transpose(Z), tmp_np, 1., 1.)
	# (Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹
	LinearAlgebra.inv!(cholesky!(Hermitian(tmp_p)))
	# (Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹×Z′×H⁻¹
	mul!(tmp_pn, tmp_p, transpose(tmp_np))
	# -H⁻¹×Z×(Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹×Z'×H⁻¹
	mul!(Fi_t, tmp_np, tmp_pn, -1., .0)
	# Fₜ⁻¹ = H⁻¹ - H⁻¹×Z×(Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹×Z'×H⁻¹
	Fi_t.+= Hi
	
	return nothing
end

function error_prec!(Fi_t::AbstractMatrix, P_t::AbstractMatrix, Z::AbstractMatrix, 
					Hi::Diagonal, tmp_np::AbstractMatrix, tmp_pn::AbstractMatrix, 
					tmp_p::AbstractMatrix)
	# Pₜ⁻¹
	copyto!(tmp_p, P_t)
	C= cholesky!(Hermitian(tmp_p))		# pointer(C.factors) = pointer(tmp_p)
	LinearAlgebra.inv!(C)
	# H⁻¹×Z
	mul!(tmp_np, Hi, Z)
	# Pₜ⁻¹ + Z'×H⁻¹×Z
	mul!(Pi, transpose(Z), tmp_np, 1., 1.)
	# (Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹
	C= cholesky!(Hermitian(tmp_p))		# pointer(C.factors) = pointer(tmp_p)
	LinearAlgebra.inv!(C)
	# (Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹×Z′×H⁻¹
	mul!(tmp_pn, tmp_p, transpose(tmp_np))
	# -H⁻¹×Z×(Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹×Z'×H⁻¹
	mul!(Fi_t, tmp_np, tmp_pn, -1., .0)
	# Fₜ⁻¹ = H⁻¹ - H⁻¹×Z×(Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹×Z'×H⁻¹
	@inbounds @fastmath for i in axes(Hi,1)
		Fi_t[i,i]+= Hi.diag[i]
	end
	
	return nothing
end

"""
	gain!(K_t, P_t, Fi_t, Z, T, tmp)
	
Compute Kalman gain ``K`` at time ``t`` using the inverse of ``F``, storing
the result in `K_t`.

#### Arguments
  - `P_t::AbstractMatrix`	: predicted state variance (p x p)
  - `Fi_t::AbstractMatrix`	: inverse of ``Fₜ`` (n x n)
  - `Z::AbstractMatrix`		: system matrix ``Z`` (n x p)
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)
  - `tmp::AbstractMatrix`	: tmp storage array (p x n)

#### Returns
  - `K_t::AbstractMatrix`	: Kalman gain (p x n)
"""
function gain!(K_t::AbstractMatrix, P_t::AbstractMatrix, Fi_t::AbstractMatrix, 
				Z::AbstractMatrix, T::AbstractMatrix, tmp::AbstractMatrix)
    # PₜｘZ'
	mul!(K_t, P_t, transpose(Z))
    # PₜｘZ'ｘFₜ⁻¹
	mul!(tmp, K_t, Fi_t)
	# Kₜ = TｘPₜｘZ'ｘFₜ⁻¹
	mul!(K_t, T, tmp)
	
	return nothing
end

"""
	gain!(K_t, P_t, fac, Z, T, tmp)
	
Compute Kalman gain ``K`` at time ``t`` using the factorization of ``F``, storing
the result in `K_t`.
"""
function gain!(K_t::AbstractMatrix, P_t::AbstractMatrix, fac::Factorization, 
				Z::AbstractMatrix, T::AbstractMatrix, tmp::AbstractMatrix)
    # PₜｘZ'
	mul!(tmp, P_t, transpose(Z))
    # PₜｘZ'ｘFₜ⁻¹
	rdiv!(tmp, fac)
	# Kₜ = TｘPₜｘZ'ｘFₜ⁻¹
	mul!(K_t, T, tmp)
	
	return nothing
end

"""
	predict_state!(a_p, a_t, K_t, v_t, T, c)
	
Predict states ``a`` at time ``t+1``, storing the result in `a_p`.

#### Arguments
  - `a_t::AbstractVector`	: predicted state at time ``t`` (p x 1)
  - `K_t::AbstractMatrix`	: Kalman gain (p x n)
  - `v_t::AbstractVector`	: forecast error (n x 1)
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)
  - `c::AbstractVector`		: mean adjustment (p x 1)

#### Returns
  - `a_p::AbstractVector`	: predicted state at time ``t+1`` (p x 1)
"""
function predict_state!(a_p::AbstractVector, a_t::AbstractVector, K_t::AbstractMatrix, 
						v_t::AbstractVector, T::AbstractMatrix, c::AbstractVector)
    # Kₜｘvₜ
    mul!(a_p, K_t, v_t)
    # aₜ₊₁ = Tｘaₜ + Kₜｘvₜ
	mul!(a_p, T, a_t, 1., 1.)
	# aₜ₊₁ = Tｘaₜ + Kₜｘvₜ + c
	a_p.+= c
	
	return nothing
end

"""
	predict_state_var!(P_p, P_t, K_t, Z, T, Q, tmp)
	
Predict states variance ``P`` at time ``t+1``, storing the result in `P_p`.

#### Arguments
  - `P_t::AbstractMatrix`	: predicted states variance at time ``t`` (p x p)
  - `K_t::AbstractMatrix`	: Kalman gain (p x n)
  - `Z::AbstractMatrix`		: system matrix ``Z`` (n x p)
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)
  - `Q::AbstractMatrix`		: system matrix ``Q`` (p x p)
  - `tmp::AbstractMatrix`	: tmp storage array (p x p)

#### Returns
  - `P_p::AbstractMatrix`	: predicted states variance at time ``t+1`` (p x p)
"""
function predict_state_var!(P_p::AbstractMatrix, P_t::AbstractMatrix, K_t::AbstractMatrix, 
						Z::AbstractMatrix, T::AbstractMatrix, Q::AbstractMatrix, 
						tmp::AbstractMatrix)
    # -KₜｘZ
	mul!(P_p, K_t, Z, -1., .0)
	# T - KₜｘZ
	P_p.+= T
    # Pₜｘ(T - KₜｘZ)'
	mul!(tmp, P_t, transpose(P_p))
	# TｘPₜｘ(T - KₜｘZ)'
	mul!(P_p, T, tmp)
	# TｘPₜｘ(T - KₜｘZ)' + Q
	P_p.+= Q
	
	return nothing
end

function predict_state_var!(P_p::AbstractMatrix, P_t::AbstractMatrix, K_t::AbstractMatrix, 
						Z::AbstractMatrix, T::Diagonal, Q::AbstractMatrix, 
						tmp::AbstractMatrix)
    # -KₜｘZ
	mul!(P_p, K_t, Z, -1., .0)
	# T - KₜｘZ
	@inbounds @fastmath for i in axes(T,1)
		P_p[i,i]+= T.diag[i]
	end
    # Pₜｘ(T - KₜｘZ)'
	mul!(tmp, P_t, transpose(P_p))
	# TｘPₜｘ(T - KₜｘZ)'
	mul!(P_p, T, tmp)
	# TｘPₜｘ(T - KₜｘZ)' + Q
	P_p.+= Q
	
	return nothing
end

function predict_state_var!(P_p::AbstractMatrix, P_t::AbstractMatrix, K_t::AbstractMatrix, 
						Z::AbstractMatrix, T::AbstractMatrix, Q::Diagonal, 
						tmp::AbstractMatrix)
    # -KₜｘZ
	mul!(P_p, K_t, Z, -1., .0)
	# T - KₜｘZ
	P_p.+= T
    # Pₜｘ(T - KₜｘZ)'
	mul!(tmp, P_t, transpose(P_p))
	# TｘPₜｘ(T - KₜｘZ)'
	mul!(P_p, T, tmp)
	# TｘPₜｘ(T - KₜｘZ)' + Q
	@inbounds @fastmath for i in axes(Q,1)
		P_p[i,i]+= Q.diag[i]
	end
	
	return nothing
end

function predict_state_var!(P_p::AbstractMatrix, P_t::AbstractMatrix, K_t::AbstractMatrix, 
						Z::AbstractMatrix, T::Diagonal, Q::Diagonal, 
						tmp::AbstractMatrix)
    # -KₜｘZ
	mul!(P_p, K_t, Z, -1., .0)
	# T - KₜｘZ
	@inbounds @fastmath for i in axes(T,1)
		P_p[i,i]+= T.diag[i]
	end
    # Pₜｘ(T - KₜｘZ)'
	mul!(tmp, P_t, transpose(P_p))
	# TｘPₜｘ(T - KₜｘZ)'
	mul!(P_p, T, tmp)
	# TｘPₜｘ(T - KₜｘZ)' + Q
	@inbounds @fastmath for i in axes(Q,1)
		P_p[i,i]+= Q.diag[i]
	end
	
	return nothing
end

"""
	predict_state_var!(P_p, P_t, T, Q, tmp)
	
Predict states variance ``P`` at time ``t+1`` when observation ``y`` at time
``t`` is missing, i.e. NaN, storing the result in `P_p`.

#### Arguments
  - `P_t::AbstractMatrix`	: predicted states variance at time ``t`` (p x p)
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)
  - `Q::AbstractMatrix`		: system matrix ``Q`` (p x p)
  - `tmp::AbstractMatrix`	: tmp storage array (p x p)

#### Returns
  - `P_p::AbstractMatrix`	: predicted states variance at time ``t+1`` (p x p)
"""
function predict_state_var!(P_p::AbstractMatrix, P_t::AbstractMatrix, 
                            T::AbstractMatrix, Q::AbstractMatrix, 
						    tmp::AbstractMatrix)
    # PₜｘT'
    mul!(tmp, P_t, transpose(T))
	# TｘPₜｘT'
	mul!(P_p, T, tmp)
	# TｘPₜｘT' + Q
	P_p.+= Q
	
	return nothing
end

function predict_state_var!(P_p::AbstractMatrix, P_t::AbstractMatrix, T::Diagonal, 
                            Q::AbstractMatrix, tmp::AbstractMatrix)
	# TｘPₜｘT' + Q
	P_p.= T.diag .* P_t .* transpose(T.diag) .+ Q
	
	return nothing
end

function predict_state_var!(P_p::AbstractMatrix, P_t::AbstractMatrix, 
                            T::AbstractMatrix, Q::Diagonal, tmp::AbstractMatrix)
    # PₜｘT'
    mul!(tmp, P_t, transpose(T))
	# TｘPₜｘT'
	mul!(P_p, T, tmp)
	# TｘPₜｘT' + Q
	@inbounds @fastmath for i in axes(Q,1)
		P_p[i,i]+= Q.diag[i]
	end
	
	return nothing
end

function predict_state_var!(P_p::AbstractMatrix, P_t::AbstractMatrix, T::Diagonal, 
                            Q::Diagonal, tmp::AbstractMatrix)
    # TｘPₜｘT'
	P_p.= T.diag .* P_t .* transpose(T.diag)
    # TｘPₜｘT' + Q
    @inbounds @fastmath for i in axes(Q,1)
		P_p[i,i]+= Q.diag[i]
	end
	
	return nothing
end

"""
    update_filter!(filter, y, Z, T, d, c, H, Q, t)

Update Kalman filter components at time `t`, storing the results in `filter`.

#### Arguments
  - `y::AbstractVector`     : data
  - `Z::AbstractMatrix`     : system matrix ``Z``
  - `T::AbstractMatrix`     : system matrix ``T``
  - `d::AbstractVector`     : mean adjustment observation
  - `c::AbstractVector`     : mean adjustment state
  - `H::AbstractMatrix`     : system matrix ``H``
  - `Q::AbstractMatrix`     : system matrix ``Q``
  - `t::Integer`            : time

#### Returns
  - `filter::KalmanFilter`  : Kalman filter components
"""
function update_filter!(filter::MultivariateFilter, 
                        y::AbstractVector, 
                        Z::AbstractMatrix, T::AbstractMatrix, 
                        d::AbstractVector, c::AbstractVector, 
                        H::AbstractMatrix, Q::AbstractMatrix, 
                        t::Integer)
    # Store views 
    a= view(filter.a,:,t)
    P= view(filter.P,:,:,t)
    v= view(filter.v,:,t)
    F= view(filter.F,:,:,t)
    K= view(filter.K,:,:,t)
    
    # Forecast error
    error!(v, y, Z, a, d)
    
    # Forecast error variance
    error_var!(F, P, Z, H, filter.tmp_pn)
    
    # Cholesky factorization of Fₜ
    copyto!(filter.tmp_n, F)
    fac= cholesky!(Hermitian(filter.tmp_n))
    
    # Kalman gain
    gain!(K, P, fac, Z, T, filter.tmp_pn)
    
    if t < T_len
        # Store views
        a_p= view(filter.a,:,t+1)
        P_p= view(filter.P,:,:,t+1)
        
        # Predict states
        predict_state!(a_p, a, K, v, T, c)
        
        # Predict states variance
        predict_state_var!(P_p, P, K, Z, T, Q, filter.tmp_p)
    end

    return nothing
end

function update_filter!(filter::WoodburyFilter, 
                        y::AbstractVector, 
                        Z::AbstractMatrix, T::AbstractMatrix, 
                        d::AbstractVector, c::AbstractVector, 
                        Hi::AbstractMatrix, Q::AbstractMatrix, 
                        t::Integer)
    # Store views 
    a= view(filter.a,:,t)
    P= view(filter.P,:,:,t)
    v= view(filter.v,:,t)
    Fi= view(filter.Fi,:,:,t)
    K= view(filter.K,:,:,t)
    
    # Forecast error
    error!(v, y, Z, a, d)
    
    # Forecast error precision
    error_prec!(Fi, P, Z, Hi, filter.tmp_np, filter.tmp_pn, filter.tmp_p)
    
    # Kalman gain
    gain!(K, P, Fi, Z, T, filter.tmp_pn)
    
    if t < T_len
        # Store views
        a_p= view(filter.a,:,t+1)
        P_p= view(filter.P,:,:,t+1)
        
        # Predict states
        predict_state!(a_p, a, K, v, T, c)
        
        # Predict states variance
        predict_state_var!(P_p, P, K, Z, T, Q, filter.tmp_p)
    end

    return nothing
end

"""
    update_filter!(filter, T, d, c, H, Q, t)

Update Kalman filter components at time `t` when observation ``y_t`` is missing,
i.e. NaN, storing the results in `filter`.

#### Arguments
  - `T::AbstractMatrix`     : system matrix ``T``
  - `d::AbstractVector`     : mean adjustment observation
  - `c::AbstractVector`     : mean adjustment state
  - `H::AbstractMatrix`     : system matrix ``H``
  - `Q::AbstractMatrix`     : system matrix ``Q``
  - `t::Integer`            : time

#### Returns
  - `filter::KalmanFilter`  : Kalman filter components
"""
function update_filter!(filter::MultivariateFilter, 
                        Z::AbstractMatrix, T::AbstractMatrix, 
                        c::AbstractVector, 
                        H::AbstractMatrix, Q::AbstractMatrix, 
                        t::Integer)
    # Store views 
    a= view(filter.a,:,t)
    P= view(filter.P,:,:,t)
    v= view(filter.v,:,t)
    F= view(filter.F,:,:,t)
    K= view(filter.K,:,:,t)
    
    # Forecast error
    v.= NaN
    
    # Forecast error variance
    error_var!(F, P, Z, H, filter.tmp_pn)
    
    # Kalman gain
    K.= zero(eltype(K))
    
    if t < T_len
        # Store views
        a_p= view(filter.a,:,t+1)
        P_p= view(filter.P,:,:,t+1)
        
        # Predict states
        a_p.= c
        mul!(a_p, T, a, 1., 1.)
        
        # Predict states variance
        predict_state_var!(P_p, P, T, Q, filter.tmp_p)
    end

    return nothing
end

function update_filter!(filter::WoodburyFilter,
                        Z::AbstractMatrix, T::AbstractMatrix, 
                        c::AbstractVector, 
                        Hi::AbstractMatrix, Q::AbstractMatrix, 
                        t::Integer)
    # Store views 
    a= view(filter.a,:,t)
    P= view(filter.P,:,:,t)
    v= view(filter.v,:,t)
    Fi= view(filter.F,:,:,t)
    K= view(filter.K,:,:,t)
    
    # Forecast error
    v.= NaN
    
    # Forecast error precision
    error_prec!(Fi, P, Z, Hi, filter.tmp_np, filter.tmp_pn, filter.tmp_p)
    
    # Kalman gain
    K.= zero(eltype(K))
    
    if t < T_len
        # Store views
        a_p= view(filter.a,:,t+1)
        P_p= view(filter.P,:,:,t+1)
        
        # Predict states
        a_p.= c
        mul!(a_p, T, a, 1., 1.)
        
        # Predict states variance
        predict_state_var!(P_p, P, T, Q, filter.tmp_p)
    end

    return nothing
end

"""
	kalman_filter!(filter, sys) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variances ``P`` and ``F`` and Kalman gain ``K`` for a linear Gaussian State
Space model with system matrices `sys` using the Kalman filter, storing the
results in `filter`.

#### Arguments
  - `sys::StateSpaceSystem`	: state space system matrices
  
#### Returns
  - `filter::MultivariateFilter`: Kalman filter output
"""
function kalman_filter!(filter::MultivariateFilter, sys::LinearTimeInvariant)
	# Get dims
	T_len= size(sys.y,2)
	
	# Initialize filter
	filter.a[:,1]= sys.a1
	filter.P[:,:,1]= sys.P1
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
        # data
        y_t= view(sys.y, :, t)

        # update filter
        if any(isnan, y_t)
            update_filter!(filter, sys.Z, sys.T, sys.c, sys.H, sys.Q, t)
        else
            update_filter!(filter, y_t, sys.Z, sys.T, sys.d, sys.c, sys.H, sys.Q, t)
        end
	end
	
	return nothing
end

function kalman_filter!(filter::MultivariateFilter, sys::LinearTimeVariant)
	# Get dims
	T_len= size(sys.y,2)
	
	# Initialize filter
	filter.a[:,1]= sys.a1
	filter.P[:,:,1]= sys.P1
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
        # data
        y_t= view(sys.y, :, t)

        # means
        d_t= view(sys.d, :, t)
        c_t= view(sys.c, :, t)

        # update filter
        if any(isnan, y_t)
            update_filter!(filter, sys.Z[t], sys.T[t], c_t, sys.H[t], sys.Q[t], t)
        else
            update_filter!(filter, y_t, sys.Z[t], sys.T[t], d_t, c_t, sys.H[t], sys.Q[t], t)
        end
	end
	
	return nothing
end

"""
	kalman_filter!(filter, sys) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variance ``P`` and precision ``F⁻¹`` and Kalman gain ``K`` for a linear Gaussian
State Space model with system matrices `sys` using the Kalman filter based on
Woodbury's Identity, storing the results in `filter`.

Woodbury's Identity allows direct computation of the inverse variance
(precision) ``F⁻¹``.

#### Arguments
  - `sys::StateSpaceSystem`	: state space system matrices
  
#### Returns
  - `filter::WoodburyFilter`: Kalman filter output
"""
function kalman_filter!(filter::WoodburyFilter, sys::LinearTimeInvariant)
	# Get dims
	T_len= size(sys.y,2)
	
	# Inverse of H
	Hi= copy(sys.H)
	if sys.H isa Diagonal
		Hi.diag.= inv.(sys.H.diag)
	else
		LinearAlgebra.inv!(cholesky!(Hi))
	end
	
	# Initialize filter
	filter.a[:,1]= sys.a1
	filter.P[:,:,1]= sys.P1
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
        # data
        y_t= view(sys.y, :, t)

        # update filter
        if any(isnan, y_t)
            update_filter!(filter, sys.Z, sys.T, sys.c, Hi, sys.Q, t)
        else
            update_filter!(filter, y_t, sys.Z, sys.T, sys.d, sys.c, Hi, sys.Q, t)
        end
	end
	
	return nothing
end

function kalman_filter!(filter::WoodburyFilter, sys::LinearTimeVariant)
	# Get dims
	T_len= size(sys.y,2)
	
    # Initialize temp. containers
	Hi= similar(sys.H[1])
	
	# Initialize filter
	filter.a[:,1]= sys.a1
	filter.P[:,:,1]= sys.P1
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
        # data
        y_t= view(sys.y, :, t)

        # means
        d_t= view(sys.d, :, t)
        c_t= view(sys.c, :, t)

        # Inverse of H
        Hi.= sys.H[t]
        if sys.H[t] isa Diagonal
            Hi.diag.= inv.(sys.H[t].diag)
        else
            LinearAlgebra.inv!(cholesky!(Hi))
        end

        # update filter
        if any(isnan, y_t)
            update_filter!(filter, sys.Z[t], sys.T[t], c_t, Hi, sys.Q[t], t)
        else
            update_filter!(filter, y_t, sys.Z[t], sys.T[t], d_t, c_t, Hi, sys.Q[t], t)
        end
	end
	
	return nothing
end