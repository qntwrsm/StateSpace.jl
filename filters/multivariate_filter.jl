#=
multivariate_filter.jl

    Multivariate Kalman filtering routines for a linear Gaussian State Space 
	model

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2020/07/28
=#

# Structs
struct MultivariateFilter{Ta, TP, Tv, TF, TK} <: KalmanFilter
	a::Ta	# filtered state
	P::TP	# filtered state variance
	v::Tv  	# forecast error
	F::TF	# forecast error variance
	K::TK	# Kalman gain
end

struct WoodburyFilter{Ta, TP, Tv, TFi, TK} <: KalmanFilter
	a::Ta	# filtered state
	P::TP	# filtered state variance
	v::Tv	# forecast error
	Fi::TFi	# inverse forecast error variance
	K::TK	# Kalman gain
end

"""
	error!(v_t, y, Z, a_t)
	
Compute forecast error ``v`` at time ``t``, storing the result in `v_t`.

#### Arguments
  - `y::AbstractVector`		: data (n x 1)
  - `Z::AbstractMatrix`		: system matrix ``Z`` (n x p)
  - `a_t::AbstractVector`	: predicted state (p x 1)

#### Returns
  - `v_t::AbstractVector`	: forecast error (n x 1)
"""
function error!(v_t::AbstractVector, y::AbstractVector, Z::AbstractMatrix, a_t::AbstractVector)
    # Z×aₜ
    mul!(v_t, Z, a_t, -1., .0)
    # vₜ = yₜ - Z×aₜ
	v_t.+= y
	
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
	@. F_t+= H
	
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
	LinearAlgebra.inv!(cholesky!(tmp_p))
	# H⁻¹×Z
	mul!(tmp_np, Hi, Z)
	# Pₜ⁻¹ + Z'×H⁻¹×Z
	mul!(tmp_p, transpose(Z), tmp_np, 1., 1.)
	# (Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹
	LinearAlgebra.inv!(cholesky!(tmp_p))
	# (Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹×Z′×H⁻¹
	mul!(tmp_pn, tmp_p, transpose(HiZ))
	# -H⁻¹×Z×(Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹×Z'×H⁻¹
	mul!(Fi_t, tmp_np, tmp_pn, -1., .0)
	# Fₜ⁻¹ = H⁻¹ - H⁻¹×Z×(Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹×Z'×H⁻¹
	@. Fi_t+= Hi
	
	return nothing
end

function error_prec!(Fi_t::AbstractMatrix, P_t::AbstractMatrix, Z::AbstractMatrix, 
					Hi::Diagonal, tmp_np::AbstractMatrix, tmp_pn::AbstractMatrix, 
					tmp_p::AbstractMatrix)
	# Pₜ⁻¹
	copyto!(tmp_p, P_t)
	C= cholesky!(tmp_p)		# pointer(C.factors) = pointer(tmp_p)
	LinearAlgebra.inv!(C)
	# H⁻¹×Z
	mul!(tmp_np, Hi, Z)
	# Pₜ⁻¹ + Z'×H⁻¹×Z
	mul!(Pi, transpose(Z), tmp_np, 1., 1.)
	# (Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹
	C= cholesky!(tmp_p)		# pointer(C.factors) = pointer(tmp_p)
	LinearAlgebra.inv!(C)
	# (Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹×Z′×H⁻¹
	mul!(tmp_pn, tmp_p, transpose(HiZ))
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
	mul!(K_t, P_t, tranpose(Z))
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
	mul!(K_t, P_t, tranpose(Z))
    # PₜｘZ'ｘFₜ⁻¹
	rdiv!(tmp, fac)
	# Kₜ = TｘPₜｘZ'ｘFₜ⁻¹
	mul!(K_t, T, tmp)
	
	return nothing
end

"""
	predict_state!(a_p, a_t, K_t, v_t, T)
	
Predict states ``a`` at time ``t+1``, storing the result in `a_p`.

#### Arguments
  - `a_t::AbstractVector`	: predicted state at time ``t`` (p x 1)
  - `K_t::AbstractMatrix`	: Kalman gain (p x n)
  - `v_t::AbstractVector`	: forecast error (n x 1)
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)

#### Returns
  - `a_p::AbstractVector`	: predicted state at time ``t+1`` (p x 1)
"""
function predict_state!(a_p::AbstractVector, a_t::AbstractVector, K_t::AbstractMatrix, 
						v_t::AbstractVector, T::AbstractMatrix)
    # Kₜｘvₜ
    mul!(a_p, K_t, v_t)
    # aₜ₊₁ = Tｘaₜ + Kₜｘvₜ
	mul!(a_p, T, a_t, 1., 1.)
	
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
	@. P_p+= T
    # Pₜｘ(T - KₜｘZ)'
	mul!(tmp, P_t, transpose(P_p))
	# TｘPₜｘ(T - KₜｘZ)'
	mul!(P_p, T, tmp)
	# TｘPₜｘ(T - KₜｘZ)' + Q
	@. P_p+= Q
	
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
	@. P_p+= Q
	
	return nothing
end

function predict_state_var!(P_p::AbstractMatrix, P_t::AbstractMatrix, K_t::AbstractMatrix, 
						Z::AbstractMatrix, T::AbstractMatrix, Q::Diagonal, 
						tmp::AbstractMatrix)
    # -KₜｘZ
	mul!(P_p, K_t, Z, -1., .0)
	# T - KₜｘZ
	@. P_p+= T
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
	kalman_filter!(f, Y, mat) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variances ``P`` and ``F`` and Kalman gain ``K`` for a linear Gaussian State
Space model with system matrices `mat` and data `Y` using the Kalman filter,
storing the results in `f`.

#### Arguments
  - `Y::AbstractMatrix`	: data (n x T)
  - `mat::SysMat`		: State Space system matrices
  
#### Returns
  - `f::MultivariateFilter`	: Kalman filter output
"""
function kalman_filter!(f::MultivariateFilter, Y::AbstractMatrix, mat::SysMat)
	# Get dims
	(n,T_len)= size(Y)
	p= length(mat.a1)
	
    # Initialize temp. containers
    tmp_pn= Matrix{Float64}(undef, (p,n))
    tmp_p= Matrix{Float64}(undef, (p,p))
	tmp_n= Matrix{Float64}(undef, (n,n))
	
	# Initialize filter
	f.a[:,1]= mat.a1
	f.P[:,:,1]= mat.P1
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
        # Store views 
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        F_t= view(f.F,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Forecast error
        error!(v_t, view(Y,:,t), mat.Z, a_t)
		
		# Forecast error variance
		error_var!(F_t, P_t, mat.Z, mat.H, tmp_pn)
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_n, F_t)
		fac= cholesky!(tmp_n)
		
		# Kalman gain
		gain!(K_t, P_t, fac, mat.Z, mat.T, tmp_pn)
		
		if t < T_len
			# Store views
	        a_p= view(f.a,:,t+1)
			P_p= view(f.P,:,:,t+1)
			
			# Predict states
			predict_state!(a_p, a_t, K_t, v_t, mat.T)
			
			# Predict states variance
			predict_state_var!(P_p, P_t, K_t, mat.Z, mat.T, mat.Q, tmp_p)
		end
	end
	
	return nothing
end

"""
	kalman_filter!(f, Y, mat) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variance ``P`` and precision ``F⁻¹`` and Kalman gain ``K`` for a linear Gaussian
State Space model with system matrices `mat` and data `Y` using the Kalman
filter based on Woodbury's Identity, storing the results in `f`.

Woodbury's Identity allows direct computation of the inverse variance
(precision) ``F⁻¹``.

#### Arguments
  - `Y::AbstractMatrix`	: data (n x T)
  - `mat::SysMat`		: State Space system matrices
  
#### Returns
  - `f::WoodburyFilter`	: Kalman filter output
"""
function kalman_filter!(f::WoodburyFilter, Y::AbstractMatrix, mat::SysMat)
	# Get dims
	(n,T_len)= size(Y)
	p= length(mat.a1)
	
    # Initialize temp. containers
	tmp_np= Matrix{Float64}(undef, (n,p))
    tmp_pn= Matrix{Float64}(undef, (p,n))
    tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Inverse of H
	Hi= copy(mat.H)
	if isa(mat.H, Diagonal)
		@. Hi.diag= inv(mat.H.diag)
	else
		LinearAlgebra.inv!(cholesky!(Hi))
	end
	
	# Initialize filter
	f.a[:,1]= mat.a1
	f.P[:,:,1]= mat.P1
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
        # Store views 
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        Fi_t= view(f.Fi,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Forecast error
        error!(v_t, view(Y,:,t), mat.Z, a_t)
		
		# Forecast error precision
		error_prec!(Fi_t, P_t, mat.Z, Hi, tmp_np, tmp_pn, tmp_p)
		
		# Kalman gain
		gain!(K_t, P_t, Fi_t, mat.Z, mat.T, tmp_pn)
		
		if t < T_len
			# Store views
	        a_p= view(f.a,:,t+1)
			P_p= view(f.P,:,:,t+1)
			
			# Predict states
			predict_state!(a_p, a_t, K_t, v_t, mat.T)
			
			# Predict states variance
			predict_state_var!(P_p, P_t, K_t, mat.Z, mat.T, mat.Q, tmp_p)
		end
	end
	
	return nothing
end