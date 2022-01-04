#=
filter.jl

    Kalman filtering routines for a linear Gaussian State Space model

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2020/07/28
=#

# Include numerical helper routines
include("../misc/num.jl")

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
    BLAS.gemm!('N', 'T', 1., P_t, Z, .0, tmp)
    # ZｘPₜｘZ'
	mul!(F_t, Z, tmp)
	# Fₜ = ZｘPₜｘZ' + H
	@. F_t+= H
	
	return nothing
end

function error_var!(F_t::AbstractMatrix, P_t::AbstractMatrix, Z::AbstractMatrix, 
					H::Diagonal, tmp::AbstractMatrix)	
    # PₜｘZ'
    BLAS.gemm!('N', 'T', 1., P_t, Z, .0, tmp)
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
	inInv!(Pi, P_t)
	# H⁻¹×Z
	mul!(tmp_np, Hi, Z)
	# Pₜ⁻¹ + Z'×H⁻¹×Z
	BLAS.gemm!('T', 'N', 1., Z, tmp_np, 1., Pi)
	# (Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹
	inInv!(tmp_p, Pi)
	# (Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹×Z′×H⁻¹
	BLAS.gemm!('N', 'T', 1., tmp_p, HiZ, .0, tmp_pn)
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
	inInv!(Pi, P_t)
	# H⁻¹×Z
	mul!(tmp_np, Hi, Z)
	# Pₜ⁻¹ + Z'×H⁻¹×Z
	BLAS.gemm!('T', 'N', 1., Z, tmp_np, 1., Pi)
	# (Pₜ⁻¹ + Z'×H⁻¹×Z)⁻¹
	inInv!(tmp_p, Pi)
	# (Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹×Z′×H⁻¹
	BLAS.gemm!('N', 'T', 1., tmp_p, HiZ, .0, tmp_pn)
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
    BLAS.gemm!('N', 'T', 1., P_t, Z, .0, K_t)
    # PₜｘZ'ｘFₜ⁻¹
	mul!(tmp, K_t, Fi_t)
	# Kₜ = TｘPₜｘZ'ｘFₜ⁻¹
	mul!(K_t, T, tmp)
	
	return nothing
end

"""
	gain!(K_t, P_t, fac::Factorization, Z, T, tmp)
	
Compute Kalman gain ``K`` at time ``t`` using the factorization of ``F``, storing
the result in `K_t`.
"""
function gain!(K_t::AbstractMatrix, P_t::AbstractMatrix, fac::Factorization, 
				Z::AbstractMatrix, T::AbstractMatrix, tmp::AbstractMatrix)
    # PₜｘZ'
    BLAS.gemm!('N', 'T', 1., P_t, Z, .0, tmp)
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
	BLAS.gemm!('N', 'T', 1., P_t, P_p, .0, tmp)
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
	BLAS.gemm!('N', 'T', 1., P_t, P_p, .0, tmp)
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
	BLAS.gemm!('N', 'T', 1., P_t, P_p, .0, tmp)
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
	BLAS.gemm!('N', 'T', 1., P_t, P_p, .0, tmp)
	# TｘPₜｘ(T - KₜｘZ)'
	mul!(P_p, T, tmp)
	# TｘPₜｘ(T - KₜｘZ)' + Q
	@inbounds @fastmath for i in axes(Q,1)
		P_p[i,i]+= Q.diag[i]
	end
	
	return nothing
end

"""
	forward_eq!(a_f, P_f, a_i, P_i, K_i, v, F)
	
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
function forward_eq!(a_f::AbstractVector, P_f::AbstractMatrix, a_i::AbstractVector, 
						P_i::AbstractMatrix, K_i::AbstractVector, v::Real, F::Real)
	# aᵢ₊₁ = aᵢ + vｘKᵢ
	@. a_f= a_i + v*K_i
	# Pᵢ₊₁ = Pᵢ - KᵢｘKᵢ'/F
	@. P_f= P_i - inv(F)*K_i*transpose(K_i)
	
	return nothing
end

"""
	predict_eq!(a_p, P_p, a_t, P_t, T, Q, tmp)
	
Predict states ``a`` and corresponding variance ``P`` at time ``t+1`` using
univariate treatment, storing the result in `a_p` and `P_p`.

#### Arguments
  - `a_t::AbstractVector`	: predicted state at time ``t`` (p x 1)
  - `P_t::AbstractMatrix`	: predicted states variance at time ``t`` (p x p)
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)
  - `Q::AbstractMatrix`		: system matrix ``Q`` (p x p)
  - `tmp::AbstractMatrix`	: tmp storage array (p x p)

#### Returns
  - `a_p::AbstractVector`	: predicted state at time ``t+1`` (p x 1)
  - `P_p::AbstractMatrix`	: predicted states variance at time ``t+1`` (p x p)
"""
function predict_eq!(a_p::AbstractVector, P_p::AbstractMatrix, a_t::AbstractVector, 
						P_t::AbstractMatrix, T::AbstractMatrix, Q::AbstractMatrix, 
						tmp::AbstractMatrix)
	# Predict states
	mul!(a_p, T, a_t)

	# Predict states variance
	# PₜｘT'
	BLAS.gemm!('N', 'T', 1., P_t, T, .0, tmp)
	# TｘPₜｘT'
	mul!(P_p, T, tmp)
	# Pₜ₊₁ = TｘPₜｘT' + Q
	@. P_p+= Q
	
	return nothing
end

function predict_eq!(a_p::AbstractVector, P_p::AbstractMatrix, a_t::AbstractVector, 
						P_t::AbstractMatrix, T::AbstractMatrix, Q::Diagonal, 
						tmp::AbstractMatrix)
	# Predict states
	mul!(a_p, T, a_t)

	# Predict states variance
	# PₜｘT'
	BLAS.gemm!('N', 'T', 1., P_t, T, .0, tmp)
	# TｘPₜｘT'
	mul!(P_p, T, tmp)
	# Pₜ₊₁ = TｘPₜｘT' + Q
	@inbounds @fastmath for i in axes(Q,1)
		P_p[i,i]+= Q.diag[i]
	end
	
	return nothing
end

function predict_eq!(a_p::AbstractVector, P_p::AbstractMatrix, a_t::AbstractVector, 
						P_t::AbstractMatrix, T::Diagonal, Q::AbstractMatrix, 
						tmp::AbstractMatrix)
	# Predict states
	@. a_p= T.diag*a_t

	# Predict states variance
	@. P_p= T.diag*P_t*transpose(T.diag) + mat.Q
	
	return nothing
end

"""
	kalmanfilter(Y, mat) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variances ``P`` and ``F`` and Kalman gain ``K`` for a linear Gaussian State
Space model with system matrices `mat` and data `Y` using the Kalman filter.

#### Arguments
  - `Y::AbstractMatrix`	: data (n x T)
  - `mat::SysMat`		: State Space system matrices
  
#### Returns
  - `f::Filter`	: Kalman filter output
"""
function kalmanfilter(Y::AbstractMatrix, mat::SysMat)
	# Get dims
	(n,T_len)= size(Y)
	p= length(mat.a1)
	
    # Initialize temp. containers
    tmp_pn= Matrix{Float64}(undef, (p,n))
    tmp_p= Matrix{Float64}(undef, (p,p))
	tmp_n= Matrix{Float64}(undef, (n,n))
	
	# Initaliaze filter res. cont.
	a= Matrix{Float64}(undef, (p,T_len))
	P= Array{Float64,3}(undef, (p,p,T_len))
	v= Matrix{Float64}(undef, (n,T_len))
	F= Array{Float64,3}(undef, (n,n,T_len))
	K= Array{Float64,3}(undef, (p,n,T_len))
	f= Filter(a,P,v,F,K)
	
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
	
	return f
end

"""
	kalmanfilter!(f, Y, mat) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variances ``P`` and ``F`` and Kalman gain ``K``, storing the results in `f`. See
also `kalmanfilter`.
"""
function kalmanfilter!(f::Filter, Y::AbstractMatrix, mat::SysMat)
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
	kalmanfilter_wb(Y, mat) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variance ``P`` and precision ``F⁻¹`` and Kalman gain ``K`` for a linear Gaussian
State Space model with system matrices `mat` and data `Y` using the Kalman
filter based on Woodbury's Identity.

Woodbury's Identity allows direct computation of the inverse variance
(precision) ``F⁻¹``.

#### Arguments
  - `Y::AbstractMatrix`	: data (n x T)
  - `mat::SysMat`		: State Space system matrices
  
#### Returns
  - `f::FilterWb`	: Kalman filter output
"""
function kalmanfilter_wb(Y::AbstractMatrix, mat::SysMat)
	# Get dims
	(n,T_len)= size(Y)
	p= length(mat.a1)
	
    # Initialize temp. containers
	tmp_np= Matrix{Float64}(undef, (n,p))
    tmp_pn= Matrix{Float64}(undef, (p,n))
    tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Initaliaze filter res. cont.
	a= Matrix{Float64}(undef, (p,T_len))
	P= Array{Float64,3}(undef, (p,p,T_len))
	v= Matrix{Float64}(undef, (n,T_len))
	Fi= Array{Float64,3}(undef, (n,n,T_len))
	K= Array{Float64,3}(undef, (p,n,T_len))
	f= FilterWb(a,P,v,Fi,K)
	
	# Inverse of H
	Hi= similar(mat.H)
	if isa(mat.H, Diagonal)
		@. Hi.diag= inv(mat.H.diag)
	else
		inInv!(Hi, mat.H)
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
	
	return f
end

"""
	kalmanfilter_wb!(f, Y, mat) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variance ``P`` and precision ``F⁻¹`` and Kalman gain ``K``, storing the results
in `f`. See also `kalmanfilter_wb`.
"""
function kalmanfilter_wb!(f::FilterWb, Y::AbstractMatrix, mat::SysMat)
	# Get dims
	(n,T_len)= size(Y)
	p= length(mat.a1)
	
    # Initialize temp. containers
	tmp_np= Matrix{Float64}(undef, (n,p))
    tmp_pn= Matrix{Float64}(undef, (p,n))
    tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Inverse of H
	Hi= similar(mat.H)
	if isa(mat.H, Diagonal)
		@. Hi.diag= inv(mat.H.diag)
	else
		inInv!(Hi, mat.H)
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

"""
	kalmanfilter_eq(Y, mat) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variances ``P`` and ``F`` and Kalman gain ``K`` for a linear Gaussian State
Space model with system matrices `mat` and data `Y` using the
equation-by-equation or univariate version of the Kalman filter.

#### Arguments
  - `Y::AbstractMatrix`	: data (n x T)
  - `mat::SysMat`		: State Space system matrices
  
#### Returns
  - `f::Filter`	: Kalman filter output
"""
function kalmanfilter_eq(Y::AbstractMatrix, mat::SysMat)
	# Get dims
	(n,T_len)= size(Y)
	p= length(mat.a1)
	
    # Initialize temp. containers
    tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Initaliaze filter res. cont.
	a= Array{Float64,3}(undef, (p,n+1,T_len))
	P= Array{Float64,4}(undef, (p,p,n+1,T_len))
	v= Matrix{Float64}(undef, (n,T_len))
	F= Matrix{Float64}(undef, (n,T_len))
	K= Array{Float64,3}(undef, (p,n,T_len))
	f= Filter(a,P,v,F,K)
	
	# Initialize filter
	f.a[:,1,1]= mat.a1
	f.P[:,:,1,1]= mat.P1
	
	# Tranpose
	Zt= transpose(mat.Z)
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
		for i in 1:n
	        # Store views 
	        a_it= view(f.a,:,i,t)
			a_f= view(f.a,:,i+1,t)
			P_it= view(f.P,:,:,i,t)
			P_f= view(f.P,:,:,i+1,t)
	        K_it= view(f.K,:,i,t)
			Z_i= view(Zt,:,i)
		
			# Forecast error
			f.v[i,t]= Y[i,t] - dot(Z_i, a_it)
		
			# Forecast error variance
			f.F[i,t]= dot(Z_i, P_it, Z_i) + H.diag[i]
		
			# Kalman gain
			mul!(K_it, P_it, Z_i, inv(f.F[i,t]), .0)
		
			# Move states and variances forward
			forward_eq!(a_f, P_f, a_it, P_it, K_it, f.v[i,t], f.F[i,t])
		end
		
		if t < T_len
			# Store views
			a_f= view(f.a,:,n+1,t)
			P_f= view(f.P,:,:,n+1,t)
	        a_p= view(f.a,:,1,t+1)
			P_p= view(f.P,:,:,1,t+1)
			
			# Predict states and variances
			predict_eq!(a_p, P_p, a_f, P_f, mat.T, mat,Q, tmp_p)
		end
	end
	
	return f
end

"""
	kalmanfilter_eq!(f, Y, mat) 
	
Compute predicted states ``a`` and forecast errors ``v`` with corresponding
variances ``P`` and ``F`` and Kalman gain ``K``, storing the results in `f`. See
also `kalmanfilter_eq`.
"""
function kalmanfilter_eq!(f::Filter, Y::AbstractMatrix, mat::SysMat)
	# Get dims
	(n,T_len)= size(Y)
	p= length(mat.a1)
	
    # Initialize temp. containers
    tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Initialize filter
	f.a[:,1,1]= mat.a1
	f.P[:,:,1,1]= mat.P1
	
	# Tranpose
	Zt= transpose(mat.Z)
	
	# Filter
	@inbounds @fastmath for t in 1:T_len
		for i in 1:n
	        # Store views 
	        a_it= view(f.a,:,i,t)
			a_f= view(f.a,:,i+1,t)
			P_it= view(f.P,:,:,i,t)
			P_f= view(f.P,:,:,i+1,t)
	        K_it= view(f.K,:,i,t)
			Z_i= view(Zt,:,i)
		
			# Forecast error
			f.v[i,t]= Y[i,t] - dot(Z_i, a_it)
		
			# Forecast error variance
			f.F[i,t]= dot(Z_i, P_it, Z_i) + H.diag[i]
		
			# Kalman gain
			mul!(K_it, P_it, Z_i, inv(f.F[i,t]), .0)
		
			# Move states and variances forward
			forward_eq!(a_f, P_f, a_it, P_it, K_it, f.v[i,t], f.F[i,t])
		end
		
		if t < T_len
			# Store views
			a_f= view(f.a,:,n+1,t)
			P_f= view(f.P,:,:,n+1,t)
	        a_p= view(f.a,:,1,t+1)
			P_p= view(f.P,:,:,1,t+1)
			
			# Predict states and variances
			predict_eq!(a_p, P_p, a_f, P_f, mat.T, mat,Q, tmp_p)
		end
	end
	
	return nothing
end