#=
smoother.jl

    Kalman smoother routines for a linear Gaussian State Space model

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2020/07/29
=#

# Struct
struct Smoother{Tα, TV, TL, Tr, TN} <: KalmanSmoother
	α::Tα	# smoothed state
	V::TV	# smoothed state covariances
	L::TL	# innovation error smoother transition matrix
	r::Tr	# backward smoothing recursion (state)
	N::TN	# backward smoothing recursion (variance)
end

"""
	computeL!(L_t, K_t, Z, T)
	
Compute the ``L`` matrix at time ``t``, storing the result in `L_t`.

#### Arguments
  - `K_t::AbstractMatrix`	: Kalman gain (p x n)
  - `Z::AbstractMatrix`		: system matrix ``Z`` (n x p)
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)
  
#### Returns
  - `L_t::AbstractMatrix`	: ``Lₜ`` (p x p)
"""
function computeL!(L_t::AbstractMatrix, K_t::AbstractMatrix, Z::AbstractMatrix, T::AbstractMatrix)
	# -KₜｘZ
	mul!(L_t, K_t, Z, -1., .0)
	# Lₜ = T - KₜｘZ
	L_t.+= T
	
	return nothing
end

function computeL!(L_t::AbstractMatrix, K_t::AbstractMatrix, Z::AbstractMatrix, T::Diagonal)
	# -KₜｘZ
	mul!(L_t, K_t, Z, -1., .0)
	# Lₜ = T - KₜｘZ
	@inbounds @fastmath for i in axes(T,1)
		L_t[i,i]+= T.diag[i]
	end
	
	return nothing
end

"""
	computeL_eq!(L_it, K_it, Z_i)
	
Compute the ``L`` matrix at time ``t`` for series ``i``, storing the result in
`L_it`.

#### Arguments
  - `K_it::AbstractVector`	: Kalman gain for series ``i``(p x 1)
  - `Z_i::AbstractVector`	: system vector ``Z`` for series ``i`` (p x 1)
  
#### Returns
  - `L_it::AbstractMatrix`	: ``Lₜᵢ`` (p x p)
"""
function computeL_eq!(L_it::AbstractMatrix, K_it::AbstractVector, Z_i::AbstractVector)
	# -KₜᵢｘZᵢ'
	L_it.= -K_it .* transpose(Z_i)
	# Lₜᵢ = I - KₜᵢｘZᵢ'
	@inbounds @fastmath for i in axes(L_it,1)
		L_it[i,i]+= 1.
	end
	
	return nothing
end

"""
	back_state!(r, L_t, fac, v_t, Z, tmp_n, tmp_p)
	
Compute the backward recursion for the smoothed state ``r`` at time ``t`` using
the inverse of ``F``, storing the result in `r`.

#### Arguments
  - `L_t::AbstractMatrix`	: ``Lₜ`` (p x p)
  - `Fi_t::AbstractMatrix`	: inverse of ``Fₜ`` (n x n)
  - `v_t::AbstractVector`	: forecast error (n x 1)
  - `Z::AbstractMatrix`		: system matrix ``Z`` (n x p)
  - `tmp_n::AbstractVector`	: tmp storage array (n x 1)
  - `tmp_p::AbstractVector`	: tmp storage array (p x 1)

#### Returns
  - `r::AbstractVector`	: Backward recursion for the smoothed state (p x 1)
"""
function back_state!(r::AbstractVector, L_t::AbstractMatrix, Fi_t::AbstractMatrix, 
					v_t::AbstractVector, Z::AbstractMatrix, tmp_n::AbstractVector, 
					tmp_p::AbstractVector)
	# Lₜ'ｘrₜ
	mul!(tmp_p, transpose(L_t), r)
	# Fₜ⁻¹ｘvₜ
	mul!(tmp_n, Fi_t, v_t)
	# Z'ｘFₜ⁻¹ｘvₜ
	mul!(r, transpose(Z), tmp_n)
	# rₜ₋₁ = Z'ｘFₜ⁻¹ｘvₜ + Lₜ'ｘrₜ
	r.+= tmp_p
	
	return nothing
end

"""
	back_state!(r, L_t, fac, v_t, Z, tmp_n, tmp_p)
	
Compute the backward recursion for the smoothed state ``r`` at time ``t`` using
the factorization of ``F``, storing the result in `r`.
"""
function back_state!(r::AbstractVector, L_t::AbstractMatrix, fac::Factorization, 
					v_t::AbstractVector, Z::AbstractMatrix, tmp_n::AbstractVector, 
					tmp_p::AbstractVector)
	# Lₜ'ｘrₜ
	mul!(tmp_p, transpose(L_t), r)
	# Fₜ⁻¹ｘvₜ
	ldiv!(tmp_n, fac, v_t)
	# Z'ｘFₜ⁻¹ｘvₜ
	mul!(r, transpose(Z), tmp_n)
	# rₜ₋₁ = Z'ｘFₜ⁻¹ｘvₜ + Lₜ'ｘrₜ
	r.+= tmp_p
	
	return nothing
end

"""
	back_state_eq!(r, L_i, F, v, Z_i, tmp_p)
	
Compute the backward recursion for the smoothed state ``r`` at time ``t`` for
series ``i-1`` using univariate treatment, storing the result in `r`.

#### Arguments
  - `L_i::AbstractMatrix`	: ``Lₜᵢ`` (p x p)
  - `F::Real`				: forecst error variance for series ``i``
  - `v::Real`				: forecast error for series ``i``
  - `Z_i::AbstractVector`	: system vector ``Z`` for series ``i`` (p x 1)
  - `tmp_p::AbstractVector`	: tmp storage array (p x 1)

#### Returns
  - `r::AbstractVector`	: Backward recursion for the smoothed state (p x 1)
"""
function back_state_eq!(r::AbstractVector, L_i::AbstractMatrix, F::Real, v::Real, 
						Z_i::AbstractVector, tmp_p::AbstractVector)
	# Lᵢ'ｘrᵢ
	mul!(tmp_p, transpose(L_i), r)
	# rᵢ₋₁ = Zᵢ'ｘF⁻¹ｘv + Lₜ'ｘrᵢ
	r.= inv(F) .* v .* Z_i .+ tmp_p
	
	return nothing
end

"""
	back_state_var!(N, L_t, Fi_t, Z, tmp_pn, tmp_p)
	
Compute the backward recursion for the smoothed state variance ``N`` at time
``t`` using the inverse of ``F``, storing the result in `N`.

#### Arguments
  - `L_t::AbstractMatrix`	: ``Lₜ`` (p x p)
  - `Fi_t::AbstractMatrix`	: inverse of ``Fₜ`` (n x n)
  - `Z::AbstractMatrix`		: system matrix ``Z`` (n x p)
  - `tmp_np::AbstractMatrix`: tmp storage array (n x p)
  - `tmp_pp::AbstractMatrix`: tmp storage array (p x p)

#### Returns
  - `N::AbstractMatrix`	: Backward recursion for the smoothed state variance (p x p)
"""
function back_state_var!(N::AbstractMatrix, L_t::AbstractMatrix, Fi_t::AbstractMatrix, 
						Z::AbstractMatrix, tmp_np::AbstractMatrix, tmp_pp::AbstractMatrix)
	# NₜｘLₜ
	mul!(tmp_pp, N, L_t)
	# LₜｘNₜｘLₜ
	mul!(N, transpose(L_t), tmp_pp)
	# Fₜ⁻¹ｘZ
	mul!(tmp_np, Fi_t, Z)
	# Nₜ₋₁ = Z'ｘFₜ⁻¹ｘZ + LₜｘNₜｘLₜ
	mul!(N, transpose(Z), tmp_np, 1., 1.)
	
	return nothing
end

"""
	back_state_var!(N, L_t, fac, Z, tmp_np, tmp_pp)
	
Compute the backward recursion for the smoothed state variance ``N`` at time
``t`` using the factorization of ``F``, storing the result in `N`.
"""
function back_state_var!(N::AbstractMatrix, L_t::AbstractMatrix, fac::Factorization, 
						Z::AbstractMatrix, tmp_np::AbstractMatrix, tmp_pp::AbstractMatrix)
	# NₜｘLₜ
	mul!(tmp_pp, N, L_t)
	# LₜｘNₜｘLₜ
	mul!(N, transpose(L_t), tmp_pp)
	# Fₜ⁻¹ｘZ
	ldiv!(tmp_np, fac, Z)
	# Nₜ₋₁ = Z'ｘFₜ⁻¹ｘZ + LₜｘNₜｘLₜ
	mul!(N, transpose(Z), tmp_np, 1., 1.)
	
	return nothing
end

"""
	back_state_var_eq!(N, L_i, F, Z_i, tmp_p)
	
Compute the backward recursion for the smoothed state variance ``N`` at time
``t`` for series ``i-1`` using the univariate treatment, storing the result in
`N`.

#### Arguments
  - `L_i::AbstractMatrix`	: ``Lₜᵢ`` (p x p)
  - `F::Real`				: forecst error variance for series ``i``
  - `Z_i::AbstractVector`	: system vector ``Z`` for series ``i`` (p x 1)
  - `tmp_pp::AbstractMatrix`: tmp storage array (p x p)

#### Returns
  - `N::AbstractMatrix`	: Backward recursion for the smoothed state variance (p x p)
"""
function back_state_var_eq!(N::AbstractMatrix, L_i::AbstractMatrix, F::Real, 
						Z_i::AbstractVector, tmp_pp::AbstractMatrix)
	# NᵢｘLᵢ
	mul!(tmp_pp, N, L_i)
	# LᵢｘNᵢｘLᵢ
	mul!(N, transpose(L_i), tmp_pp)
	# Nᵢ₋₁ = Zᵢ'ｘF⁻¹ｘZᵢ + LᵢｘNᵢｘLᵢ
	N.+= inv(F) .* Z_i .* transpose(Z_i)
	
	return nothing
end

"""
	transition!(X, T, tT, tmp)
	
Compute the linear time transition for smoother recursion objects from time
``t`` to ``t-1`` using the equation-by-equation or univariate version of the
Kalman smoother. Storing the results in `X`.

#### Arguments
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)
  - `tT::AbstractChar`		: transpose indicator
  - `tmp::AbstractVecOrMat`	: tmp storage array (p x 1) or (p x p)

#### Returns
  - `X::AbstractVecOrMat`	: smoother object (p x 1) or (p x p)
"""
function transition!(X::AbstractVecOrMat, T::AbstractMatrix, tT::AbstractChar, tmp::AbstractVecOrMat)
	# Store
	copyto!(tmp, X)
	
	if tT == 'T'
		# Xₜ₋₁ = T'ｘXₜ
		mul!(X, transpose(T), tmp)
	elseif tT == 'N'
		# Xₜ₋₁ = TｘXₜ
		mul!(X, T, tmp)
	end
	
	return nothing
end

function transition!(X::AbstractVecOrMat, T::Diagonal, tT::AbstractChar, tmp::AbstractVecOrMat)	
	# Xₜ₋₁ = TｘXₜ
	X.= T.diag .* X
	
	return nothing
end

"""
	transition2!(X, T, tmp)
	
Compute the quadratic time transition for smoother recursion objects from time
``t`` to ``t-1`` using the equation-by-equation or univariate version of the
Kalman smoother. Storing the results in `X`.

#### Arguments
  - `T::AbstractMatrix`		: system matrix ``T`` (p x p)
  - `tmp::AbstractMatrix`	: tmp storage array (p x p)

#### Returns
  - `X::AbstractMatrix`		: smoother object (p x p)
"""
function transition2!(X::AbstractMatrix, T::AbstractMatrix, tmp::AbstractMatrix)	
	# T'ｘXₜ
	mul!(tmp, transpose(T), X)
	# Xₜ₋₁ = T'ｘXₜｘT
	mul!(X, tmp, T)
	
	return nothing
end

function transition2!(X::AbstractMatrix, T::Diagonal, tmp::AbstractMatrix)	
	# Xₜ₋₁ = T'ｘXₜｘT
	X.= T.diag .* X .* transpose(T.diag)
	
	return nothing
end

"""
	smooth_state!(α_t, a_t, P_t, r)
	
Compute the smoothed state ``α`` at time ``t``, storing the result in `α_t`.

#### Arguments
  - `a_t::AbstractVector`	: filtered state at time ``t`` (p x 1)
  - `P_t::AbstractMatrix`	: filtered state variance at time ``t`` (p x p)
  - `r::AbstractVector`		: Backward recursion for states (p x 1)

#### Returns
  - `α_t::AbstractVector`	: smoothed state at time ``t`` (p x 1)
"""
function smooth_state!(α_t::AbstractVector, a_t::AbstractVector, P_t::AbstractMatrix, r::AbstractVector)
	# Pₜｘrₜ₋₁
	mul!(α_t, P_t, r)
	# αₜ = aₜ + Pₜｘrₜ₋₁
	α_t.+= a_t
	
	return nothing
end

"""
	smooth_state_var!(V_t, P_t, N, tmp)
	
Compute the smoothed state variance ``V`` at time ``t``, storing the result in
`V_t`.

#### Arguments
  - `P_t::AbstractMatrix`	: filtered state variance at time ``t`` (p x p)
  - `N::AbstractMatrix`		: Backward recursion for states variance (p x p)
  - `tmp::AbstractMatrix`	: tmp storage array (p x p)

#### Returns
  - `V_t::AbstractMatrix`	: smoothed state variance at time ``t`` (p x p)
"""
function smooth_state_var!(V_t::AbstractMatrix, P_t::AbstractMatrix, N::AbstractMatrix, tmp::AbstractMatrix)
	# Nₜ₋₁ｘPₜ
	mul!(tmp, N, P_t)
	# -PₜｘNₜ₋₁ｘPₜ
	mul!(V_t, P_t, tmp, -1., .0)
	# Vₜ = Pₜ - PₜｘNₜ₋₁ｘPₜ
	V_t.+= P_t 
	
	return nothing
end

"""
	smooth_state_cov!(V, N, L, P_t, t, h, tmp_0, tmp_1)
	
Compute the smoothed state variance and autocovariances ``V(j)`` at lags 0 to
``h`` at time ``t``, storing the result in `V`.

#### Arguments
  - `N::AbstractMatrix`		: Backward recursion for states variance (p x p)
  - `L::AbstractMatrix`		: ``L`` (p x p)
  - `P_t::AbstractMatrix`	: filtered state variance at time ``t`` (p x p)
  - `t::Integer`			: time index
  - `h::Integer`			: lag length
  - `tmp_0::AbstractMatrix`	: tmp storage array (p x p)
  - `tmp_1::AbstractMatrix`	: tmp storage array (p x p)

#### Returns
  - `V::AbstractArray`	: smoothed state autocovariances at time ``t`` (p x p x h+1 x T)
"""
function smooth_state_cov!(V::AbstractArray, N::AbstractMatrix, L::AbstractMatrix, 
							P_t::AbstractMatrix, t::Integer, h::Integer, 
							tmp_0::AbstractMatrix, tmp_1::AbstractMatrix)
	# Get dims
	T_len= size(V, 4)
	
	# -PₜｘNₜ₋₁
	mul!(tmp_0, P_t, N, -1., .0)
	# I - PₜｘNₜ₋₁
	@inbounds @fastmath for i in axes(P_t,1)
		tmp_0[i,i]+= 1.
	end
	
	# Loop through lags
	@inbounds @fastmath for j in 0:h
		# Store
		V[:,:,j+1,t].= tmp_0
		
		for s in t+1:min(t+j,T_len)
			# Store view
			V_js= view(V,:,:,j+1,s)
			
			# Propagate backwards with Lₜ₋₁
			mul!(tmp_1, V_js, L)
			V_js.= tmp_1
		end
		
		# Complete autocovariance
		if t+j <= T_len
			# Store view
			V_jt= view(V,:,:,j+1,t+j)
			
			# Vₜ(j) = (I - Pₜ₊ⱼｘNₜ₊ⱼ₋₁)ｘLₜ₊ⱼ₋₁ｘ...ｘLₜ₊₁ｘLₜｘPₜ
			mul!(tmp_1, V_jt, P_t)
			V_jt.= tmp_1
		end
	end
	
	return nothing
end

"""
	kalman_smoother!(smoother, filter, sys)
	
Compute smoothed states ``α`` and corresponding variances ``V`` for a linear
Gaussian State Space model with system matrices `sys` and Kalman filter output
`filter`, storing the result in `smoother`.

#### Arguments
  - `filter::KalmanFilter`	: Kalman filter output
  - `sys::StateSpaceSystem`	: state space system matrices

#### Returns
  - `smoother::Smoother`	: Kalman smoother output
"""
function kalman_smoother!(smoother::Smoother, filter::MultivariateFilter, sys::LinearTimeInvariant)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)

	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(smoother.α,:,t)
		V_t= view(smoother.V,:,:,t) 
        a_t= view(filter.a,:,t)
		P_t= view(filter.P,:,:,t)
        v_t= view(filter.v,:,t)
        F_t= view(filter.F,:,:,t)
        K_t= view(filter.K,:,:,t)
		
		# Lₜ
		computeL!(smoother.L, K_t, sys.Z, sys.T)
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		fac= cholesky!(Hermitian(tmp_nn))
		
		# Backward recursion state
		back_state!(smoother.r, smoother.L, fac, v_t, sys.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(smoother.N, smoother.L, fac, sys.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, smoother.N, tmp_pp)
	end
	
	return nothing
end

function kalman_smoother!(smoother::Smoother, filter::MultivariateFilter, sys::LinearTimeVariant)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)

	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(smoother.α,:,t)
		V_t= view(smoother.V,:,:,t) 
        a_t= view(filter.a,:,t)
		P_t= view(filter.P,:,:,t)
        v_t= view(filter.v,:,t)
        F_t= view(filter.F,:,:,t)
        K_t= view(filter.K,:,:,t)
		
		# Lₜ
		computeL!(smoother.L, K_t, sys.Z[t], sys.T[t])
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		fac= cholesky!(Hermitian(tmp_nn))
		
		# Backward recursion state
		back_state!(smoother.r, smoother.L, fac, v_t, sys.Z[t], tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(smoother.N, smoother.L, fac, sys.Z[t], tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, smoother.N, tmp_pp)
	end
	
	return nothing
end

function kalman_smoother!(smoother::Smoother, filter::WoodburyFilter, sys::LinearTimeInvariant)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)

	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(smoother.α,:,t)
		V_t= view(smoother.V,:,:,t) 
        a_t= view(filter.a,:,t)
		P_t= view(filter.P,:,:,t)
        v_t= view(filter.v,:,t)
        Fi_t= view(filter.Fi,:,:,t)
        K_t= view(filter.K,:,:,t)
		
		# Lₜ
		computeL!(smoother.L, K_t, sys.Z, sys.T)
		
		# Backward recursion state
		back_state!(smoother.r, smoother.L, Fi_t, v_t, sys.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(smoother.N, smoother.L, Fi_t, sys.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, smoother.N, tmp_pp)
	end
	
	return nothing
end

function kalman_smoother!(smoother::Smoother, filter::WoodburyFilter, sys::LinearTimeVariant)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)

	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(smoother.α,:,t)
		V_t= view(smoother.V,:,:,t) 
        a_t= view(filter.a,:,t)
		P_t= view(filter.P,:,:,t)
        v_t= view(filter.v,:,t)
        Fi_t= view(filter.Fi,:,:,t)
        K_t= view(filter.K,:,:,t)
		
		# Lₜ
		computeL!(smoother.L, K_t, sys.Z[t], sys.T[t])
		
		# Backward recursion state
		back_state!(smoother.r, smoother.L, Fi_t, v_t, sys.Z[t], tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(smoother.N, smoother.L, Fi_t, sys.Z[t], tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, smoother.N, tmp_pp)
	end
	
	return nothing
end

function kalman_smoother!(smoother::Smoother, filter::UnivariateFilter, sys::LinearTimeInvariant)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_p= Vector{Float64}(undef, p)
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)

	# Tranpose
	Zt= transpose(sys.Z)
	
	# Smoother
	@inbounds @fastmath for	t in T_len:-1:1
		# Store views
		α_t= view(smoother.α,:,t)
		V_t= view(smoother.V,:,:,t) 
        a_t= view(filter.a,:,1,t)
		P_t= view(filter.P,:,:,1,t)
		
		# Time transition
		transition!(smoother.r, sys.T, 'T', tmp_p)
		transition2!(smoother.N, sys.T, tmp_pp)
		
		for i in n:-1:1
			# Store views
	        K_it= view(filter.K,:,i,t)
			Z_i= view(Zt,:,i)
			
			# Lₜᵢ
			computeL_eq!(smoother.L, K_it, Z_i)
			
			# Backward recursion state
			back_state_eq!(smoother.r, smoother.L, filter.F[i,t], filter.v[i,t], Z_i, tmp_p)
		
			# Backward recursion state variance
			back_state_var_eq!(smoother.N, smoother.L, filter.F[i,t], Z_i, tmp_pp)
		end
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, smoother.N, tmp_pp)
	end
	
	return nothing
end

function kalman_smoother!(smoother::Smoother, filter::UnivariateFilter, sys::LinearTimeVariant)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_p= Vector{Float64}(undef, p)
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)
	
	# Smoother
	@inbounds @fastmath for	t in T_len:-1:1
		# store
		Z_t= sys.Z[t]

		# Store views
		α_t= view(smoother.α,:,t)
		V_t= view(smoother.V,:,:,t) 
        a_t= view(filter.a,:,1,t)
		P_t= view(filter.P,:,:,1,t)
		
		# Time transition
		transition!(smoother.r, sys.T[t], 'T', tmp_p)
		transition2!(smoother.N, sys.T[t], tmp_pp)
		
		for i in n:-1:1
			# Store views
	        K_it= view(filter.K,:,i,t)
			Z_it= view(Z_t,i,:)
			
			# Lₜᵢ
			computeL_eq!(smoother.L, K_it, Z_it)
			
			# Backward recursion state
			back_state_eq!(smoother.r, smoother.L, filter.F[i,t], filter.v[i,t], Z_it, tmp_p)
		
			# Backward recursion state variance
			back_state_var_eq!(smoother.N, smoother.L, filter.F[i,t], Z_it, tmp_pp)
		end
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, smoother.N, tmp_pp)
	end
	
	return nothing
end

"""
	kalman_smoother_cov!(smoother, filter, sys; h=1)
	
Compute smoothed states ``α`` and corresponding variances ``V`` and
autocovariances ``V(h)`` up until lag `h` for a linear Gaussian State Space
model with system matrices `sys` and Kalman filter output `filter`, storing the
results in `smoother`.

#### Arguments
  - `filter::KalmanFilter`	: Kalman filter output
  - `sys::StateSpaceSystem`	: state space system matrices
  - `h::Integer`			: lag length

#### Returns
  - `smoother::Smoother`	: Kalman smoother output
"""
function kalman_smoother_cov!(smoother::Smoother, filter::MultivariateFilter,
								sys::LinearTimeInvariant; h::Integer=1)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))

	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(smoother.α,:,t)
        a_t= view(filter.a,:,t)
		P_t= view(filter.P,:,:,t)
        v_t= view(filter.v,:,t)
        F_t= view(filter.F,:,:,t)
        K_t= view(filter.K,:,:,t)
		
		# Lₜ
		computeL!(smoother.L, K_t, sys.Z, sys.T)
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		fac= cholesky!(Hermitian(tmp_nn))
		
		# Backward recursion state
		back_state!(smoother.r, smoother.L, fac, v_t, sys.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(smoother.N, smoother.L, fac, sys.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(smoother.V, smoother.N, smoother.L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return nothing
end

function kalman_smoother_cov!(smoother::Smoother, filter::MultivariateFilter,
								sys::LinearTimeVariant; h::Integer=1)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))

	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(smoother.α,:,t)
        a_t= view(filter.a,:,t)
		P_t= view(filter.P,:,:,t)
        v_t= view(filter.v,:,t)
        F_t= view(filter.F,:,:,t)
        K_t= view(filter.K,:,:,t)
		
		# Lₜ
		computeL!(smoother.L, K_t, sys.Z[t], sys.T[t])
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		fac= cholesky!(Hermitian(tmp_nn))
		
		# Backward recursion state
		back_state!(smoother.r, smoother.L, fac, v_t, sys.Z[t], tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(smoother.N, smoother.L, fac, sys.Z[t], tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(smoother.V, smoother.N, smoother.L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return nothing
end

function kalman_smoother_cov!(smoother::Smoother, filter::WoodburyFilter, 
								sys::LinearTimeInvariant; h::Integer=1)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(smoother.α,:,t)
        a_t= view(filter.a,:,t)
		P_t= view(filter.P,:,:,t)
        v_t= view(filter.v,:,t)
        Fi_t= view(filter.Fi,:,:,t)
        K_t= view(filter.K,:,:,t)
		
		# Lₜ
		computeL!(smoother.L, K_t, sys.Z, sys.T)
		
		# Backward recursion state
		back_state!(smoother.r, smoother.L, Fi_t, v_t, sys.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(smoother.N, smoother.L, Fi_t, sys.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(smoother.V, smoother.N, smoother.L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return nothing
end

function kalman_smoother_cov!(smoother::Smoother, filter::WoodburyFilter, 
								sys::LinearTimeVariant; h::Integer=1)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(smoother.α,:,t)
        a_t= view(filter.a,:,t)
		P_t= view(filter.P,:,:,t)
        v_t= view(filter.v,:,t)
        Fi_t= view(filter.Fi,:,:,t)
        K_t= view(filter.K,:,:,t)
		
		# Lₜ
		computeL!(smoother.L, K_t, sys.Z[t], sys.T[t])
		
		# Backward recursion state
		back_state!(smoother.r, smoother.L, Fi_t, v_t, sys.Z[t], tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(smoother.N, smoother.L, Fi_t, sys.Z[t], tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(smoother.V, smoother.N, smoother.L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return nothing
end

function kalman_smoother_cov!(smoother::Smoother, filter::UnivariateFilter, 
								sys::LinearTimeInvariant; h::Integer=1)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
	L_i= Matrix{Float64}(undef, (p,p))
    tmp_p= Vector{Float64}(undef, p)
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)

	# Smoother
	@inbounds @fastmath for	t in T_len:-1:1
		# Store views
		α_t= view(smoother.α,:,t)
        a_t= view(filter.a,:,1,t)
		P_t= view(filter.P,:,:,1,t)

		# reset
		smoother.L.= zero(Float64)
		for i in 1:p
			smoother.L[i,i]= one(Float64)
		end
		
		# Time transitions
		transition!(smoother.r, sys.T, 'T', tmp_p)
		transition2!(smoother.N, sys.T, tmp_pp)
		
		for i in n:-1:1
			# Store views
	        K_it= view(filter.K,:,i,t)
			Z_i= view(Z,i,:)
			
			# Lₜᵢ
			computeL_eq!(L_i, K_it, Z_i)
			
			# Lₜᵢｘ...ｘLₜ₁
			copyto!(tmp_pp, smoother.L)
			mul!(smoother.L, tmp_pp, L_i)
			
			# Backward recursion state
			back_state_eq!(smoother.r, L_i, filter.F[i,t], filter.v[i,t], Z_i, tmp_p)
		
			# Backward recursion state variance
			back_state_var_eq!(smoother.N, L_i, filter.F[i,t], Z_i, tmp_pp)
		end
		
		# Lₜ = TｘLₜₙｘ...ｘLₜ₁
		transition!(smoother.L, sys.T, 'N', tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(smoother.V, smoother.N, smoother.L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return nothing
end

function kalman_smoother_cov!(smoother::Smoother, filter::UnivariateFilter, 
								sys::LinearTimeVariant; h::Integer=1)
	# Get dims
	(p,n,T_len)= size(filter.K)
	
    # Initialize temp. containers
	L_i= Matrix{Float64}(undef, (p,p))
    tmp_p= Vector{Float64}(undef, p)
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	smoother.r.= zero(Float64)
	smoother.N.= zero(Float64)
	
	# Smoother
	@inbounds @fastmath for	t in T_len:-1:1
		# store
		Z_t= sys.Z[t]

		# Store views
		α_t= view(smoother.α,:,t)
        a_t= view(filter.a,:,1,t)
		P_t= view(filter.P,:,:,1,t)

		# reset
		smoother.L.= zero(Float64)
		for i in 1:p
			smoother.L[i,i]= one(Float64)
		end
		
		# Time transitions
		transition!(smoother.r, sys.T[t], 'T', tmp_p)
		transition2!(smoother.N, sys.T[t], tmp_pp)
		
		for i in n:-1:1
			# Store views
	        K_it= view(filter.K,:,i,t)
			Z_it= view(Z_t,i,:)
			
			# Lₜᵢ
			computeL_eq!(L_i, K_it, Z_it)
			
			# Lₜᵢｘ...ｘLₜ₁
			copyto!(tmp_pp, smoother.L)
			mul!(smoother.L, tmp_pp, L_i)
			
			# Backward recursion state
			back_state_eq!(smoother.r, L_i, filter.F[i,t], filter.v[i,t], Z_it, tmp_p)
		
			# Backward recursion state variance
			back_state_var_eq!(smoother.N, L_i, filter.F[i,t], Z_it, tmp_pp)
		end
		
		# Lₜ = TｘLₜₙｘ...ｘLₜ₁
		transition!(smoother.L, sys.T[t], 'N', tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, smoother.r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(smoother.V, smoother.N, smoother.L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return nothing
end