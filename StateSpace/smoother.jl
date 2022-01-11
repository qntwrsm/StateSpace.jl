#=
smoother.jl

    Kalman smoother routines for a linear Gaussian State Space model

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2020/07/29
=#

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
	mul!(L_t, K_t, mat.Z, -1., .0)
	# Lₜ = T - KₜｘZ
	@. L_t+= mat.T
	
	return nothing
end

function computeL!(L_t::AbstractMatrix, K_t::AbstractMatrix, Z::AbstractMatrix, T::Diagonal)
	# -KₜｘZ
	mul!(L_t, K_t, mat.Z, -1., .0)
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
	@. L_it= -K_it*transpose(Z_i)
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
	@. r+= tmp_p
	
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
	@. r+= tmp_p
	
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
	mul!(tmp_p, transpose(L_t), r)
	# rᵢ₋₁ = Zᵢ'ｘF⁻¹ｘv + Lₜ'ｘrᵢ
	@. r= inv(F)*v*Z_i + tmp_p
	
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
	back_state_var!(N, L_t, fac, Z)
	
Compute the backward recursion for the smoothed state variance ``N`` at time
``t`` using the factorization of ``F``, storing the result in `N`.
"""
function back_state_var!(N::AbstractMatrix, L_t::AbstractMatrix, fac::Factorization, 
						Z::AbstractMatrix, tmp_np::AbstractMatrix, tmp_pp::AbstractMatrix)
	# NₜｘLₜ
	mul!(tmp_p, N, L_t)
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
	@. N+= inv(F)*Z_i*transpose(Z_i)
	
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
	@. X= T.diag*X
	
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
	@. X= T.diag*X*transpose(T.diag)
	
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
	@. α_t+= a_t
	
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
	mul!(V_t, P_t, N, -1., .0)
	# Vₜ = Pₜ - PₜｘNₜ₋₁ｘPₜ
	@. V_t+= P_t 
	
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
	for j in 0:h
		# Store
		V[:,:,j+1,t]= tmp_0
		
		for s in t+1:min(t+j,T_len)
			# Store view
			V_js= view(V,:,:,j+1,s)
			
			# Propagate backwards with Lₜ₋₁
			mul!(tmp_1, V_js, L)
			V_js= tmp_1
		end
		
		# Complete autocovariance
		if t+j <= T_len
			# Store view
			V_jt= view(V,:,:,j+1,t+j)
			
			# Vₜ(j) = (I - Pₜ₊ⱼｘNₜ₊ⱼ₋₁)ｘLₜ₊ⱼ₋₁ｘ...ｘLₜ₊₁ｘLₜｘPₜ
			mul!(tmp_1, V_jt, P_t)
			V_jt= tmp_1
		end
	end
	
	return nothing
end
	
"""
	kalmansmoother(f, mat)
	
Compute smoothed states ``α`` and corresponding variances ``V`` for a linear
Gaussian State Space model with system matrices `mat` and Kalman filter output
`f`.

#### Arguments
  - `f::Filter`		: Kalman filter output
  - `mat::SysMat`	: State Space system matrices

#### Returns
  - `s::Smoother`	: Kalman smoother output
"""
function kalmansmoother(f::Filter, mat::SysMat)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initaliaze filter res. cont.
	α= Matrix{Float64}(undef, (p,T_len))
	V= Array{Float64,3}(undef, (p,p,T_len))
	s= Smoother(α,V)
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(s.α,:,t)
		V_t= view(s.V,:,:,t) 
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        F_t= view(f.F,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Lₜ
		computeL!(L, K_t, mat.Z, mat.T)
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		fac= cholesky!(tmp_nn)
		
		# Backward recursion state
		back_state!(r, L, fac, v_t, mat.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(N, L, fac, mat.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, N, tmp_pp)
	end
	
	return s
end

"""
	kalmansmoother(f, mat)
	
Compute smoothed states ``α`` and corresponding variances ``V``, using the
output of the Kalman filter based on Woodbury's identity. See also
`kalmansmoother`.
"""
function kalmansmoother(f::FilterWb, mat::SysMat)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initaliaze filter res. cont.
	α= Matrix{Float64}(undef, (p,T_len))
	V= Array{Float64,3}(undef, (p,p,T_len))
	s= Smoother(α,V)
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(s.α,:,t)
		V_t= view(s.V,:,:,t) 
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        Fi_t= view(f.Fi,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Lₜ
		computeL!(L, K_t, mat.Z, mat.T)
		
		# Backward recursion state
		back_state!(r, L, Fi_t, v_t, mat.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(N, L, Fi_t, mat.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, N, tmp_pp)
	end
	
	return s
end

"""
	kalmansmoother!(s, f, mat)
	
Compute smoothed states ``α`` and corresponding variances ``V``, storing the
results in `s`. See also `kalmansmoother`.
"""
function kalmansmoother!(s::Smoother, f::Filter, mat::SysMat)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(s.α,:,t)
		V_t= view(s.V,:,:,t) 
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        F_t= view(f.F,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Lₜ
		computeL!(L, K_t, mat.Z, mat.T)
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		fac= cholesky!(tmp_nn)
		
		# Backward recursion state
		back_state!(r, L, fac, v_t, mat.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(N, L, fac, mat.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, N, tmp_pp)
	end
	
	return nothing
end

"""
	kalmansmoother!(s, f, mat)
	
Compute smoothed states ``α`` and corresponding variances ``V``, using the
output of the Kalman filter based on Woodbury's identity. Storing the
results in `s`. See also `kalmansmoother`.
"""
function kalmansmoother!(s::Smoother, f::FilterWb, mat::SysMat)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(s.α,:,t)
		V_t= view(s.V,:,:,t) 
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        Fi_t= view(f.Fi,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Lₜ
		computeL!(L, K_t, mat.Z, mat.T)
		
		# Backward recursion state
		back_state!(r, L, Fi_t, v_t, mat.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(N, L, Fi_t, mat.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, N, tmp_pp)
	end
	
	return nothing
end

"""
	kalmansmoother_cov(f, mat, h)
	
Compute smoothed states ``α`` and corresponding variances ``V`` and
autocovariances ``V(h)`` up until lag `h` for a linear Gaussian State Space
model with system matrices `mat` and Kalman filter output `f`.

#### Arguments
  - `f::Filter`		: Kalman filter output
  - `mat::SysMat`	: State Space system matrices
  - `h::Integer`	: lag length

#### Returns
  - `s::Smoother`	: Kalman smoother output
"""
function kalmansmoother_cov(f::Filter, mat::SysMat, h::Integer)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initaliaze filter res. cont.
	α= Matrix{Float64}(undef, (p,T_len))
	V= Array{Float64,4}(undef, (p,p,h+1,T_len))
	s= Smoother(α,V)
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(s.α,:,t)
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        F_t= view(f.F,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Lₜ
		computeL!(L, K_t, mat.Z, mat.T)
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		fac= cholesky!(tmp_nn)
		
		# Backward recursion state
		back_state!(r, L, fac, v_t, mat.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(N, L, fac, mat.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(s.V, N, L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return s
end

"""
	kalmansmoother_cov(f, mat, h)
	
Compute smoothed states ``α`` and corresponding variances ``V`` and
autocovariances ``V(h)``, using the output of the Kalman filter based on
Woodbury's identity. See also `kalmansmoother_cov`.
"""
function kalmansmoother_cov(f::FilterWb, mat::SysMat, h::Integer)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initaliaze filter res. cont.
	α= Matrix{Float64}(undef, (p,T_len))
	V= Array{Float64,4}(undef, (p,p,h+1,T_len))
	s= Smoother(α,V)
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(s.α,:,t)
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        Fi_t= view(f.Fi,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Lₜ
		computeL!(L, K_t, mat.Z, mat.T)
		
		# Backward recursion state
		back_state!(r, L, Fi_t, v_t, mat.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(N, L, Fi_t, mat.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(s.V, N, L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return s
end

"""
	kalmansmoother_cov!(s, f, mat, h)
	
Compute smoothed states ``α`` and corresponding variances ``V`` and
autocovariances ``V(h)``, storing the results in `s`. See also
`kalmansmoother_cov`.
"""
function kalmansmoother_cov!(s::Smoother, f::Filter, mat::SysMat, h::Integer)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_nn= Matrix{Float64}(undef, (n,n))
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(s.α,:,t)
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        F_t= view(f.F,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Lₜ
		computeL!(L, K_t, mat.Z, mat.T)
		
		# Cholesky factorization of Fₜ
		copyto!(tmp_nn, F_t)
		fac= cholesky!(tmp_nn)
		
		# Backward recursion state
		back_state!(r, L, fac, v_t, mat.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(N, L, fac, mat.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(s.V, N, L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return nothing
end

"""
	kalmansmoother_cov!(s, f, mat, h)
	
Compute smoothed states ``α`` and corresponding variances ``V`` and
autocovariances ``V(h)``, using the output of the Kalman filter based on
Woodbury's identity. Storing the results in `s`. See also `kalmansmoother_cov`.
"""
function kalmansmoother_cov!(s::Smoother, f::FilterWb, mat::SysMat, h::Integer)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
    tmp_n= Vector{Float64}(undef, n)
    tmp_p= Vector{Float64}(undef, p)
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Smoother
	@inbounds @fastmath for t in T_len:-1:1
        # Store views
		α_t= view(s.α,:,t)
        a_t= view(f.a,:,t)
		P_t= view(f.P,:,:,t)
        v_t= view(f.v,:,t)
        Fi_t= view(f.Fi,:,:,t)
        K_t= view(f.K,:,:,t)
		
		# Lₜ
		computeL!(L, K_t, mat.Z, mat.T)
		
		# Backward recursion state
		back_state!(r, L, Fi_t, v_t, mat.Z, tmp_n, tmp_p)
		
		# Backward recursion state variance
		back_state_var!(N, L, Fi_t, mat.Z, tmp_np, tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(s.V, N, L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return nothing
end

"""
	kalmansmoother_eq(f, mat)
	
Compute smoothed states ``α`` and corresponding variances ``V`` for a linear
Gaussian State Space model with system matrices `mat` and Kalman filter output
`f` using the equation-by-equation or univariate version of the Kalman smoother.

#### Arguments
  - `f::Filter`		: Kalman filter output
  - `mat::SysMat`	: State Space system matrices

#### Returns
  - `s::Smoother`	: Kalman smoother output
"""
function kalmansmoother_eq(f::Filter, mat::SysMat)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L_i= Matrix{Float64}(undef, (p,p))
    tmp_p= Vector{Float64}(undef, p)
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initaliaze filter res. cont.
	α= Matrix{Float64}(undef, (p,T_len))
	V= Array{Float64,3}(undef, (p,p,T_len))
	s= Smoother(α,V)
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Tranpose
	Zt= transpose(mat.Z)
	
	# Smoother
	@inbounds @fastmath for	t in T_len:-1:1
		# Store views
		α_t= view(s.α,:,t)
		V_t= view(s.V,:,:,t) 
        a_t= view(f.a,:,1,t)
		P_t= view(f.P,:,:,1,t)
		
		# Time transition
		transition!(r, mat.T, 'T', tmp_p)
		transition2!(N, mat.T, tmp_pp)
		
		for i in n:-1:1
			# Store views
	        K_it= view(f.K,:,i,t)
			Z_i= view(Zt,:,i)
			
			# Lₜᵢ
			computeL_eq!(L_i, K_it, Z_i)
			
			# Backward recursion state
			back_state_eq!(r, L_i, f.F[i,t], f.v[i,t], Z_i, tmp_p)
		
			# Backward recursion state variance
			back_state_var_eq!(N, L_i, f.F[i,t], Z_i, tmp_pp)
		end
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, N, tmp_pp)
	end
	
	return s
end

"""
	kalmansmoother_eq!(s, f, mat)
	
Compute smoothed states ``α`` and corresponding variances ``V`` using the
equation-by-equation or univariate version of the Kalman smoother. Storing the
results in `s`. See also `kalmansmoother_eq`.
"""
function kalmansmoother_eq!(s::Smoother, f::Filter, mat::SysMat)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L_i= Matrix{Float64}(undef, (p,p))
    tmp_p= Vector{Float64}(undef, p)
	tmp_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Tranpose
	Zt= transpose(mat.Z)
	
	# Smoother
	@inbounds @fastmath for	t in T_len:-1:1
		# Store views
		α_t= view(s.α,:,t)
		V_t= view(s.V,:,:,t) 
        a_t= view(f.a,:,1,t)
		P_t= view(f.P,:,:,1,t)
		
		# Time transition
		transition!(r, mat.T, 'T', tmp_p)
		transition2!(N, mat.T, tmp_pp)
		
		for i in n:-1:1
			# Store views
	        K_it= view(f.K,:,i,t)
			Z_i= view(Zt,:,i)
			
			# Lₜᵢ
			computeL_eq!(L_i, K_it, Z_i)
			
			# Backward recursion state
			back_state_eq!(r, L_i, f.F[i,t], f.v[i,t], Z_i, tmp_p)
		
			# Backward recursion state variance
			back_state_var_eq!(N, L_i, f.F[i,t], Z_i, tmp_pp)
		end
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states variance
		smooth_state_var!(V_t, P_t, N, tmp_pp)
	end
	
	return nothing
end

"""
	kalmansmoother_cov_eq(f, mat, h)
	
Compute smoothed states ``α`` and corresponding variances ``V`` and
autocovariances ``V(h)`` up until lag `h` for a linear Gaussian State Space
model with system matrices `mat` and Kalman filter output `f` using the
equation-by-equation or univariate version of the Kalman smoother.

#### Arguments
  - `f::Filter`		: Kalman filter output
  - `mat::SysMat`	: State Space system matrices
  - `h::Integer`	: lag length

#### Returns
  - `s::Smoother`	: Kalman smoother output
"""
function kalmansmoother_cov_eq(f::Filter, mat::SysMat, h::Integer)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
	L_i= Matrix{Float64}(undef, (p,p))
    tmp_p= Vector{Float64}(undef, p)
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initaliaze filter res. cont.
	α= Matrix{Float64}(undef, (p,T_len))
	V= Array{Float64,4}(undef, (p,p,h+1,T_len))
	s= Smoother(α,V)
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Tranpose
	Zt= transpose(mat.Z)
	
	# Smoother
	@inbounds @fastmath for	t in T_len:-1:1
		# Store views
		α_t= view(s.α,:,t)
        a_t= view(f.a,:,1,t)
		P_t= view(f.P,:,:,1,t)
		
		# Time transitions
		transition!(r, mat.T, 'T', tmp_p)
		transition2!(N, mat.T, tmp_pp)
		
		for i in n:-1:1
			# Store views
	        K_it= view(f.K,:,i,t)
			Z_i= view(Zt,:,i)
			
			# Lₜᵢ
			computeL_eq!(L_i, K_it, Z_i)
			
			# Lₜᵢｘ...ｘLₜ₁
			copyto!(temp_pp, L)
			mul!(L, temp_pp, L_i)
			
			# Backward recursion state
			back_state_eq!(r, L_i, f.F[i,t], f.v[i,t], Z_i, tmp_p)
		
			# Backward recursion state variance
			back_state_var_eq!(N, L_i, f.F[i,t], Z_i, tmp_pp)
		end
		
		# Lₜ = TｘLₜₙｘ...ｘLₜ₁
		transition!(L, mat.T, 'N', tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(s.V, N, L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return s
end

"""
	kalmansmoother_cov_eq!(s, f, mat, h)
	
Compute smoothed states ``α`` and corresponding variances ``V`` and
autocovariances ``V(h)`` using the equation-by-equation or univariate version
of the Kalman smoother. Storing the results in `s`. See also
`kalmansmoother_cov_eq`.
"""
function kalmansmoother_cov_eq!(s::Smoother, f::Filter, mat::SysMat, h::Integer)
	# Get dims
	(p,n,T_len)= size(f.K)
	
    # Initialize temp. containers
	L= Matrix{Float64}(undef, (p,p))
	L_i= Matrix{Float64}(undef, (p,p))
    tmp_p= Vector{Float64}(undef, p)
	tmp_pp= Matrix{Float64}(undef, (p,p))
	tmp1_pp= Matrix{Float64}(undef, (p,p))
	
	# Initialize smoother
	r= zeros(Float64, p)
	N= zeros(Float64, (p,p))
	
	# Tranpose
	Zt= transpose(mat.Z)
	
	# Smoother
	@inbounds @fastmath for	t in T_len:-1:1
		# Store views
		α_t= view(s.α,:,t)
        a_t= view(f.a,:,1,t)
		P_t= view(f.P,:,:,1,t)
		
		# Time transitions
		transition!(r, mat.T, 'T', tmp_p)
		transition2!(N, mat.T, tmp_pp)
		
		for i in n:-1:1
			# Store views
	        K_it= view(f.K,:,i,t)
			Z_i= view(Zt,:,i)
			
			# Lₜᵢ
			computeL_eq!(L_i, K_it, Z_i)
			
			# Lₜᵢｘ...ｘLₜ₁
			copyto!(temp_pp, L)
			mul!(L, temp_pp, L_i)
			
			# Backward recursion state
			back_state_eq!(r, L_i, f.F[i,t], f.v[i,t], Z_i, tmp_p)
		
			# Backward recursion state variance
			back_state_var_eq!(N, L_i, f.F[i,t], Z_i, tmp_pp)
		end
		
		# Lₜ = TｘLₜₙｘ...ｘLₜ₁
		transition!(L, mat.T, 'N', tmp_pp)
		
		# Smoothed states
		smooth_state!(α_t, a_t, P_t, r)
		
		# Smoothed states autocovariance
		smooth_state_cov!(s.V, N, L, P_t, t, h, tmp_pp, tmp1_pp)
	end
	
	return nothing
end