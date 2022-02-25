#=
forecast.jl

    Forecasting routines for linear Gaussian state space models

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2021/12/06
=#

"""
	quad_update!(C, X, A, B, tmp)
	
Compute the quadratic time transition for forecast objects for time ``T+h`` for
a linear Gaussian State Space model, with transition matrix `A` and constant
`B`. Storing the results in `C`.

#### Arguments
  - `X::AbstractMatrix`		: input (k x k)
  - `A::AbstractMatrix`		: transition matrix (n x k)
  - `B::AbstractMatrix`		: constant matrix (n x n)
  - `tmp::AbstractMatrix`	: tmp storage array (n x k)

#### Returns
  - `C::AbstractMatrix`		: forecast object (n x n)
"""
function quad_update!(C::AbstractMatrix, X::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, tmp::AbstractMatrix)
	# AｘX
	mul!(tmp, A, X)
	# AｘXｘA'
	mul!(C, tmp, transpose(A))
	# C = AｘXｘA' + B
	@. C+= B
end

function quad_update!(C::AbstractMatrix, X::AbstractMatrix, A::Diagonal, B::AbstractMatrix, tmp::AbstractMatrix)
	# C = AｘXｘA' + B
	@. C= A.diag*X*transpose(A.diag) + B
end

function quad_update!(C::AbstractMatrix, X::AbstractMatrix, A::AbstractMatrix, B::Diagonal, tmp::AbstractMatrix)
	# AｘX
	mul!(tmp, A, X)
	# AｘXｘA'
	mul!(C, tmp, transpose(A))
	# C = AｘXｘA' + B
	@inbounds @fastmath for i in axes(B,1)
		C[i,i]+= B.diag[i]
	end
end

"""
	forecast(f, mat, h)
	
Compute ``h``-step ahead state forecasts and corresponding variances for the
latent states as well as forecast error variances for a State Space model using
the Kalman filter output `f`.

#### Arguments
  - `f::KalmanFilter`		: Kalman filter output
  - `sys::StateSpaceSystem`	: state space system matrices
  - `h::Integer`			: forecast horizon

#### Returns
  - `a_h::AbstractMatrix`	: h-step ahead forecasts of states (p x h)
  - `P_h::AbstractArray`	: h-step ahead forecast variances (p x p x h)
  - `F_h::AbstractArray`	: h-step ahead forecast error variances (n x n x h)
"""
function forecast(f::KalmanFilter, sys::StateSpaceSystem, h::Integer)
	# Get dimensions
	(n,p)= size(mat.Z)
	T_len= size(f.a,2)
	
	#Initialize tmp. cont.
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Intialize return cont.
	a_h= Matrix{Float64}(undef, (p,h))
	P_h= Array{Float64,3}(undef, (p,p,h))
	F_h= Array{Float64,3}(undef, (n,n,h))
	
	# Loop over horizon
	@inbounds @fastmath for j in 1:h
		# Store views
		a_f= view(a_h,:,j)
		P_f= view(P_h[:,:,j])
		if j != 1
			a_c= view(a_h,:,j-1)
			P_c= view(P_h[:,:,j-1])
		else
			a_c= view(f.a,:,T_len)
			P_c= view(f.P,:,:,T_len)
		end
		F_f= view(F_h,:,:,j)
		
		# Forecast
		mul!(a_f, mat.T, a_c)
		# Forecast variance
		quad_update!(P_f, P_c, mat.T, mat.Q, tmp_p)
		# Forecast error variance
		quad_update!(F_f, P_f, mat.Z, mat.H, tmp_np)
	end
	
	return (a_h, P_h, F_h)
end

"""
	forecast!(a_h, P_h, F_h, f, mat, h)
	
Compute ``h``-step ahead state forecasts and corresponding variances for the
latent states as well as forecast error variances for a State Space model and
storing them in `a_h`, `P_h`, and `F_h`. See also `forecast`.
"""
function forecast!(a_h::AbstractMatrix, P_h::AbstractArray, F_h::AbstractArray, 
					f::KalmanFilter, sys::StateSpaceSystem, h::Integer)
	# Get dimensions
	(n,p)= size(mat.Z)
	T_len= size(f.a,2)

	#Initialize tmp. cont.
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_p= Matrix{Float64}(undef, (p,p))

	# Loop over horizon
	@inbounds @fastmath for j in 1:h
		# Store views
		a_f= view(a_h,:,j)
		P_f= view(P_h[:,:,j])
		if j != 1
			a_c= view(a_h,:,j-1)
			P_c= view(P_h[:,:,j-1])
		else
			a_c= view(f.a,:,T_len)
			P_c= view(f.P,:,:,T_len)
		end
		F_f= view(F_h,:,:,j)

		# Forecast
		mul!(a_f, mat.T, a_c)
		# Forecast variance
		quad_update!(P_f, P_c, mat.T, mat.Q, tmp_p)
		# Forecast error variance
		quad_update!(F_f, P_f, mat.Z, mat.H, tmp_np)
	end

	return nothing
end