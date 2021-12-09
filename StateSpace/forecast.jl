#=
forecast.jl

    Forecasting routines for linear Gaussian state space models

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2021/12/06

TODO: In-place version and eq-by-eq version
=#

"""
	forecast(kalman::Kalman, Z, T, H, Q, h)
	
Compute h-step ahead state forecasts and corresponding variances for the latent
states as well as forecast error variances for a State Space model using the
Kalman filter output and routines.

#### Arguments
  - `k::Kalman`			: Kalman filter/smoother output
  - `Z::AbstractMatrix`	: system matrix Z (n x p)
  - `T::AbstractMatrix`	: system matrix T (p x p)
  - `H::AbstractMatrix`	: system matrix H (n x n)
  - `Q::AbstractMatrix`	: system matrix Q (p x p)
  - `h::Integer`		: forecast horizon

#### Returns
  - `a_h::AbstractMatrix`	: h-step ahead forecasts of states (p x h)
  - `P_h::AbstractArray`	: h-step ahead forecast variances (p x p x h)
  - `F_h::AbstractArray`	: h-step ahead forecast error variances (n x n x h)
"""
function forecast(k::Kalman, Z::AbstractMatrix, T::AbstractMatrix, H::AbstractMatrix, Q::AbstractMatrix, h::Integer)
	# Get dimensions
	(n,p)= size(Z)
	T_len= size(k.a,2)
	
	# Store views of transpose
	Tt= transpose(T)
	Zt= transpose(Z)
	
	#Initialize tmp. cont.
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Intialize return cont.
	a_h= Matrix{Float64}(undef, (p,h))
	P_h= Array{Float64,3}(undef, (p,p,h))
	F_h= Array{Float64,3}(undef, (n,n,h))
	
	# Initialize forecast recursions
	# Forecast
	@views mul!(a_h[:,1], T, k.a[:,T_len])
	# Forecast variance
	P_h[:,:,1]= Q
	@views mul!(tmp_p, k.P[:,p*(T_len-1)+1:p*T_len], Tt)
	@views mul!(P_h[:,:,1], T, tmp_p, 1., 1.)
	# Forecast error variance
	F_h[:,:,1]= H
	@views mul!(tmp_np, Z, P_h[:,:,1])
	@views mul!(F_h[:,:,1], tmp_np, Zt, 1., 1.)
	
	# Loop over horizon
	@inbounds @fastmath for j in 2:h
		# Forecast
		@views mul!(a_h[:,j], T, a_h[:,j-1])
		# Forecast variance
		P_h[:,:,j]= Q
		@views mul!(tmp_p, P_h[:,:,j-1], Tt)
		@views mul!(P_h[:,:,j], T, tmp_p, 1., 1.)
		# Forecast error variance
		F_h[:,:,j]= H
		@views mul!(tmp_np, Z, P_h[:,:,j])
		@views mul!(F_h[:,:,j], tmp_np, Zt, 1., 1.)
	end
	
	return (a_h, P_h, F_h)
end