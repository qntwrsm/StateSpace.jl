#=
forecast.jl

    Forecasting routines for linear Gaussian state space models

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2021/12/06
=#

"""
	forecast(f, mat, h)
	
Compute ``h``-step ahead state forecasts and corresponding variances for the
latent states as well as forecast error variances for a State Space model using
the Kalman filter output `f`.

#### Arguments
  - `f::Filter`		: Kalman filter output
  - `mat::SysMat`	: State Space system matrices
  - `h::Integer`	: forecast horizon

#### Returns
  - `a_h::AbstractMatrix`	: h-step ahead forecasts of states (p x h)
  - `P_h::AbstractArray`	: h-step ahead forecast variances (p x p x h)
  - `F_h::AbstractArray`	: h-step ahead forecast error variances (n x n x h)
"""
function forecast(f::Filter, mat::SysMat, h::Integer)
	# Get dimensions
	(n,p)= size(mat.Z)
	T_len= size(f.a,2)
	
	# Store views of transpose
	Tt= transpose(mat.T)
	Zt= transpose(mat.Z)
	
	#Initialize tmp. cont.
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Intialize return cont.
	a_h= Matrix{Float64}(undef, (p,h))
	P_h= Array{Float64,3}(undef, (p,p,h))
	F_h= Array{Float64,3}(undef, (n,n,h))
	
	# Initialize forecast recursions
	# Forecast
	@views mul!(a_h[:,1], mat.T, f.a[:,T_len])
	# Forecast variance
	P_h[:,:,1]= Q
	@views mul!(tmp_p, f.P[:,p*(T_len-1)+1:p*T_len], Tt)
	@views mul!(P_h[:,:,1], mat.T, tmp_p, 1., 1.)
	# Forecast error variance
	F_h[:,:,1]= mat.H
	@views mul!(tmp_np, mat.Z, P_h[:,:,1])
	@views mul!(F_h[:,:,1], tmp_np, Zt, 1., 1.)
	
	# Loop over horizon
	@inbounds @fastmath for j in 2:h
		# Forecast
		@views mul!(a_h[:,j], mat.T, a_h[:,j-1])
		# Forecast variance
		P_h[:,:,j]= mat.Q
		@views mul!(tmp_p, P_h[:,:,j-1], Tt)
		@views mul!(P_h[:,:,j], mat.T, tmp_p, 1., 1.)
		# Forecast error variance
		F_h[:,:,j]= mat.H
		@views mul!(tmp_np, mat.Z, P_h[:,:,j])
		@views mul!(F_h[:,:,j], tmp_np, Zt, 1., 1.)
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
					f::Filter, mat::SysMat, h::Integer)
	# Get dimensions
	(n,p)= size(mat.Z)
	T_len= size(k.a,2)
	
	# Store views of transpose
	Tt= transpose(mat.T)
	Zt= transpose(mat.Z)
	
	#Initialize tmp. cont.
	tmp_np= Matrix{Float64}(undef, (n,p))
	tmp_p= Matrix{Float64}(undef, (p,p))
	
	# Initialize forecast recursions
	# Forecast
	@views mul!(a_h[:,1], mat.T, f.a[:,T_len])
	# Forecast variance
	P_h[:,:,1]= Q
	@views mul!(tmp_p, f.P[:,p*(T_len-1)+1:p*T_len], Tt)
	@views mul!(P_h[:,:,1], mat.T, tmp_p, 1., 1.)
	# Forecast error variance
	F_h[:,:,1]= mat.H
	@views mul!(tmp_np, mat.Z, P_h[:,:,1])
	@views mul!(F_h[:,:,1], tmp_np, Zt, 1., 1.)
	
	# Loop over horizon
	@inbounds @fastmath for j in 2:h
		# Forecast
		@views mul!(a_h[:,j], mat.T, a_h[:,j-1])
		# Forecast variance
		P_h[:,:,j]= mat.Q
		@views mul!(tmp_p, P_h[:,:,j-1], Tt)
		@views mul!(P_h[:,:,j], mat.T, tmp_p, 1., 1.)
		# Forecast error variance
		F_h[:,:,j]= mat.H
		@views mul!(tmp_np, mat.Z, P_h[:,:,j])
		@views mul!(F_h[:,:,j], tmp_np, Zt, 1., 1.)
	end
	
	return nothing
end