#=
estim.jl

    State space model optimization abstract types and general fallback routines

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/15
=#

"""
    StateSpaceOptimizerState

Abstract type for state space model optimizer.
"""
abstract type StateSpaceOptimizerState end
abstract type EMOptimizerState <: StateSpaceOptimizerState end 
abstract type MLOptimizerState <: StateSpaceOptimizerState end

"""
    Penalization

Abstract type for penalization of state space model hyper parameters.
"""
abstract type Penalization end

# Penalization structs
struct Lasso{Tγ, Tw} <: Penalization
	γ::Tγ		# penalization strength
	weights::Tw	# weights
end

struct GroupLasso{Tγ, Tw, Tg} <: Penalization
	γ::Tγ		# penalization strength
	weights::Tw	# weights
	groups::Tg	# group structure
end

struct SparseGroupLasso{Tγ, Tw1, Tw2, Tg} <: Penalization
	γ::Tγ			# penalization strength
	α::Tγ			# mixing parameter
	weights_l1::Tw1	# weights ℓ₁-norm
	weights_l2::Tw2	# weights ℓ₂-norm
	groups::Tg		# group structure
end