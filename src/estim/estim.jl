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
struct NoPen <: Penalization
end

mutable struct Lasso{Tγ, Tw} <: Penalization
	γ::Tγ		# penalization strength
	weights::Tw	# weights
end

mutable struct GenLasso{Tγ, Tw, TD} <: Penalization
	γ::Tγ		# penalization strength
	weights::Tw	# weights
    D::TD       # penalty matrix
end

mutable struct GroupLasso{Tγ, Tw, Tg} <: Penalization
	γ::Tγ		# penalization strength
	weights::Tw	# weights
	groups::Tg	# group structure
end

mutable struct SparseGroupLasso{Tγ, Tw1, Tw2, Tg} <: Penalization
	γ::Tγ			# penalization strength
	α::Tγ			# mixing parameter
	weights_l1::Tw1	# weights ℓ₁-norm
	weights_l2::Tw2	# weights ℓ₂-norm
	groups::Tg		# group structure
end

# Proximal Operators
"""
    prox!(x, λ, pen)

Compute proximal operatior for type `pen` at point `x` with scale `λ`, storing
the result in `x`.

#### Arguments
  - `x::AbstractVector` : input
  - `λ::Real`           : scaling parameter
  - `pen::Penalization` : penalization variables
"""
prox!(x::AbstractVector, λ::Real, pen::NoPen)= nothing
prox!(x::AbstractVector, λ::Real, pen::Lasso)= x.= soft_thresh.(x, λ .* pen.γ .* vec(pen.weights))
function prox!(x::AbstractVector, λ::Real, pen::GroupLasso)
    idx= 1
    for g ∈ eachindex(pen.groups)
        rng= idx:idx+pen.groups[g]-1
        x_g= view(x,rng)
        block_soft_thresh!(x_g, λ * pen.γ * pen.weights[g])
        idx+= pen.groups[g]
    end 

    return nothing
end 
function prox!(x::AbstractVector, λ::Real, pen::SparseGroupLasso)
    idx= 1
    x.= soft_thresh.(x, λ .* pen.γ .* pen.α .* vec(pen.weights_l1))
    for g ∈ eachindex(pen.groups)
        rng= idx:idx+pen.groups[g]-1
        x_g= view(x,rng)
        block_soft_thresh!(x_g, λ * pen.γ * (one(pen.α) - pen.α) * pen.weights_l2[g])
        idx+= pen.groups[g]
    end

    return nothing
end

# Transformations
"""
    logistic(x; offset=0.0, scale=1.0)

Evaluate the logistic function at point `x`, with optionally an `offset` and
`scale`, mapping `x` ``∈ [-∞, +∞]`` to [-`offset`, `scale`-`offset`].

#### Arguments
  - `x::Real`       : input
  - `offset::Real`  : offset
  - `scale::Real`   : scale

#### Returns
  - `y::Real`   : output
"""
logistic(x::Real; offset::Real=0.0, scale::Real=1.0)= scale * inv(one(x) + exp(-x)) - offset

"""
    logit(x; offset=0.0, scale=1.0)

Evaluate the logit function (inverse logistic) at point `x`, with optionally an
`offset` and `scale`, mapping `x` ``∈`` [-`offset`, `scale`-`offset`] to ``[-∞,
+∞]``.

#### Arguments
  - `x::Real`       : input
  - `offset::Real`  : offset
  - `scale::Real`   : scale

#### Returns
  - `y::Real`   : output
"""
logit(x::Real; offset::Real=0.0, scale::Real=1.0)= log(inv(scale) * (x + offset) * inv(one(x) - inv(scale) * (x + offset)))