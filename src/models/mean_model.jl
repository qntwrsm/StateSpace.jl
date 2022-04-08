#=
mean_model.jl

    Mean model specifications

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/03/01
=#

"""
    AbstractMeanModel

Abstract type for mean model specifications.
"""
abstract type AbstractMeanModel end

"""
    NoConstant <: AbstractMeanModel

Constructor for a no constant mean specification instance of the mean model
type.
"""
struct NoConstant <: AbstractMeanModel end

"""
    Constant <: AbstractMeanModel

Constructor for a constant mean specification instance of the mean model type,
with hyperparameter `μ`.
"""
struct Constant{Tμ} <: AbstractMeanModel
    μ::Tμ   # constant
end

"""
    Exogeneous <: AbstractMeanModel

Constructor for a exogeneous regressors mean model specification instance of the
mean model type, with hyperparameter `β` and exogenous regressors `X`.
"""
struct Exogeneous{Tμ, Tβ, TX, TC} <: AbstractMeanModel
    μ::Tμ   # mean
    β::Tβ   # slopes
    X::TX   # regressors
    C::TC   # Cholesky factorization of Gram matrix
end
# Constructor
function Exogeneous(β::AbstractMatrix, X::AbstractMatrix)
    # mean
    μ= β * X
    # Cholesky factorization of Gram matrix
    C= cholesky!(Hermitian(X*transpose(X)))

    return Exogeneous(μ, β, X, C)
end

# Methods
nparams(model::NoConstant)= 0
nparams(model::Constant)= length(model.μ)
nparams(model::Exogeneous)= length(model.β)

get_params!(ψ::AbstractVector, model::NoConstant)= nothing
get_params!(ψ::AbstractVector, model::Constant)= ψ.= model.μ
get_params!(ψ::AbstractVector, model::Exogeneous)= ψ.= vec(model.β)

store_params!(model::NoConstant, ψ::AbstractVector)= nothing
store_params!(model::Constant, ψ::AbstractVector)= model.μ.= ψ
store_params!(model::Exogeneous, ψ::AbstractVector)= vec(model.β).= ψ

"""
    mean(model)

Retrieve mean of mean model

#### Arguments
  - `model::AbstractMeanModel`   : mean model
"""
mean(model::AbstractMeanModel)= model.μ
mean(model::NoConstant)= nothing 

"""
    mean!(model)

Calculate and update mean of mean model

#### Arguments
  - `model::AbstractMeanModel`  : mean model
"""
mean!(model::AbstractMeanModel)= nothing
mean!(model::Exogeneous)= mul!(model.μ, model.β, model.X)

"""
    mean!(model, β)

Calculate and update mean of exogeneous mean model with user provided slope
coefficients

#### Arguments
  - `model::AbstractMeanModel`  : mean model
  - `β::AbstractMatrix`         : slopes 
"""
mean!(model::Exogeneous, β::AbstractMatrix)= mul!(model.μ, β, model.X)

"""
    init_mean!(model, y)

Initialize the mean model hyper parameters, storing the results in `model`.

#### Arguments
  - `model::AbstractMeanModel`  : mean model
  - `y::AbstractMatrix`         : data
"""
init_mean!(model::NoConstant, y::AbstractMatrix, init::NamedTuple)= nothing
init_mean!(model::Constant, y::AbstractMatrix, init::NamedTuple)=  haskey(init, :mean) ? model.μ.= init.mean : sum!(model.μ, y) .* inv(size(y,2))
function init_mean!(model::Exogeneous, y::AbstractMatrix, init::NamedTuple)
    if haskey(init, :mean)
        model.β.= init.mean
    else
        mul!(model.β, y, transpose(model.X))
        rdiv!(model.β, model.C)
    end
    mul!(model.μ, model.β, model.X)

    return nothing
end

"""
    update_mean!(model, init_resid!, error, pen)

Update mean model hyper parameters, storing the results in `model`.

#### Arguments
  - `model::AbstractMeanModel`  : error model
  - `init_resid!::Function`     : initialize residual function
  - `error::AbstractErrorModel` : error model
  - `pen::Penalization`         : penalization variables
"""
update_mean!(model::NoConstant, init_resid!::Function, error::AbstractErrorModel, pen::Penalization)= nothing
function update_mean!(model::Exogeneous, init_resid!::Function, error::AbstractErrorModel, pen::Penalization)
    # residuals
    init_resid!(resid(error))
    # precision matrix
    Ω= prec(error)
    
    # linear coefficient b
    b= -inv(prod(size(model.μ))) * vec(Ω * resid(error) * transpose(model.X))
    # I + A, with A quadratic coefficient
    tmp= inv(prod(size(model.μ))) * kron(model.X * transpose(model.X), Ω)
    @inbounds @fastmath for i in axes(tmp,1)
        tmp[i,i]+= one(eltype(tmp))
    end
    # Cholesky decomposition
    C= cholesky!(Hermitian(tmp))

    # Proximal operators
    prox_g!(x::AbstractVector, λ::Real)= prox!(x, λ, pen)
    prox_f!(x::AbstractVector, λ::Real)= shrinkage!(x, λ, C, b)

    # Penalized estimation via admm
    vec(model.β).= admm!(vec(model.β), prox_f!, prox_g!)

    # mean
    mean!(model)

    return nothing
end
function update_mean!(model::Exogeneous, init_resid!::Function, error::AbstractErrorModel, pen::NoPen)
    # update residuals
    ε= resid(error) # retrieve residuals
    init_resid!(ε)

    # OLS
    mul!(model.β, ε, transpose(model.X))
    rdiv!(model.β, model.C)

    # mean
    mean!(model)

    return nothing
end