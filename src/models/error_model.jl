#=
error_model.jl

    Error model specifications, with accompanying utility routines and update 
    and initialization steps for the EM estimation routine.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/03/01
=#

"""
    AbstractErrorModel

Abstract type for error model specifications.
"""
abstract type AbstractErrorModel end

"""
    Independent <: AbstractErrorModel

Constructor for a independent error specification instance of the error model
type, with hyperparameters `σ` and `ω`.
"""
struct Independent{Tε, Tσ} <: AbstractErrorModel
    ε::Tε   # errors
    σ::Tσ   # variances
    ω::Tσ   # precisions
end

"""
    Idiosyncratic <: AbstractErrorModel

Constructor for a idiosyncratic error specification instance of the error model
type, with hyperparameters `Σ` and `Ω`.
"""
struct Idiosyncratic{Tε, TΣ <: Symmetric} <: AbstractErrorModel
    ε::Tε   # errors
    Σ::TΣ   # covariance matrix
    Ω::TΣ   # precision matrix
end

"""
    SpatialErrorModel <: AbstractErrorModel

Constructor for a spatial error model specification instance of the error model
type, with hyperparameter `ρ`, idiosyncratic error specification `error`,
spatial weight matrix `W`, and groups structure `groups`.
"""
struct SpatialErrorModel{Tε, TΣ, Tρ, Tf, TW, TG, Tg, Te} <: AbstractErrorModel
    ε::Tε       # spatial errors
    Σ::TΣ       # covariance matrix
    Ω::TΣ       # precision matrix
    ρ::Tρ       # spatial dependence
    ρ_max::Tf   # upper bound on absolute value of ρ
    W::TW       # spatial weight matrix
    G::TG       # spatial lag polynomial      
    groups::Tg  # group structure
    error::Te   # idiosyncratic error specification
end
# Constructors
function SpatialErrorModel(ε::AbstractMatrix, ρ::AbstractVector, W::AbstractMatrix, 
                            groups::AbstractVector, error::AbstractErrorModel)
    # dims
    n= sum(groups)

    # covariance and precision matrices
    Σ= Symmetric(similar(ε, n, n))
    Ω= similar(Σ)

    # spatial lag polynomial
    G= similar(ρ, n, n)

    # Upper bound on abs. val. of ρ to ensure G is row and column bounded
    ρ_max= max(inv(opnorm(W, 1)), inv(opnorm(W, Inf)))
    
    return SpatialErrorModel(ε, Σ, Ω, ρ, ρ_max, W, G, groups, error)
end

# Methods
nparams(model::Independent)= length(model.σ)
nparams(model::Idiosyncratic)= length(model.Σ) * (length(model.Σ) + 1) ÷ 2
nparams(model::SpatialErrorModel)= length(model.ρ) + nparams(model.error)

get_params!(ψ::AbstractVector, model::Independent)= ψ.= model.σ
function get_params!(ψ::AbstractVector, model::Idiosyncratic)
    k= 0
    @inbounds @fastmath for j in axes(model.Σ,2)
        for i in 1:j
            k+= 1
            ψ[idx+k]= model.Σ[i,j]
        end
    end

    return nothing
end
function get_params!(ψ::AbstractVector, model::SpatialErrorModel)
    get_params!(view(ψ,1:nparams(model.error)), model.error)
    ψ[nparams(model.error)+1:end].= model.ρ

    return nothing
end

store_params!(model::Independent, ψ::AbstractVector)= model.σ.= ψ
function store_params!(model::Idiosyncratic, ψ::AbstractVector)
    k= 0
    @inbounds @fastmath for j in axes(model.Σ,2)
        for i in 1:j
            k+= 1
            model.Σε[i,j]= ψ[idx+k]
        end
    end

    return nothing
end
function store_params!(model::SpatialErrorModel, ψ::AbstractVector)
    n= length(ψ)
    store_params!(model.error, view(ψ,1:nparams(model.error)))
    model.ρ.= view(ψ,nparams(model.error)+1:n)

    return nothing
end

"""
    resid(model)

Retrieve residuals of error model

#### Arguments
  - `model::AbstractErrorModel` : error model
"""
resid(model::AbstractErrorModel)= model.ε

"""
    cov(model)

Retrieve covariance matrix of error model

#### Arguments
  - `model::AbstractErrorModel` : error model
"""
cov(model::Independent)= Diagonal(model.σ)
cov(model::Idiosyncratic)= model.Σ
cov(model::SpatialErrorModel)= model.Σ

"""
    prec(model)

Retrieve precision matrix of error model

#### Arguments
  - `model::AbstractErrorModel` : error model
"""
prec(model::Independent)= Diagonal(model.ω)
prec(model::Idiosyncratic)= model.Ω
prec(model::SpatialErrorModel)= model.Ω

"""
    init_ρ!(ρ, ε, W, groups, ρ_max)

Initialize spatial dependence parameter `ρ` using spatial lag regression of
residuals `ε`, storing the result in `ρ`.

#### Arguments
  - `ε::AbstractMatrix`     : residuals
  - `W::AbstractMatrix`     : spatial weight matrix
  - `groups::AbstractVector`: group structure
  - `ρ_max::Real`           : upper bound on abs. value

#### Returns
  - `ρ::AbstractVector` : spatial dependence
"""
function init_ρ!(ρ::AbstractVector, ε::AbstractMatrix, W::AbstractMatrix, 
                groups::AbstractVector, ρ_max::Real)
    # squared residuals
    ε_sq= ε * transpose(ε)

    idx= 1
    @inbounds @fastmath for i in 1:length(ρ)
        rng= idx:idx+groups[i]-1
        # numerator and denominator
        num= zero(eltype(ρ))
        denom= zero(eltype(ρ))
        for j in rng
            w= view(W,j,:)
            num+= dot(w, view(ε_sq,:,j))
            denom+= dot(w, ε_sq, w)
        end
        # ρᵢ
        ρ_tmp= num * inv(denom)
        ρ[i]= sign(ρ_tmp) * min(abs(ρ_tmp), .9 * ρ_max)
        idx+= groups[i]
    end

    return nothing
end

"""
    spatial_polynomial!(G, ρ, W, groups)

Compute spatial lag polynomial matrix `G` based on spatial dependence `ρ` and
spatial weights `W`, storing the result in `G`.

#### Arguments
  - `ρ::AbstractVector`     : spatial dependence
  - `W::AbstractMatrix`     : spatial weight matrix
  - `groups::AbstractVector`: group structure

#### Returns
  - `G::AbstractMatrix` : spatial lag polynomial  
"""
function spatial_polynomial!(G::AbstractMatrix, ρ::AbstractVector, 
                                W::AbstractMatrix, groups::AbstractVector)
    idx= 1
    @inbounds @fastmath for i in 1:length(ρ)
        rng= idx:idx+groups[i]-1
        G[rng,:].= -ρ[i] .* view(W,rng,:)
        idx+= groups[i]
    end
    @inbounds @fastmath for i in axes(G,1)
        G[i,i]+= one(eltype(G)) 
    end

    return nothing
end

"""
    init_error!(model)

Initialize the error model hyper parameters, storing the results in `model`.

#### Arguments
  - `model::AbstractErrorModel` : error model
"""
function init_error!(model::Independent, init::NamedTuple)
    # variances and precisions
    if haskey(init, :error)
        model.σ.= init.error.σ
        model.ω.= init.error.ω
    else
        T= size(model.ε,2)
        @inbounds @fastmath for i in axes(model.σ,1)
            ε_i= view(model.ε,i,:)
            model.σ[i]= inv(T) * sum(abs2, ε_i)
            model.ω[i]= inv(model.σ[i])
        end
    end
    model.σ.= 1.
    model.ω.= 1.
    
    return nothing
end
function init_error!(model::Idiosyncratic, init::NamedTuple)
    # variances and precisions
    if haskey(init, :error)
        model.Σ.= init.error.Σ
        model.Ω.= init.error.Ω
    else
        model.Σ.= zero(eltype(model.Σ))
        model.Ω.= zero(eltype(model.Ω))
        T= size(model.ε,2)
        @inbounds @fastmath for i in axes(model.Σ,1)
            ε_i= view(model.ε,i,:)
            model.Σ[i,i]= inv(T) * sum(abs2, ε_i)
            model.Ω[i,i]= inv(model.Σ[i,i])
        end
    end
    
    return nothing
end
function init_error!(model::SpatialErrorModel, init::NamedTuple)
    # spatial dependence
    haskey(init, :error) ? model.ρ.= init.error.ρ : init_ρ!(model.ρ, resid(model), model.W, model.groups, model.ρ_max)

    # spatial lag polynomial
    spatial_polynomial!(model.G, model.ρ, model.W, model.groups)

    # idsiosyncratic errors
    mul!(resid(model.error), model.G, resid(model))
    init_error!(model.error, init)

    # variance and precision
    model.Ω.data.= transpose(model.G) * prec(model.error) * model.G
    model.Σ.= model.Ω
    LinearAlgebra.inv!(cholesky!(model.Σ))

    return nothing
end

"""
    f_ρ(ρ, model, quad)

Compute objective function value `f` w.r.t. spatial dependence``ρ`` (average
negative log-likelihood w.r.t. ``ρ``).

#### Arguments
  - `ρ::AbstractVector`         : spatial dependence
  - `model::SpatialErrorModel`  : error model
  - `quad::AbstractMatrix`      : quadratic smoother state space model component

#### Returns
  - `f::Real`   : objective function value
"""
function f_ρ(ρ::AbstractVector, model::SpatialErrorModel, quad::AbstractMatrix)
    # Get dims
    (n,T)= size(model.ε)

    # transform parameters back
    ρ.= logistic.(ρ; offset=model.ρ_max, scale=2 * model.ρ_max)
    # Spatial lag polynomial G
    spatial_polynomial!(model.G, ρ, model.W, model.groups)
    # transform parameters
    ρ.= logit.(ρ; offset=model.ρ_max, scale=2 * model.ρ_max)    

    # error component
    e= model.ε * transpose(model.ε) .+ quad

    # precision matrix
    tmp_nn= transpose(model.G) * prec(model.error) # buffer
    mul!(model.Ω.data, tmp_nn, model.G)

    # objective function 
    f= zero(eltype(ρ))
    @inbounds @fastmath for i in 1:n
        f+= dot(view(model.Ω,:,i), view(e,:,i))
    end
    # logabsdet
    d,s= logabsdet(lu!(model.G))

    return f * inv(2 * T) - d - log(s)
end

"""
    ∇f_ρ!(∇f, ρ, model, quad)

Compute gradient `∇f` of objective function `f` w.r.t. spatial dependence ``ρ``
(gradient of average negative log-likelihood w.r.t. ``ρ``).

#### Arguments
  - `ρ::AbstractVector`         : spatial dependence
  - `model::SpatialErrorModel`  : error model
  - `quad::AbstractMatrix`      : quadratic smoother state space model component

#### Returns
  - `∇f::AbstractVector`: gradient
"""
function ∇f_ρ!(∇f::AbstractVector, ρ::AbstractVector, model::SpatialErrorModel, 
                quad::AbstractMatrix)
    # Get dims
    T= size(model.ε,2)

    # transform parameters back
    ρ.= logistic.(ρ; offset=model.ρ_max, scale=2 * model.ρ_max)
    # Spatial lag polynomial G
    spatial_polynomial!(model.G, ρ, model.W, model.groups)
    # transform parameters
    ρ.= logit.(ρ; offset=model.ρ_max, scale=2 * model.ρ_max)

    # error component
    tmp_nn= model.ε * transpose(model.ε) .+ quad
    e= tmp_nn * transpose(model.W)

    # variance component
    mul!(tmp_nn, transpose(model.G), prec(model.error))

    # log-determinant component 
    ∇logdet= model.W / lu!(model.G)

    # gradient
    idx= 1
    @inbounds @fastmath for i in 1:length(ρ)
        rng= idx:idx+model.groups[i]-1
        ∇f_i= zero(eltype(∇f))
        for j in rng
            ∇f_i+= ∇logdet[j,j] - inv(T) * dot(view(e,:,j), view(tmp_nn,:,j))
        end
        jacob= 2 * model.ρ_max * logistic(ρ[i]) * (one(ρ[i]) - logistic(ρ[i]))
        ∇f[i]= ∇f_i * jacob
        idx+= model.groups[i]
    end

    return nothing
end

"""
    update_error!(model, quad, pen)

Update error model hyper parameters, storing the results in `model`.

#### Arguments
  - `model::AbstractErrorModel` : error model
  - `quad::AbstractMatrix`      : quadratic smoother state space model component
  - `pen::Penalization`         : penalization variables
"""
function update_error!(model::Independent, quad::AbstractMatrix, pen::Penalization)
    T= size(model.ε,2)
    @inbounds @fastmath for i in axes(model.σ,1)
        ε_i= view(model.ε,i,:)
        # variance
        model.σ[i]= inv(T) * ( sum(abs2, ε_i) + quad[i,i] )
        # precision
        model.ω[i]= inv(model.σ[i])
    end

    return nothing
end
function update_error!(model::Idiosyncratic, quad::AbstractMatrix, pen::Penalization)
    # variance
    Σ= model.Σ.data
    mul!(Σ, model.ε, transpose(model.ε))
    @inbounds @fastmath for j in axes(Σ,2)
        for i in 1:j
            Σ[i,j]+= quad[i,j]
        end
    end
    T= size(model.ε,2)
    lmul!(inv(T), Σ)

    # precision
    Ω.= Σ
    LinearAlgebra.inv!(cholesky!(Ω))

    return nothing
end
function update_error!(model::SpatialErrorModel, quad::AbstractMatrix, pen::Penalization)
    # Idiosyncratic components
    mul!(resid(model.error), model.G, resid(model))
    quad_spat= model.G * quad * transpose(model.G)
    update_error!(model.error, quad_spat, pen)

    # Spatial dependence
    # Closure of functions
    f_cl(x::AbstractVector)= f_ρ(x, model, quad)
    ∇f_cl!(∇f::AbstractVector, x::AbstractVector)=  ∇f_ρ!(∇f, x, model, quad)
    prox_cl!(x::AbstractVector, λ::Real)= prox!(x, λ, pen)

    # Transform parameters
    model.ρ.= logit.(model.ρ; offset=model.ρ_max, scale=2 * model.ρ_max)

    # Penalized estimation via proximal gradient method
    prox_grad!(model.ρ, f_cl, ∇f_cl!, prox_cl!; style="nesterov")

    # Transform parameters back
    model.ρ.= logistic.(model.ρ; offset=model.ρ_max, scale=2 * model.ρ_max)

    # Spatial lag polynomial G
    spatial_polynomial!(model.G, model.ρ, model.W, model.groups)

    # variance and precision
    model.Ω.data.= transpose(model.G) * prec(model.error) * model.G
    model.Σ.= model.Ω
    LinearAlgebra.inv!(cholesky!(model.Σ))


    return nothing
end