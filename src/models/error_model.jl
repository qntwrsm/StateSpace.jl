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
# Constructor
function Independent(ε::AbstractMatrix)
    # Get dims
    n= size(ε, 1)

    # covariance and precision vectors
    σ= similar(ε, n)
    ω= similar(σ)

    return Independent(ε, σ, ω)
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
# Constructor
function Idiosyncratic(ε::AbstractMatrix)
    # Get dims
    n= size(ε, 1)

    # covariance and precision matrices
    Σ= Symmetric(similar(ε, n, n))
    Ω= similar(Σ)

    return Idiosyncratic(ε, Σ, Ω)
end

"""
    SpatialErrorModel <: AbstractErrorModel

Constructor for a spatial error model specification instance of the error model
type, with hyperparameter `ρ`, idiosyncratic error specification `error`,
spatial weight matrix `W`, and group structure `groups`.
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
# Constructor
function SpatialErrorModel( ε::AbstractMatrix, 
                            ρ::AbstractVector, 
                            W::AbstractMatrix, 
                            groups::AbstractVector, 
                            error::AbstractErrorModel
                        )
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

"""
    SpatialMovingAverageModel <: AbstractErrorModel

Constructor for a spatial moving average model specification instance of the
error model type, with hyperparameter `ρ`, idiosyncratic error specification
`error`, spatial weight matrix `W`, and group structure `groups`.
"""
struct SpatialMovingAverageModel{Tε, TΣ, Tρ, TW, TG, Tg, Te} <: AbstractErrorModel
    ε::Tε       # spatial errors
    Σ::TΣ       # covariance matrix
    Ω::TΣ       # precision matrix
    ρ::Tλ       # spatial dependence
    W::TW       # spatial weight matrix
    G::TG       # spatial MA polynomial      
    groups::Tg  # group structure
    error::Te   # idiosyncratic error specification
end
# Constructor
function SpatialMovingAverageModel( ε::AbstractMatrix, 
                                    ρ::AbstractVector, 
                                    W::AbstractMatrix, 
                                    groups::AbstractVector, 
                                    error::AbstractErrorModel
                                )
    # dims
    n= sum(groups)

    # covariance and precision matrices
    Σ= Symmetric(similar(ε, n, n))
    Ω= similar(Σ)

    # spatial MA polynomial
    G= similar(ρ, n, n)
    
    return SpatialMovingAverageModel(ε, Σ, Ω, ρ, W, G, groups, error)
end

# Methods
nparams(model::Independent)= length(model.σ)
nparams(model::Idiosyncratic)= length(model.Σ) * (length(model.Σ) + 1) ÷ 2
nparams(model::SpatialErrorModel)= length(model.ρ) + nparams(model.error)
nparams(model::SpatialMovingAverageModel)= length(model.ρ) + nparams(model.error)

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
function get_params!(ψ::AbstractVector, model::SpatialMovingAverageModel)
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
function store_params!(model::SpatialMovingAverageModel, ψ::AbstractVector)
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
cov(model::AbstractErrorModel)= model.Σ
cov(model::Independent)= Diagonal(model.σ)

"""
    prec(model)

Retrieve precision matrix of error model

#### Arguments
  - `model::AbstractErrorModel` : error model
"""
prec(model::AbstractErrorModel)= model.Ω
prec(model::Independent)= Diagonal(model.ω)

"""
    spatial(model)

Retrieve spatial dependence parameter of error model

#### Arguments
  - `model::AbstractErrorModel` : error model
"""
spatial(model::AbstractErrorModel)= nothing
spatial(model::SpatialErrorModel)= model.ρ
spatial(model::SpatialMovingAverageModel)= model.ρ

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
function init_ρ!(   ρ::AbstractVector, 
                    ε::AbstractMatrix, 
                    W::AbstractMatrix, 
                    groups::AbstractVector, 
                    ρ_max::Real
                )
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
function spatial_polynomial!(   G::AbstractMatrix, 
                                ρ::AbstractVector, 
                                W::AbstractMatrix, 
                                groups::AbstractVector
                            )
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
    
    return nothing
end
function init_error!(model::Idiosyncratic, init::NamedTuple)
    # variances and precisions
    if haskey(init, :error)
        model.Σ.= init.error.Σ
        model.Ω.= init.error.Ω
    else
        T= size(model.ε,2)
        mul!(model.Σ.data, model.ε, transpose(model.ε))
        lmul!(T, model.Σ.data)
        model.Ω.= model.Σ
        LinearAlgebra.inv!(cholesky!(model.Ω))
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
function init_error!(model::SpatialMovingAverageModel, init::NamedTuple)
    # spatial dependence
    haskey(init, :error) ? model.ρ.= init.error.ρ : zero(eltype(model.ρ))

    # spatial MA polynomial
    spatial_polynomial!(model.G, model.ρ, model.W, model.groups)

    # factorization
    fac= lu(model.G)

    # idsiosyncratic errors
    ldiv!(resid(model.error), fac, resid(model))
    init_error!(model.error, init)

    # variance and precision
    model.Σ.data.= model.G * cov(model.error) * transpose(model.G)
    model.Ω.= model.Σ
    LinearAlgebra.inv!(cholesky!(model.Ω))

    return nothing
end

# Estimation
# Log-likelihood
"""
    f_spatial(ρ, model, quad)

Compute objective function value `f` w.r.t. spatial dependence parameter ``ρ``
(average negative log-likelihood w.r.t. ``ρ``) for either a spatial error or
spatial moving average model.

#### Arguments
  - `ρ::AbstractVector`         : spatial dependence
  - `model::AbstractErrorModel` : error model
  - `quad::AbstractMatrix`      : quadratic smoother state space model component

#### Returns
  - `f::Real`   : objective function value
"""
function f_spatial( ρ::AbstractVector, 
                    model::SpatialErrorModel, 
                    quad::AbstractMatrix
                )
    # Get dims
    (n,T)= size(resid(model))

    # transform parameters back
    ρ.= logistic.(ρ; offset=model.ρ_max, scale=2 * model.ρ_max)
    # Spatial lag polynomial G
    spatial_polynomial!(model.G, ρ, model.W, model.groups)
    # transform parameters
    ρ.= logit.(ρ; offset=model.ρ_max, scale=2 * model.ρ_max)    

    # error component
    e= resid(model) * transpose(resid(model)) .+ quad

    # precision matrix
    tmp_nn= transpose(model.G) * prec(model.error) # buffer
    mul!(prec(model).data, tmp_nn, model.G)

    # objective function 
    f= zero(eltype(ρ))
    @inbounds @fastmath for i in 1:n
        f+= dot(view(prec(model),:,i), view(e,:,i))
    end
    # logabsdet
    d,s= logabsdet(lu!(model.G))

    return f * inv(2 * T) - d - log(s)
end
function f_spatial( ρ::AbstractVector, 
                    model::SpatialMovingAverageModel, 
                    quad::AbstractMatrix
                )
    # Get dims
    (n,T)= size(resid(model))

    # transform parameters back
    ρ.= logistic.(ρ)
    # Spatial lag polynomial G
    spatial_polynomial!(model.G, ρ, model.W, model.groups)
    # transform parameters
    ρ.= logit.(ρ)    

    # error component
    e= resid(model) * transpose(resid(model)) .+ quad

    # variance and precision
    cov(model).data.= model.G * cov(model.error) * transpose(model.G)
    prec(model).= cov(model)
    LinearAlgebra.inv!(cholesky!(prec(model)))

    # objective function 
    f= zero(eltype(ρ))
    @inbounds @fastmath for i in 1:n
        f+= dot(view(prec(moodel),:,i), view(e,:,i))
    end
    # logabsdet
    d,s= logabsdet(lu!(model.G))

    return f * inv(2 * T) + d + log(s)
end

# Gradient
"""
    ∇f_spatial!(∇f, ρ, model, quad)

Compute gradient `∇f` of objective function `f` w.r.t. spatial dependence ``ρ``
(gradient of average negative log-likelihood w.r.t. ``ρ``) for either a spatial
error or spatial moving average model.

#### Arguments
  - `ρ::AbstractVector`         : spatial dependence
  - `model::AbstractErrorModel` : error model
  - `quad::AbstractMatrix`      : quadratic smoother state space model component

#### Returns
  - `∇f::AbstractVector`: gradient
"""
function ∇f_spatial!(   ∇f::AbstractVector, 
                        ρ::AbstractVector, 
                        model::SpatialErrorModel, 
                        quad::AbstractMatrix
                    )
    # Get dims
    T= size(resid(model),2)

    # transform parameters back
    ρ.= logistic.(ρ; offset=model.ρ_max, scale=2 * model.ρ_max)
    # Spatial lag polynomial G
    spatial_polynomial!(model.G, ρ, model.W, model.groups)
    # transform parameters
    ρ.= logit.(ρ; offset=model.ρ_max, scale=2 * model.ρ_max)

    # error component
    tmp_nn= resid(model) * transpose(resid(model)) .+ quad
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
function ∇f_spatial!(   ∇f::AbstractVector, 
                        ρ::AbstractVector, 
                        model::SpatialMovingAverageModel, 
                        quad::AbstractMatrix
                    )
    # Get dims
    T= size(resid(model),2)

    # transform parameters back
    ρ.= logistic.(ρ)
    # Spatial lag polynomial G
    spatial_polynomial!(model.G, ρ, model.W, model.groups)
    # transform parameters
    ρ.= logit.(ρ)

    # variance and precision
    tmp_nn= cov(model.error) * transpose(model.G)
    mul!(cov(model).data, model.G, tmp_nn)
    prec(model).= cov(model)
    LinearAlgebra.inv!(cholesky!(prec(model)))

    # log-determinant component 
    ∇logdet= model.W / lu!(model.G)

    # error component
    tmp_nn.= quad
    mul!(tmp_nn, resid(model), transpose(resid(model)), 1., 1.)
    e= tmp_nn * transpose(∇logdet)

    # gradient
    idx= 1
    @inbounds @fastmath for i in 1:length(ρ)
        rng= idx:idx+model.groups[i]-1
        ∇f_i= zero(eltype(∇f))
        for j in rng
            ∇f_i+= ∇logdet[j,j] - inv(T) * dot(view(e,:,j), view(prec(model),:,j))
        end
        jacob= logistic(ρ[i]) * (one(ρ[i]) - logistic(ρ[i]))
        ∇f[i]= ∇f_i * jacob
        idx+= model.groups[i]
    end

    return nothing
end

# EM
"""
    update_error!(model, quad, pen)

Update error model hyper parameters, storing the results in `model`.

#### Arguments
  - `model::AbstractErrorModel` : error model
  - `quad::AbstractMatrix`      : quadratic smoother state space model component
  - `pen::Penalization`         : penalization variables
"""
function update_error!(model::Independent, quad::AbstractMatrix, pen::Penalization)
    T= size(resid(model),2)
    @inbounds @fastmath for i in axes(model.σ,1)
        ε_i= view(resid(model),i,:)
        # variance
        model.σ[i]= inv(T) * ( sum(abs2, ε_i) + quad[i,i] )
        # precision
        model.ω[i]= inv(model.σ[i])
    end

    return nothing
end
function update_error!(model::Idiosyncratic, quad::AbstractMatrix, pen::Penalization)
    # variance
    Σ= cov(model).data
    mul!(Σ, resid(model), transpose(resid(model)))
    @inbounds @fastmath for j in axes(Σ,2)
        for i in 1:j
            Σ[i,j]+= quad[i,j]
        end
    end
    T= size(resid(model),2)
    lmul!(inv(T), Σ)

    # precision
    prec(model).= Σ
    LinearAlgebra.inv!(cholesky!(prec(model)))

    return nothing
end
function update_error!(model::SpatialErrorModel, quad::AbstractMatrix, pen::Penalization)
    # Idiosyncratic components
    mul!(resid(model.error), model.G, resid(model))
    quad_spat= model.G * quad * transpose(model.G)
    update_error!(model.error, quad_spat, pen)

    # Spatial dependence
    # Closure of functions
    f(x::AbstractVector)= f_spatial(x, model, quad)
    ∇f!(∇f::AbstractVector, x::AbstractVector)=  ∇f_spatial!(∇f, x, model, quad)

    # Initial value for proximal operator
    x0= logit.(model.ρ; offset=model.ρ_max, scale=2 * model.ρ_max)

    # Proximal operators
    prox_g!(x::AbstractVector, λ::Real)= prox!(x, λ, pen)
    prox_f!(x::AbstractVector, λ::Real)= smooth!(x, λ, f, ∇f!, x0)

    # Transform parameters
    model.ρ.= logit.(model.ρ; offset=model.ρ_max, scale=2 * model.ρ_max)

    # Penalized estimation via admm
    model.ρ.= admm!(model.ρ, prox_f!, prox_g!)

    # Transform parameters back
    model.ρ.= logistic.(model.ρ; offset=model.ρ_max, scale=2 * model.ρ_max)

    # Spatial lag polynomial G
    spatial_polynomial!(model.G, model.ρ, model.W, model.groups)

    # variance and precision
    prec(model).data.= transpose(model.G) * prec(model.error) * model.G
    cov(model).= prec(model)
    LinearAlgebra.inv!(cholesky!(cov(model)))

    return nothing
end
function update_error!(model::SpatialMovingAverageModel, quad::AbstractMatrix, pen::Penalization)
    # inverse
    Ginv= inv(model.G)

    # idiosyncratic components
    mul!(resid(model.error), Ginv, resid(model))
    quad_spat= Ginv * quad * transpose(Ginv)
    update_error!(model.error, quad_spat, pen)

    # Spatial dependence
    # Closure of functions
    f(x::AbstractVector)= f_spatial(x, model, quad)
    ∇f!(∇f::AbstractVector, x::AbstractVector)=  ∇f_spatial!(∇f, x, model, quad)

    # Initial value for proximal operator
    x0= logit.(model.ρ)

    # Proximal operators
    prox_g!(x::AbstractVector, λ::Real)= prox!(x, λ, pen)
    prox_f!(x::AbstractVector, λ::Real)= smooth!(x, λ, f, ∇f!, x0)

    # Transform parameters
    model.ρ.= logit.(model.ρ)

    # Penalized estimation via admm
    model.ρ.= admm!(model.ρ, prox_f!, prox_g!)

    # Transform parameters back
    model.ρ.= logistic.(model.ρ)

    # Spatial lag polynomial G
    spatial_polynomial!(model.G, model.ρ, model.W, model.groups)

    # variance and precision
    cov(model).data.= model.G * cov(model.error) * transpose(model.G)
    prec(model).= cov(model)
    LinearAlgebra.inv!(cholesky!(prec(model)))

    return nothing
end