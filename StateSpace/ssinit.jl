"""
ssinit.jl

Purpose:
    Initialize system matrices and state mean and covariance of state space
    model

Version:
    0       Based on ssinit.py

Date:
  2020/08/18

@author: wqt200
"""
# Include PCA routines
include("../stats/pca.jl")

function ssInit(mY, tInit, iP, args...)
    """
    Purpose:
        Initialize system matrices and state mean and covariance of a state
        space model if they aren't already initialized and extract them

    Inputs:
        mY          iN x iT matrix, data
        tInit       tuple, initial parameter estimates
        iP          int, number of states
        args        (optional) arguments for sparse estimation of sys. mat. Z

    Return value:
        mZ1         iN x iP matrix, initial system matrix Z
        mT1         iP x iP matrix, initial system matrix T
        mR1         iP x iP matrix, initial system matrix R
        mH1         iN x iN matrix, initial system matrix H
        mQ1         iP x iP matrix, initial system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance
    """
    # Check R
    if haskey(tInit,:R)
        mR1= copy(tInit.R)
    else
        mR1= Diagonal(ones(Float64,iP))
    end  
    # Check Z and T
    if (!haskey(tInit,:Z) || !haskey(tInit,:T))
        if !isempty(args)
            # Initialize via sparse PCA
            (mPC, mL)= SPCA(mY, iP, args...)
        else
            # Initialize via PCA
            (mPC, mL)= PCA(mY, iP)
        end
    end
    if haskey(tInit,:Z)
        mZ1= copy(tInit.Z)
    else
        mZ1= copy(mL)
    end
    if haskey(tInit,:T)
        mT1= copy(tInit.T)
    else
        mT1= Diagonal(diag(mPC[:,1:end-1]*mPC[:,2:end]'./(mPC[:,1:end-1]*mPC[:,1:end-1]' .+ eps(Float64))))
    end
    # Check H
    if haskey(tInit,:H)
        mH1= copy(tInit.H)
    else
        mH1= Diagonal(vec(var(mY, dims=2)))
    end
    # Check Q
    if haskey(tInit,:Q)
        mQ1= copy(tInit.Q)
    else
        mQ1= Diagonal(vec(var(mPC, dims=2)))
    end
    # Check a
    if haskey(tInit,:a)
        va1= copy(tInit.a)
    else
        va1= zeros(Float64,iP)
    end
    # Check P
    if haskey(tInit,:P)
        mP1= copy(tInit.P)
    else
        mI= Array{Float64}(I,(iP^2,iP^2))
        mITTi= inv(mI .- kron(mT1, mT1))
        mP1= reshape(mITTi*vec(diagm(diag(mQ1))),(iP,iP))
    end
        
    return (mZ1, mT1, mR1, mH1, mQ1, va1, mP1)
end

function ssInit!(mZ1, mT1, mR1, mH1, mQ1, va1, mP1, mY, tInit, iP, args...)
    """
    Purpose:
        Initialize system matrices and state mean and covariance of a state
        space model if they aren't already initialized and extract them

    Inputs:
        mY          iN x iT matrix, data
        tInit       tuple, initial parameter estimates
        iP          int, number of states
        args        (optional) arguments for sparse estimation of sys. mat. Z

    Return value:
        mZ1         iN x iP matrix, initial system matrix Z
        mT1         iP x iP matrix, initial system matrix T
        mR1         iP x iP matrix, initial system matrix R
        mH1         iN x iN matrix, initial system matrix H
        mQ1         iP x iP matrix, initial system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance
    """
    # Check R
    if haskey(tInit,:R)
        mR1.= tInit.R
    else
        mR1.= Diagonal(ones(Float64,iP))
    end  
    # Check Z and T
    if (!haskey(tInit,:Z) || !haskey(tInit,:T))
        if !isempty(args)
            # Initialize via sparse PCA
            (mPC, mL)= SPCA(mY, iP, args...)
        else
            # Initialize via PCA
            (mPC, mL)= PCA(mY, iP)
        end
    end
    if haskey(tInit,:Z)
        mZ1.= tInit.Z
    else
        mZ1.= mL
    end
    if haskey(tInit,:T)
        mT1.= tInit.T
    else
        mT1.= Diagonal(diag(mPC[:,1:end-1]*mPC[:,2:end]'./(mPC[:,1:end-1]*mPC[:,1:end-1]' .+ eps(Float64))))
    end
    # Check H
    if haskey(tInit,:H)
        mH1.= tInit.H
    else
        mH1.= Diagonal(vec(var(mY, dims=2)))
    end
    # Check Q
    if haskey(tInit,:Q)
        mQ1.= tInit.Q
    else
        mQ1.= Diagonal(vec(var(mPC, dims=2)))
    end
    # Check a
    if haskey(tInit,:a)
        va1.= tInit.a
    else
        va1.= zeros(Float64,iP)
    end
    # Check P
    if haskey(tInit,:P)
        mP1.= tInit.P
    else
        mI= Array{Float64}(I,(iP^2,iP^2))
        mITTi= inv(mI .- kron(mT1, mT1))
        mP1.= reshape(mITTi*vec(mQ1),(iP,iP))
    end
        
    return nothing
end

function ssInitSE(mY, mW, tInit, tRes, iP, iR, args...)
    """
    Purpose:
        Initialize system matrices and state mean and covariance of a state
        space model with spatial error structure in the observation equation if
        they aren't already initialized and extract them

    Inputs:
        mY          iN x iT matrix, data
        mW          iN x iN matrix, symmatric spatial weight matrix
        tInit       tuple, initial parameter estimates
        tRes        tuple, parameter restrictions
        iP          int, number of states
        iR          int, number of spatial dependence parameters
        args        (optional) arguments for sparse estimation of sys. mat. Z

    Return value:
        mZ1         iN x iP matrix, initial system matrix Z
        mT1         iP x iP matrix, initial system matrix T
        vRho1       iR x 1 vector, initial spat. dep. parameter
        mR1         iP x iP matrix, initial system matrix R
        mH1         iN x iN matrix, initial system matrix H
        mQ1         iP x iP matrix, initial system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance
    """
    (iN,iT)= size(mY)

    # Check R
    if haskey(tInit,:R)
        mR1= copy(tInit.R)
    else
        mR1= Diagonal(ones(Float64,iP))
    end  
    # Factor estimation
    if !isempty(args)
        # Sparse PCA
        (mPC, mL)= SPCA(mY, iP, args...)
    else
        # PCA
        (mPC, mL)= PCA(mY, iP)
    end
    # Errors
    mE= mY .- mL*mPC
    # Check Z
    if haskey(tInit,:Z)
        mZ1= copy(tInit.Z)
    else
        mZ1= copy(mL)
    end
    # Check T
    if haskey(tInit,:T)
        mT1= copy(tInit.T)
    else
        mT1= Diagonal(diag(mPC[:,1:end-1]*mPC[:,2:end]'./(mPC[:,1:end-1]*mPC[:,1:end-1]' .+ eps(Float64))))
    end
    # Check rho
    if haskey(tInit,:rho)
        # rho
        vRho1= copy(tInit.rho)
    else
        mWE= mW*mE
        vRho1= Array{Float64}(undef,iR)
        # Spatial dependence parameter
        if (iR == 1)
            vRho1[1]= (vec(mWE)'*vec(mE))/sum(mWE.^2)
        elseif (iR < iN)
            @inbounds for r in 1:iR
                rng= sum(tRes.rho[1:r-1])+1:sum(tRes.rho[1:r])
                vRho1[r]= (vec(mWE[rng,:])'*vec(mE[rng,:]))/sum(mWE[rng,:].^2)
            end
        else
            @inbounds for r in 1:iR
                vRho1[r]= (mWE[r,:]'*mE[r,:])/sum(mWE[r,:].^2)
            end
        end
        # Check whether rho_1 satisfies constraints
        dRhom= max(1/maximum(sum(mW, dims=1)), 1/maximum(sum(mW, dims=2)))
        @inbounds for r in 1:iR
            if (abs(vRho1[r]) >= dRhom)
                vRho1[r]= .9*dRhom
            end
        end
    end
    # Check H
    if haskey(tInit,:H)
        mH1= copy(tInit.H)
    else
        # Inverse of G
        mGi= Array{Float64}(I,(iN,iN))
        if (iR == 1)
            dRho1= vRho1[1]
            @inbounds @fastmath for j in 1:iN
                @simd for k in 1:iN
                    mGi[k,j]+= -dRho1*mW[k,j]
                end
            end
        elseif (iR < iN)
            i= 1
            @inbounds @fastmath for j in 1:iN
                if (tRes.rho[i] == j)
                    i+= 1
                end
                dRho1= vRho1[i]
                @simd for k in 1:iN
                    mGi[k,j]+= -dRho1*mW[k,j]
                end
            end
        else
            @inbounds @fastmath for j in 1:iN
                dRho1= vRho1[j]
                @simd for k in 1:iN
                    mGi[k,j]+= -dRho1*mW[k,j]
                end
            end
        end
        # H
        mH1= Diagonal(vec(var(mGi'*mE, corrected=false, dims=2)))
    end
    # Check Q
    if haskey(tInit,:Q)
        mQ1= copy(tInit.Q)
    else
        mQ1= Diagonal(vec(var(mPC, corrected=false, dims=2)))
    end
    # Check a
    if haskey(tInit,:a)
        va1= copy(tInit.a)
    else
        va1= zeros(Float64,iP)
    end
    # Check P
    if haskey(tInit,:P)
        mP1= copy(tInit.P)
    else
        mI= Array{Float64}(I,(iP^2,iP^2))
        mITTi= inv(mI .- kron(mT1, mT1))
        mP1= reshape(mITTi*vec(mQ1),(iP,iP))
    end

    return (mZ1, mT1, vRho1, mR1, mH1, mQ1, va1, mP1)
end

function ssInitSE!(mZ1, mT1, vRho1, mR1, mH1, mQ1, va1, mP1, mY, mW, tInit, tRes, iP, iR, args...)
    """
    Purpose:
        Initialize system matrices and state mean and covariance of a state
        space model with spatial error structure in the observation equation if
        they aren't already initialized and extract them

    Inputs:
        mY          iN x iT matrix, data
        mW          iN x iN matrix, symmatric spatial weight matrix
        tInit       tuple, initial parameter estimates
        tRes        tuple, parameter restrictions
        iP          int, number of states
        iR          int, number of spatial dependence parameters
        args        (optional) arguments for sparse estimation of sys. mat. Z

    Return value:
        mZ1         iN x iP matrix, initial system matrix Z
        mT1         iP x iP matrix, initial system matrix T
        vRho1       iR x 1 vector, initial spat. dep. parameter
        mR1         iP x iP matrix, initial system matrix R
        mH1         iN x iN matrix, initial system matrix H
        mQ1         iP x iP matrix, initial system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance
    """
    (iN,iT)= size(mY)

    # Check R
    if haskey(tInit,:R)
        mR1.= tInit.R
    else
        mR1.= Diagonal(ones(Float64,iP))
    end  
    # Factor estimation
    if !isempty(args)
        # Sparse PCA
        (mPC, mL)= SPCA(mY, iP, args...)
    else
        # PCA
        (mPC, mL)= PCA(mY, iP)
    end
    # Errors
    mE= mY .- mL*mPC
    # Check Z
    if haskey(tInit,:Z)
        mZ1.= tInit.Z
    else
        mZ1.= mL
    end
    # Check T
    if haskey(tInit,:T)
        mT1.= tInit.T
    else
        mT1.= Diagonal(diag(mPC[:,1:end-1]*mPC[:,2:end]'./(mPC[:,1:end-1]*mPC[:,1:end-1]' .+ eps(Float64))))
    end
    # Check rho
    if haskey(tInit,:rho)
        # rho
        vRho1.= tInit.rho
    else
        mWE= mW*mE
        vRho1= Array{Float64}(undef,iR)
        # Spatial dependence parameter
        if (iR == 1)
            vRho1[1]= (vec(mWE)'*vec(mE))/sum(mWE.^2)
        elseif (iR < iN)
            @inbounds for r in 1:iR
                rng= sum(tRes.rho[1:r-1])+1:sum(tRes.rho[1:r])
                vRho1[r]= (vec(mWE[rng,:])'*vec(mE[rng,:]))/sum(mWE[rng,:].^2)
            end
        else
            @inbounds for r in 1:iR
                vRho1[r]= (mWE[r,:]'*mE[r,:])/sum(mWE[r,:].^2)
            end
        end
        # Check whether rho_1 satisfies constraints
        dRhom= max(1/maximum(sum(mW, dims=1)), 1/maximum(sum(mW, dims=2)))
        @inbounds for r in 1:iR
            if (abs(vRho1[r]) >= dRhom)
                vRho1[r]= .9*dRhom
            end
        end
    end
    # Check H
    if haskey(tInit,:H)
        mH1.= tInit.H
    else
        # Inverse of G
        mGi= Array{Float64}(I,(iN,iN))
        if (iR == 1)
            dRho1= vRho1[1]
            @inbounds @fastmath for j in 1:iN
                @simd for k in 1:iN
                    mGi[k,j]+= -dRho1*mW[k,j]
                end
            end
        elseif (iR < iN)
            i= 1
            @inbounds @fastmath for j in 1:iN
                if (tRes.rho[i] == j)
                    i+= 1
                end
                dRho1= vRho1[i]
                @simd for k in 1:iN
                    mGi[k,j]+= -dRho1*mW[k,j]
                end
            end
        else
            @inbounds @fastmath for j in 1:iN
                dRho1= vRho1[j]
                @simd for k in 1:iN
                    mGi[k,j]+= -dRho1*mW[k,j]
                end
            end
        end
        # H
        mH1.= Diagonal(vec(var(mGi'*mE, corrected=false, dims=2)))
    end
    # Check Q
    if haskey(tInit,:Q)
        mQ1.= tInit.Q
    else
        mQ1.= Diagonal(vec(var(mPC, corrected=false, dims=2)))
    end
    # Check a
    if haskey(tInit,:a)
        va1.= tInit.a
    else
        va1.= zeros(Float64,iP)
    end
    # Check P
    if haskey(tInit,:P)
        mP1.= tInit.P
    else
        mI= Array{Float64}(I,(iP^2,iP^2))
        mITTi= inv(mI .- kron(mT1, mT1))
        mP1.= reshape(mITTi*vec(mQ1),(iP,iP))
    end

    return nothing
end

function ssInitSEX(mY, mX, mW, vImap, tInit, tRes, iP, iM, iR, args...)
    """
    Purpose:
        Initialize system matrices and state mean and covariance of a state
        space model with spatial error structure in the observation equation if
        they aren't already initialized and extract them

    Inputs:
        mY          iN x iT matrix, data
        mX          iK x iT matrix, exogenous regressors
        mW          iN x iN matrix, symmatric spatial weight matrix
        tInit       tuple, initial parameter estimates
        tRes        tuple, parameter restrictions
        iP          int, number of states
        iM          int, number of different ex. regressor parameters
        iR          int, number of spatial dependence parameters
        args        (optional) arguments for sparse estimation of sys. mat. Z

    Return value:
        mZ1         iN x iP matrix, initial system matrix Z
        mBeta1      iM x iK matrix, initial ex. regressor parameters
        mT1         iP x iP matrix, initial system matrix T
        vRho1       iR x 1 vector, initial spat. dep. parameter
        mR1         iP x iP matrix, initial system matrix R
        mH1         iN x iN matrix, initial system matrix H
        mQ1         iP x iP matrix, initial system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance
    """
    (iN,iT)= size(mY)
    iK= size(mX,1)

    # Check R
    if haskey(tInit,:R)
        mR1= copy(tInit.R)
    else
        mR1= Diagonal(ones(Float64,iP))
    end
    # Factor estimation
    if !isempty(args)
        # Sparse PCA
        (mPC, mL)= SPCA(mY, iP, args...)
    else
        # PCA
        (mPC, mL)= PCA(mY, iP)
    end
    # Errors factors
    mE= mY - mL*mPC
    # Check Z
    if haskey(tInit,:Z)
        mZ1= copy(tInit.Z)
    else
        mZ1= copy(mL)
    end
    # Check T
    if haskey(tInit,:T)
        mT1= copy(tInit.T)
    else
        mT1= Diagonal(diag(mPC[:,1:end-1]*mPC[:,2:end]'./(mPC[:,1:end-1]*mPC[:,1:end-1]' .+ eps(Float64))))
    end
    # Check beta
    if haskey(tInit,:beta)
        if (iM == 1)
            mBeta1= copy(reshape(tInit.beta,(1,iK)))
        else
            mBeta1= copy(tInit.beta)
        end
    else
        if (iM == 1)
            # Homogeneous
            mXex= reshape(repeat(mX, outer=iN),iK,iN*iT)
            mBeta1= reshape(mE,1,:)*mXex'*inv(mXex*mXex')
        else
            # Heterogeneous
            mBeta1= mE*mX'*inv(mX*mX')
        end
    end
    # Update errors (beta and factors)
    if (iM == 1)
        mE-= repeat(mBeta1, outer=iN)*mX
    else
        mE-= mBeta1*mX
    end
    # Check rho
    if haskey(tInit,:rho)
        # rho
        vRho1= copy(tInit.rho)
    else
        mWE= mW*mE
        vRho1= Array{Float64}(undef,iR)
        # Spatial dependence parameter
        if (iR == 1)
            vRho1[1]= (vec(mWE)'*vec(mE))/sum(mWE.^2)
        elseif (iR < iN)
            @inbounds for r in 1:iR
                rng= sum(tRes.rho[1:r-1])+1:sum(tRes.rho[1:r])
                vRho1[r]= (vec(mWE[rng,:])'*vec(mE[rng,:]))/sum(mWE[rng,:].^2)
            end
        else
            @inbounds for r in 1:iR
                vRho1[r]= (mWE[r,:]'*mE[r,:])/sum(mWE[r,:].^2)
            end
        end
        # Check whether rho_1 satisfies constraints
        dRhom= max(1/maximum(sum(mW, dims=1)), 1/maximum(sum(mW, dims=2)))
        @inbounds for r in 1:iR
            if (abs(vRho1[r]) >= dRhom)
                vRho1[r]= .9*dRhom
            end
        end
    end
    # Check H
    if haskey(tInit,:H)
        mH1= copy(tInit.H)
    else
        # Inverse of G
        mGi= Array{Float64}(I,(iN,iN))
        if (iR == 1)
            dRho= vRho1[1]
            @inbounds @fastmath for j in 1:iN
                @simd for k in 1:iN
                    mGi[k,j]+= -dRho*mW[k,j]
                end
            end
        elseif (iR < iN)
            @inbounds @fastmath for j in 1:iN
                @simd for k in 1:iN
                    iI= vImap[k]
                    mGi[k,j]+= -vRho1[iI]*mW[k,j]
                end
            end
        else
            @inbounds @fastmath for j in 1:iN
                @simd for k in 1:iN
                    mGi[k,j]+= -vRho1[k]*mW[k,j]
                end
            end
        end
        # H
        mH1= Diagonal(vec(var(mGi'*mE, corrected=false, dims=2)))
    end
    # Check Q
    if haskey(tInit,:Q)
        mQ1= copy(tInit.Q)
    else
        mQ1= Diagonal(vec(var(mPC, corrected=false, dims=2)))
    end
    # Check a
    if haskey(tInit,:a)
        va1= copy(tInit.a)
    else
        va1= zeros(Float64,iP)
    end
    # Check P
    if haskey(tInit,:P)
        mP1= copy(tInit.P)
    else
        mI= Array{Float64}(I,(iP^2,iP^2))
        mITTi= inv(mI .- kron(mT1, mT1))
        mP1= reshape(mITTi*vec(mQ1),(iP,iP))
    end

    return (mZ1, mBeta1, mT1, vRho1, mR1, mH1, mQ1, va1, mP1)
end

function ssInitSEX!(mZ1, mBeta1, mT1, vRho1, mR1, mH1, mQ1, va1, mP1, mY, mX, mW, 
                    vImap, tInit, tRes, iP, iM, iR, args...)
    """
    Purpose:
        Initialize system matrices and state mean and covariance of a state
        space model with spatial error structure in the observation equation if
        they aren't already initialized and extract them

    Inputs:
        mY          iN x iT matrix, data
        mX          iK x iT matrix, exogenous regressors
        mW          iN x iN matrix, symmatric spatial weight matrix
        tInit       tuple, initial parameter estimates
        tRes        tuple, parameter restrictions
        iP          int, number of states
        iM          int, number of different ex. regressor parameters
        iR          int, number of spatial dependence parameters
        args        (optional) arguments for sparse estimation of sys. mat. Z

    Return value:
        mZ1         iN x iP matrix, initial system matrix Z
        mBeta1      iM x iK matrix, initial ex. regressor parameters
        mT1         iP x iP matrix, initial system matrix T
        vRho1       iR x 1 vector, initial spat. dep. parameter
        mR1         iP x iP matrix, initial system matrix R
        mH1         iN x iN matrix, initial system matrix H
        mQ1         iP x iP matrix, initial system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance
    """
    (iN,iT)= size(mY)
    iK= size(mX,1)

    # Check R
    if haskey(tInit,:R)
        mR1.= tInit.R
    else
        mR1.= Diagonal(ones(Float64,iP))
    end
    # Factor estimation
    if !isempty(args)
        # Sparse PCA
        (mPC, mL)= SPCA(mY, iP, args...)
    else
        # PCA
        (mPC, mL)= PCA(mY, iP)
    end
    # Errors factors
    mE= mY - mL*mPC
    # Check Z
    if haskey(tInit,:Z)
        mZ1.= tInit.Z
    else
        mZ1.= mL
    end
    # Check T
    if haskey(tInit,:T)
        mT1.= tInit.T
    else
        mT1.= Diagonal(diag(mPC[:,1:end-1]*mPC[:,2:end]'./(mPC[:,1:end-1]*mPC[:,1:end-1]' .+ eps(Float64))))
    end
    # Check beta
    if haskey(tInit,:beta)
        if (iM == 1)
            mBeta1.= reshape(tInit.beta,(1,iK))
        else
            mBeta1.= tInit.beta
        end
    else
        if (iM == 1)
            # Homogeneous
            mXex= reshape(repeat(mX, outer=iN),iK,iN*iT)
            mBeta1.= reshape(mE,1,:)*mXex'*inv(mXex*mXex')
        else
            # Heterogeneous
            mBeta1.= mE*mX'*inv(mX*mX')
        end
    end
    # Update errors (beta and factors)
    if (iM == 1)
        mE-= repeat(mBeta1, outer=iN)*mX
    else
        mE-= mBeta1*mX
    end
    # Check rho
    if haskey(tInit,:rho)
        # rho
        vRho1.= tInit.rho
    else
        mWE= mW*mE
        vRho1= Array{Float64}(undef,iR)
        # Spatial dependence parameter
        if (iR == 1)
            vRho1[1]= (vec(mWE)'*vec(mE))/sum(mWE.^2)
        elseif (iR < iN)
            @inbounds for r in 1:iR
                rng= sum(tRes.rho[1:r-1])+1:sum(tRes.rho[1:r])
                vRho1[r]= (vec(mWE[rng,:])'*vec(mE[rng,:]))/sum(mWE[rng,:].^2)
            end
        else
            @inbounds for r in 1:iR
                vRho1[r]= (mWE[r,:]'*mE[r,:])/sum(mWE[r,:].^2)
            end
        end
        # Check whether rho_1 satisfies constraints
        dRhom= max(1/maximum(sum(mW, dims=1)), 1/maximum(sum(mW, dims=2)))
        @inbounds for r in 1:iR
            if (abs(vRho1[r]) >= dRhom)
                vRho1[r]= .9*dRhom
            end
        end
    end
    # Check H
    if haskey(tInit,:H)
        mH1.= tInit.H
    else
        # Inverse of G
        mGi= Array{Float64}(I,(iN,iN))
        if (iR == 1)
            dRho= vRho1[1]
            @inbounds @fastmath for j in 1:iN
                @simd for k in 1:iN
                    mGi[k,j]+= -dRho*mW[k,j]
                end
            end
        elseif (iR < iN)
            @inbounds @fastmath for j in 1:iN
                @simd for k in 1:iN
                    iI= vImap[k]
                    mGi[k,j]+= -vRho1[iI]*mW[k,j]
                end
            end
        else
            @inbounds @fastmath for j in 1:iN
                @simd for k in 1:iN
                    mGi[k,j]+= -vRho1[k]*mW[k,j]
                end
            end
        end
        # H
        mH1.= Diagonal(vec(var(mGi'*mE, corrected=false, dims=2)))
    end
    # Check Q
    if haskey(tInit,:Q)
        mQ1.= tInit.Q
    else
        mQ1.= Diagonal(vec(var(mPC, corrected=false, dims=2)))
    end
    # Check a
    if haskey(tInit,:a)
        va1.= tInit.a
    else
        va1.= zeros(Float64,iP)
    end
    # Check P
    if haskey(tInit,:P)
        mP1.= tInit.P
    else
        mI= Array{Float64}(I,(iP^2,iP^2))
        mITTi= inv(mI .- kron(mT1, mT1))
        mP1.= reshape(mITTi*vec(mQ1),(iP,iP))
    end

    return nothing
end
