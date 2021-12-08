"""
ssll.py

Purpose:
    Log-likelihood of a general state space model based on the Kalman filter

Version:
    0       Based on ssll.py

Date:
    2020/08/18

@author: wqt200
"""
# Include numerical helper routines
include("../misc/num.jl")

function LnLStateSpace(kalman::Kalman)
    """
    Purpose:
        Compute log-likelihood of a state space model based on the Kalman
        filter, where the Kalman filter is an input

    Inputs:
        kalman      Kalman struct, Kalman filter/smoother output

    Return value:
        dLL         double, log-likelihood
    """
    (iN,iT)= size(kalman.v)

    # Initialize return container
    dLL= -log(2*pi)*iT*iN

    # Log-likelihood
    @inbounds @fastmath for t in 1:iT
        vv= view(kalman.v,:,t)
        mFi= view(kalman.Fi,:,(t-1)*iN+1:t*iN)
        dLL-= dot(vv, mFi, vv)
        dLL+= inLogDet!(mFi)
    end

    return .5*dLL
end

function LnLStateSpace(kalman::KalmanEq)
    """
    Purpose:
        Compute log-likelihood of a state space model based on the Kalman
        filter, where the Kalman filter is an input

    Inputs:
        kalman      KalmanEq struct, Eq-by-eq Kalman filter/smoother output

    Return value:
        dLL         double, log-likelihood
    """
    (iN,iT)= size(mv)
    
    # Nan's and/or negative variances
    bCorr= true
    @inbounds @fastmath for t in 1:iT
        for i in 1:iN
            dF= kalman.F[i,t]
            bCorr*= !isnan(dF) * (dF >= 0)
        end
    end

    if bCorr
        # Initialize return container
        dLL= .0
        # Log-likelihood equation-by-equation
        @inbounds @fastmath for t in 1:iT
            for i in 1:iN
                dF= kalman.F[i,t]
                if (dF > 0)
                    dLL+= -.5*(log(2*pi) + log(dF) + kalman.v[i,t]^2*inv(dF))
                end
            end
        end
    else
        dLL= -1e20
    end

    return dLL
end

function LnLStateSpace(mY, mZ, mT, mH::Diagonal{Float64,Vector{Float64}}, mQ, mR, va1, mP1)
    """
    Purpose:
        Compute log-likelihood of a state space model based on the Kalman
        filter, where the observation equation errors are uncorrelated i.e. H is
        diagonal

    Inputs:
        mY          iN x iT matrix, data
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        mH          iN x iN matrix, system matrix H
        mQ          iP x iP matrix, system matrix Q
        mR          iP x iP matrix, system matrix R
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance

    Return value:
        dLL         double, log-likelihood
    """
    (iN,iT)= size(mY)

    # Kalman filter equation-by-equation
    (_, _, mv, mF)= KalmanFilterEq_for(mY, mZ, mT, mH, mQ, mR, va1, mP1)

    # Nan's and/or negative variances
    bCorr= true
    @inbounds @fastmath for t in 1:iT
        for i in 1:iN
            dF= mF[i,t]
            bCorr*= !isnan(dF) * (dF >= 0)
        end
    end

    if bCorr
        # Initialize return container
        dLL= .0
        # Log-likelihood equation-by-equation
        @inbounds @fastmath for t in 1:iT
            for i in 1:iN
                dF= mF[i,t]
                if (dF > 0)
                    dLL+= -.5*(log(2*pi) + log(dF) + mv[i,t]^2*inv(dF))
                end
            end
        end
    else
        dLL= -1e20
    end

    return dLL
end

function LnLStateSpace(mY, mZ, mT, mH, mQ, mR, va1, mP1)
    """
    Purpose:
        Compute log-likelihood of a state space model with a spatial error
        structure in the observation equation based on the Kalman filter

    Inputs:
        mY          iN x iT matrix, data
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        mH          iN x iN matrix, system matrix H
        mQ          iP x iP matrix, system matrix Q
        mR          iP x iP matrix, system matrix R
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance

    Return value:
        dLL         double, log-likelihood
    """
    (iN,iT)= size(mY)

    # Cholesky decomposition
    mCl= cholesky(Hermitian(mH)).L
    mD= zeros(Float64,(iN,iN))
    @inbounds @fastmath for i in 1:iN
        dTemp= mCl[i,i]
        for j in 1:iN
            mCl[j,i]= mCl[j,i]/dTemp
        end
        mD[i,i]= dTemp^2
    end
    mCli= inv(mCl)

    # Transform variables
    # Y*
    mYs= mCli*mY
    # Z*
    mZs= mCli*mZ

    # Kalman filter equation-by-equation
    (_, _, mv, mF)= KalmanFilterEq_for(mYs, mZs, mT, mD, mQ, mR, va1, mP1)

    # Nan's and/or negative variances
    bCorr= true
    @inbounds @fastmath for t in 1:iT
        for i in 1:iN
            dF= mF[i,t]
            bCorr*= !isnan(dF) * (dF >= 0)
        end
    end

    if bCorr
        # Initialize return container
        dLL= .0
        # Log-likelihood equation-by-equation
        @inbounds @fastmath for t in 1:iT
            for i in 1:iN
                dF= mF[i,t]
                if (dF > 0)
                    dLL+= -.5*(log(2*pi) + log(dF) + mv[i,t]^2*inv(dF))
                end
            end
        end
    else
        dLL= -1e20
    end

    return dLL
end