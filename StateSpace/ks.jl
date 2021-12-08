"""
ks.py

Purpose:
    Kalman smoother in matrix form and equation-by-equation

Version:
    0       Based on ks.py

Date:
    2020/07/29

@author: wqt200
"""
function KalmanSmoother(mZ, mT, ma, mP, mv, mF)
    """
    Purpose:
        Implement the Kalman state smoother algorithm to obtain smoothed states
        and smoothed states variance

    Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT vector, predicted state vector
        mP          iP x iT*iP matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT*iN matrix, forecast error variance

    Return value:
        mah         iP x iT matrix, predicted state vectors mean
        mV          iP x iT*iP matrix, predicted state vectors variance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize return containers
    mah= Array{Float64}(undef,(iP,iT))
    mV= Array{Float64}(undef,(iP,iT*iP))

    # Initialize temp. containers
    mK= Array{Float64}(undef,(iP,iP))
    mL= Array{Float64}(undef,(iP,iP))
    mPs= Array{Float64}(undef,(iP,iP))
    mFi= Array{Float64}(undef,(iN,iN))

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))

    # Transpose
    mZt= mZ'

    # Kalman state smoother
    @inbounds for t in iT:-1:1
        mPs= view(mP,:,(t-1)*iP+1:t*iP)
        mFi= inv(view(mF,:,(t-1)*iN+1:t*iN))
        # Kalman gain
        mK= mT*mPs*mZt*mFi
        mL= mT - mK*mZ
        # Backward recursion smoothed state
        vr= mZt*mFi*view(mv,:,t) + mL'*vr
        # Backward recursion smoothed state variance
        mN= mZt*mFi*mZ + mL'*mN*mL
        # Smoothed state
        mah[:,t]= view(ma,:,t) + mPs*vr
        # Smoothed state variance
        mV[:,(t-1)*iP+1:t*iP]= mPs - mPs*mN*mPs
    end

    return (mah, mV)
end

function KalmanSmoother!(mah, mV, mZ, mT, ma, mP, mv, mF)
    """
    Purpose:
        Implement the Kalman state smoother algorithm to obtain smoothed states
        and smoothed states variance

    Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT vector, predicted state vector
        mP          iP x iT*iP matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT*iN matrix, forecast error variance

    Return value:
        mah         iP x iT matrix, predicted state vectors mean
        mV          iP x iT*iP matrix, predicted state vectors variance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize temp. containers
    mK= Array{Float64}(undef,(iP,iP))
    mL= Array{Float64}(undef,(iP,iP))
    mPs= Array{Float64}(undef,(iP,iP))
    mFi= Array{Float64}(undef,(iN,iN))

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))

    # Transpose
    mZt= mZ'

    # Kalman state smoother
    @inbounds for t in iT:-1:1
        mPs= view(mP,:,(t-1)*iP+1:t*iP)
        mFi= inv(view(mF,:,(t-1)*iN+1:t*iN))
        # Kalman gain
        mK= mT*mPs*mZt*mFi
        mL= mT - mK*mZ
        # Backward recursion smoothed state
        vr= mZt*mFi*view(mv,:,t) + mL'*vr
        # Backward recursion smoothed state variance
        mN= mZt*mFi*mZ + mL'*mN*mL
        # Smoothed state
        mah[:,t]= view(ma,:,t) + mPs*vr
        # Smoothed state variance
        mV[:,(t-1)*iP+1:t*iP]= mPs - mPs*mN*mPs
    end

    nothing
end

function KalmanSmootherEq(mZ, mT, ma, mP, mv, mF)
    """
    Purpose:
        Implement the Kalman state smoother algorithm equation-by-equation to
        obtain smoothed states and smoothed states variance

    Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT*iN vector, predicted state vector
        mP          iP x iT*iP*iN matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance

    Return value:
        mah         iP x iT matrix, predicted state vectors mean
        mV          iP x iT*iP matrix, predicted state vectors variance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize return containers
    mah= Array{Float64}(undef,(iP,iT))
    mV= Array{Float64}(undef,(iP,iT*iP))

    # Initialize temp. containers
    vK= Array{Float64}(undef,iP)
    mL= Array{Float64}(undef,(iP,iP))
    mPs= Array{Float64}(undef,(iP,iP))

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # Transpose
    mZt= mZ'
    mTt= mT'

    # Kalman state smoother
    @inbounds for t in iT:-1:1
        # Stepsize
        iSt= (t-1)*iN*iP
        for i in iN:-1:1
            # Kalman gain
            vK= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt)*view(mZt,:,i)/mF[i,t]
            mL= mI - vK*view(mZt,:,i)'
            # Backward recursion smoothed state
            vr= view(mZt,:,i)*mv[i,t]/mF[i,t] + mL'*vr
            # Backward recursion smoothed state variance
            mN= view(mZt,:,i)*view(mZt,:,i)'/mF[i,t] + mL'*mN*mL
            if (i == 1)
                mPs= view(mP,:,iSt+1:iSt+iP)
                # Smoothed state
                mah[:,t]= view(ma,:,(t-1)*iN+1) + mPs*vr
                # Smoothed state variance
                mV[:,(t-1)*iP+1:t*iP]= mPs - mPs*mN*mPs
                if (t > 1)
                    # Backward recursion smoothed state
                    vr= mTt*vr
                    # Backward recursion smoothed state variance
                    mN= mTt*mN*mT
                end
            end
        end
    end

    return (mah, mV)
end

function KalmanSmootherEq!(mah, mV, mZ, mT, ma, mP, mv, mF)
    """
    Purpose:
        Implement the Kalman state smoother algorithm equation-by-equation to
        obtain smoothed states and smoothed states variance

    Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT*iN vector, predicted state vector
        mP          iP x iT*iP*iN matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance

    Return value:
        mah         iP x iT matrix, predicted state vectors mean
        mV          iP x iT*iP matrix, predicted state vectors variance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize temp. containers
    vK= Array{Float64}(undef,iP)
    mL= Array{Float64}(undef,(iP,iP))
    mPs= Array{Float64}(undef,(iP,iP))

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # Transpose
    mZt= mZ'
    mTt= mT'

    # Kalman state smoother
    @inbounds for t in iT:-1:1
        # Stepsize
        iSt= (t-1)*iN*iP
        for i in iN:-1:1
            # Kalman gain
            vK= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt)*view(mZt,:,i)/mF[i,t]
            mL= mI - vK*view(mZt,:,i)'
            # Backward recursion smoothed state
            vr= view(mZt,:,i)*mv[i,t]/mF[i,t] + mL'*vr
            # Backward recursion smoothed state variance
            mN= view(mZt,:,i)*view(mZt,:,i)'/mF[i,t] + mL'*mN*mL
            if (i == 1)
                mPs= view(mP,:,iSt+1:iSt+iP)
                # Smoothed state
                mah[t,:]= view(ma,:,(t-1)*iN+1) + mPs*vr
                # Smoothed state variance
                mV[:,(t-1)*iP+1:t*iP]= mPs - mPs*mN*mPs
                if (t > 1)
                    # Backward recursion smoothed state
                    vr= mTt*vr
                    # Backward recursion smoothed state variance
                    mN= mTt*mN*mT
                end
            end
        end
    end

    nothing
end

function KalmanSmootherCov(mZ, mT, ma, mP, mv, mF)
    """
    Purpose:
        Implement the Kalman state smoother algorithm to obtain smoothed states,
        smoothed states variance, and smoothed states autocovariance

    Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT vector, predicted state vector
        mP          iP x iT*iP matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT*iN matrix, forecast error variance

    Return value:
        mah         iP x iT matrix, predicted state vectors mean
        mV          iP x iT*iP matrix, predicted state vectors variance
        mVcov       iP x (iT-1)*iP matrix, predicted state vectors autocovariance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize return containers
    mah= Array{Float64}(undef,(iP,iT))
    mV= Array{Float64}(undef,(iP,iT*iP))
    mVcov= Array{Float64}(undef,(iP,(iT-1)*iP))

    # Initialize temp. containers
    mK= Array{Float64}(undef,(iP,iP))
    mL= Array{Float64}(undef,(iP,iP))
    mPs= Array{Float64}(undef,(iP,iP))
    mFi= Array{Float64}(undef,(iN,iN))

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # Transpose
    mZt= mZ'

    # Kalman state smoother
    @inbounds for t in iT:-1:1
        mPs= view(mP,:,(t-1)*iP+1:t*iP)
        mFi= inv(view(mF,:,(t-1)*iN+1:t*iN))
        # Kalman gain
        mK= mT*mPs*mZt*mFi
        mL= mT - mK*mZ
        # Backward recursion smoothed state
        vr= mZt*mFi*view(mv,:,t) + mL'*vr
        # Smoothed state
        mah[:,t]= view(ma,:,t) + mPs*vr
        if (t < iT)
            # Smoothed state covariance
            mVcov[:,(t-1)*iP+1:t*iP]= (mI - view(mP,:,t*iP+1:(t+1)*iP)*mN)*mL*mPs
        end
        # Backward recursion smoothed state variance
        mN= mZt*mFi*mZ + mL'*mN*mL
        # Smoothed state variance
        mV[:,(t-1)*iP+1:t*iP]= mPs - mPs*mN*mPs
    end

    return (mah, mV, mVcov)
end

function KalmanSmootherCov!(mah, mV, mVcov, mZ, mT, ma, mP, mv, mF)
    """
    Purpose:
        Implement the Kalman state smoother algorithm to obtain smoothed states,
        smoothed states variance, and smoothed states autocovariance

        Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT vector, predicted state vector
        mP          iP x iT*iP matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT*iN matrix, forecast error variance

    Return value:
        mah         iP x iT matrix, predicted state vectors mean
        mV          iP x iT*iP matrix, predicted state vectors variance
        mVcov       iP x (iT-1)*iP matrix, predicted state vectors autocovariance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize temp. containers
    mK= Array{Float64}(undef,(iP,iP))
    mL= Array{Float64}(undef,(iP,iP))
    mPs= Array{Float64}(undef,(iP,iP))
    mFi= Array{Float64}(undef,(iN,iN))

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # Transpose
    mZt= mZ'

    # Kalman state smoother
    @inbounds for t in iT:-1:1
        mPs= view(mP,:,(t-1)*iP+1:t*iP)
        mFi= inv(view(mF,:,(t-1)*iN+1:t*iN))
        # Kalman gain
        mK= mT*mPs*mZt*mFi
        mL= mT - mK*mZ
        # Backward recursion smoothed state
        vr= mZt*mFi*view(mv,:,t) + mL'*vr
        # Smoothed state
        mah[:,t]= view(ma,:,t) + mPs*vr
        if (t < iT)
            # Smoothed state covariance
            mVcov[:,(t-1)*iP+1:t*iP]= (mI - view(mP,:,t*iP+1:(t+1)*iP)*mN)*mL*mPs
        end
        # Backward recursion smoothed state variance
        mN= mZt*mFi*mZ + mL'*mN*mL
        # Smoothed state variance
        mV[:,(t-1)*iP+1:t*iP]= mPs - mPs*mN*mPs
    end

    nothing
end

function KalmanSmootherCov!(mah, mV, mVcov, mZ, mT::Diagonal{Float64,Vector{Float64}}, ma, mP, mv, mFi, mK)
    """
    Purpose:
        Implement the Kalman state smoother algorithm to obtain smoothed states,
        smoothed states variance, and smoothed states autocovariance

        Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT vector, predicted state vector
        mP          iP x iT*iP matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mFi         iN x iT*iN matrix, inverse forecast error variance
        mK          iP x iT*iN matrix, Kalman gain

    Return value:
        mah         iP x iT matrix, predicted state vectors mean
        mV          iP x iT*iP matrix, predicted state vectors variance
        mVcov       iP x (iT-1)*iP matrix, predicted state vectors autocovariance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize temp. containers
    mTemp= Array{Float64}(undef,(iP,iN))
    mTemp1= Array{Float64}(undef,(iP,iP))
    mTemp2= Array{Float64}(undef,(iP,iP))
    vTemp= Array{Float64}(undef,iP)

    # L matrix container
    mL= Array{Float64}(undef,(iP,iP))

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))

    # Kalman state smoother
    @inbounds @fastmath for t in iT:-1:1
        # Store views
        va= view(ma,:,t)
        vah= view(mah,:,t)
        mPs= view(mP,:,(t-1)*iP+1:t*iP)
        mVs= view(mV,:,(t-1)*iP+1:t*iP)
        mFis= view(mFi,:,(t-1)*iN+1:t*iN)
        mKs= view(mK,:,(t-1)*iN+1:t*iN)

        # L matrix
        # -Kₜ×Z
        mul!(mL, mKs, mZ, -1., .0)
        # Lₜ = T - Kₜ×Z
        for p in 1:iP
            mL[p,p]+= mT[p,p]
        end

        # Backward recursion smoothed state
        # Lₜ′×rₜ
        BLAS.gemv!('T', 1., mL, vr, .0, vTemp)
        # Z′×Fₜ⁻¹
        BLAS.gemm!('T', 'N', 1., mZ, mFis, .0, mTemp)
        # Z′×Fₜ⁻¹×vₜ
        mul!(vr, mTemp, view(mv,:,t))
        # rₜ₋₁ = Z′×Fₜ⁻¹×vₜ + Lₜ′×rₜ
        axpby!(1., vTemp, 1., vr)

        # Smoothed state
        # Pₜ×rₜ₋₁
        mul!(vah, mPs, vr)
        # α̂ₜ + Pₜ×rₜ₋₁
        axpby!(1., va, 1., vah)

        # Smoothed state covariance
        if (t < iT)
            # Store views
            mVcovs= view(mVcov,:,(t-1)*iP+1:t*iP)
            mPs1= view(mP,:,t*iP+1:(t+1)*iP)

            # Lₜ×Pₜ
            mul!(mTemp1, mL, mPs)
            # -Pₜ₊₁×Nₜ
            mul!(mTemp2, mPs1, mN, -1., .0)
            # I - Pₜ₊₁×Nₜ
            for p in 1:iP
                mTemp2[p,p]+= 1.
            end
            # Vᵗₜ₊₁ = (I - Pₜ₊₁×Nₜ)×Lₜ×Pₜ
            mul!(mVcovs, mTemp2, mTemp1)
        end

        # Backward recursion smoothed state variance
        # Lₜ′×Nₜ
        BLAS.gemm!('T', 'N', 1., mL, mN, .0, mTemp1)
        # Lₜ′×Nₜ×Lₜ
        mul!(mN, mTemp1, mL)
        # Nₜ₋₁ = Z′×Fₜ⁻¹×Zₜ + Lₜ′×Nₜ×Lₜ
        mul!(mN, mTemp, mZ, 1., 1.)

        # Smoothed state variance
        # -Nₜ₋₁×Pₜ
        mul!(mTemp1, mN, mPs, -1., .0)
        # I - Nₜ₋₁×Pₜ
        for p in 1:iP
            mTemp1[p,p]+= 1.
        end
        # Vₜ = Pₜ×(I - Nₜ₋₁×Pₜ)
        mul!(mVs, mPs, mTemp1)
    end

    nothing
end

function KalmanSmootherCovEq(mZ, mT, ma, mP, mv, mF)
    """
    Purpose:
        Implement the Kalman state smoother algorithm equation-by-equation to
        obtain smoothed states, smoothed states variance, and smoothed states
        autocovariance

    Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT*iN vector, predicted state vector
        mP          iP x iT*iP*iN matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance

    Return value:
        mah       iP x iT matrix, predicted state vectors mean
        mV        iP x iT*iP matrix, predicted state vectors variance
        mVcov     iP x (iT-1)*iP matrix, predicted state vectors autocovariance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize return containers
    mah= Array{Float64}(undef,(iP,iT))
    mV= Array{Float64}(undef,(iP,iT*iP))
    mVcov= Array{Float64}(undef,(iP,(iT-1)*iP))

    # Initialize temp. containers
    mLp= Array{Float64}(undef,(iP,iP))
    mL= Array{Float64}(undef,(iP,iP))
    vK= Array{Float64}(undef,iP)
    mPs= Array{Float64}(undef,(iP,iP))
    mPs1= Array{Float64}(undef,(iP,iP))

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))
    mNf= zeros(Float64,(iP,iP))

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # Transpose
    mZt= mZ'
    mTt= mT'

    # Kalman state smoother
    @inbounds for t in iT:-1:1
        # Stepsize
        iSt= (t-1)*iN*iP
        # Initialize product of L matrices
        mLp= mI
        for i in iN:-1:1
            mPs= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt) 
            # Kalman gain
            vK= mPs*view(mZt,:,i)/mF[i,t]
            mL= mI - vK*view(mZt,:,i)'
            # Update product of L matrices
            mLp= mLp*mL
            # Backward recursion smoothed state
            vr= view(mZt,:,i)*mv[i,t]/mF[i,t] + mL'*vr
            # Backward recursion smoothed state variance
            mN= view(mZt,:,i)*view(mZt,:,i)'/mF[i,t] + mL'*mN*mL
            if (i == 1)
                # Smoothed state
                mah[:,t]= view(ma,:,(t-1)*iN+1) + mPs*vr
                # Smoothed state variance
                mV[:,(t-1)*iP+1:t*iP]= mPs - mPs*mN*mPs
                if (t < iT)
                    # Stepsize
                    iSt1= t*iN*iP
                    mPs1= mP[:,iSt1+1:iSt1+iP]
                    # Smoothed state covariance
                    mVcov[:,(t-1)*iP+1:t*iP]= (mI - mPs1*mNf)*mT*mLp*mPs
                end
                # Store N
                copyto!(mNf,mN)
                if (t > 1)
                    # Backward recursion smoothed state
                    vr= mTt*vr
                    # Backward recursion smoothed state variance
                    mN= mTt*mN*mT
                end
            end
        end
    end

    return (mah, mV, mVcov)
end

function KalmanSmootherCovEq!(mah, mV, mVcov, mZ, mT, ma, mP, mv, mF)
    """
    Purpose:
        Implement the Kalman state smoother algorithm equation-by-equation to
        obtain smoothed states, smoothed states variance, and smoothed states
        autocovariance

        Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT*iN vector, predicted state vector
        mP          iP x iT*iP*iN matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance

    Return value:
        mah       iP x iT matrix, predicted state vectors mean
        mV        iP x iT*iP matrix, predicted state vectors variance
        mVcov     iP x (iT-1)*iP matrix, predicted state vectors autocovariance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize temp. containers
    mLp= Array{Float64}(undef,(iP,iP))
    mL= Array{Float64}(undef,(iP,iP))
    vK= Array{Float64}(undef,iP)
    mPs= Array{Float64}(undef,(iP,iP))
    mPs1= Array{Float64}(undef,(iP,iP))

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))
    mNf= zeros(Float64,(iP,iP))

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # Transpose
    mZt= mZ'
    mTt= mT'

    # Kalman state smoother
    @inbounds for t in iT:-1:1
        # Stepsize
        iSt= (t-1)*iN*iP
        # Initialize product of L matrices
        mLp= mI
        for i in iN:-1:1
            mPs= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt) 
            # Kalman gain
            vK= mPs*view(mZt,:,i)/mF[i,t]
            mL= mI - vK*view(mZt,:,i)'
            # Update product of L matrices
            mLp= mLp*mL
            # Backward recursion smoothed state
            vr= view(mZt,:,i)*mv[i,t]/mF[i,t] + mL'*vr
            # Backward recursion smoothed state variance
            mN= view(mZt,:,i)*view(mZt,:,i)'/mF[i,t] + mL'*mN*mL
            if (i == 1)
                # Smoothed state
                mah[:,t]= view(ma,:,(t-1)*iN+1) + mPs*vr
                # Smoothed state variance
                mV[:,(t-1)*iP+1:t*iP]= mPs - mPs*mN*mPs
                if (t < iT)
                    # Stepsize
                    iSt1= t*iN*iP
                    mPs1= mP[:,iSt1+1:iSt1+iP]
                    # Smoothed state covariance
                    mVcov[:,(t-1)*iP+1:t*iP]= (mI - mPs1*mNf)*mT*mLp*mPs
                end
                # Store N
                copyto!(mNf,mN)
                if (t > 1)
                    # Backward recursion smoothed state
                    vr= mTt*vr
                    # Backward recursion smoothed state variance
                    mN= mTt*mN*mT
                end
            end
        end
    end

    nothing
end