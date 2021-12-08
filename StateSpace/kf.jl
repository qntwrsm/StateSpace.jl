"""
kf.py

Purpose:
    Kalman filter in matrix form and equation-by-equation

Version:
    0       Based on kf.py

Date:
    2020/07/28

@author: wqt200
"""
# Include numerical helper routines
include("../misc/num.jl")

function KalmanFilter(mY, mZ, mT, mR, mH, mQ, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm to obtain predicted states,
        predicted states variance, forecasts errors, and forecast errors
        variances

    Inputs:
        mY          iN x iT matrix, data
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        mR          iP x iP matrix, system matrix R
        mH          iN x iN matrix, system matrix H
        mQ          iP x iP matrix, system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance

    Return value:
        ma          iP x iT matrix, predicted state vectors mean
        mP          iP x iT*iP matrix, predcited state vectors variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT*iN matrix, forecast error variance
    """
    (iN,iT)= size(mY)
    iP= length(va1)

    # Initialize return containers
    ma= zeros(Float64,(iP,iT))
    mP= zeros(Float64,(iP,iT*iP))
    mv= zeros(Float64,(iN,iT))
    mF= zeros(Float64,(iN,iT*iN))

    # Initialize temp. containers
    mK= Array{Float64}(undef,(iP,iP))
    mPs= Array{Float64}(undef,(iP,iP))

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # RQR matrix product
    mRQR= mR*mQ*mR'

    # Transpose
    mZt= mZ'
    mTt= mT'

    # Kalman filter
    @inbounds for t in 1:iT
        mPs= view(mP,:,(t-1)*iP+1:t*iP)
        # Forecast error
        mv[:,t]= view(mY,:,t) - mZ*view(ma,:,t)
        # Forecast error variance
        mF[:,(t-1)*iN+1:t*iN]= mZ*mPs*mZt + mH
        # Kalman gain
        mK= mT*mPs*mZt*inv(view(mF,:,(t-1)*iN+1:t*iN))
        if (t < iT)
            # Predicted state vector
            ma[:,t+1]= mT*view(ma,:,t) + mK*view(mv,:,t)
            # Predicted state vector variance
            mP[:,t*iP+1:(t+1)*iP]= mT*mPs*(mTt - mZt*mK') + mRQR
        end
    end       

    return (ma, mP, mv, mF)
end

function KalmanFilter!(ma, mP, mv, mF, mY, mZ, mT, mR, mH, mQ, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm to obtain predicted states,
        predicted states variance, forecasts errors, and forecast errors
        variances

    Inputs:
        mY          iN x iT matrix, data
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        mR          iP x iP matrix, system matrix R
        mH          iN x iN matrix, system matrix H
        mQ          iP x iP matrix, system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance

    Return value:
        ma          iP x iT matrix, predicted state vectors mean
        mP          iP x iT*iP matrix, predcited state vectors variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT*iN matrix, forecast error variance
    """
    (iN,iT)= size(mY)
    iP= length(va1)

    # Initialize temp. containers
    mK= Array{Float64}(undef,(iP,iP))
    mPs= Array{Float64}(undef,(iP,iP))

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # RQR matrix product
    mRQR= mR*mQ*mR'

    # Transpose
    mZt= mZ'
    mTt= mT'

    # Kalman filter
    @inbounds for t in 1:iT
        mPs= view(mP,:,(t-1)*iP+1:t*iP)
        # Forecast error
        mv[:,t]= view(mY,:,t) - mZ*view(ma,:,t)
        # Forecast error variance
        mF[:,(t-1)*iN+1:t*iN]= mZ*mPs*mZt + mH
        # Kalman gain
        mK= mT*mPs*mZt*inv(view(mF,:,(t-1)*iN+1:t*iN))
        if (t < iT)
            # Predicted state vector
            ma[:,t+1]= mT*view(ma,:,t) + mK*view(mv,:,t)
            # Predicted state vector variance
            mP[:,t*iP+1:(t+1)*iP]= mT*mPs*(mTt - mZt*mK') + mRQR
        end
    end  
    
    nothing
end

function KalmanFilter!(ma, mP, mv, mFi, mK, mY, mZ, mT::Diagonal{Float64,Vector{Float64}}, mHi::Diagonal{Float64,Vector{Float64}}, mQ::Diagonal{Float64,Vector{Float64}}, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm to obtain predicted states,
        predicted states variance, forecasts errors, and forecast errors
        variances

    Inputs:
        mY          iN x iT matrix, data
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        mR          iP x iP matrix, system matrix R
        mHi         iN x iN matrix, inverse system matrix H
        mQ          iP x iP matrix, system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance

    Return value:
        ma          iP x iT matrix, predicted state vectors mean
        mP          iP x iT*iP matrix, predcited state vectors variance
        mv          iN x iT matrix, forecast error
        mFi         iN x iT*iN matrix, inverse forecast error variance
        mK          iP x iT*iN matrix, Kalman gain
    """
    (iN,iT)= size(mY)
    iP= length(va1)

    # Initialize temp. containers
    mTemp= Array{Float64}(undef,(iP,iN))
    mTemp1= Array{Float64}(undef,(iP,iP))
    mPi= Array{Float64}(undef,(iP,iP))

    # H⁻¹×Z
    mHiZ= mHi*mZ

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # Kalman filter
    @inbounds @fastmath for t in 1:iT
        # Store views 
        va= view(ma,:,t)
        vv= view(mv,:,t)
        mPs= view(mP,:,(t-1)*iP+1:t*iP)
        mFis= view(mFi,:,(t-1)*iN+1:t*iN)
        mKs= view(mK,:,(t-1)*iN+1:t*iN)

        # Forecast error
        # Z×aₜ
        mul!(vv, mZ, va, -1., .0)
        # vₜ = yₜ - Z×aₜ
        axpy!(1., view(mY,:,t), vv)

        # Inverse forecast error variance (Woodbury Identity)
        # Pₜ⁻¹
        inInv!(mPi, mPs)
        # Pₜ⁻¹ + Z′×H⁻¹×Z
        BLAS.gemm!('T', 'N', 1., mZ, mHiZ, 1., mPi)
        # (Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹
        inInv!(mTemp1, mPi)
        # (Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹×Z′×H⁻¹
        BLAS.gemm!('N', 'T', 1., mTemp1, mHiZ, .0, mTemp)
        # H⁻¹×Z×(Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹×Z′×H⁻¹
        mul!(mFis, mHiZ, mTemp, -1., .0)
        # Fₜ⁻¹ = H⁻¹ - H⁻¹×Z×(Pₜ⁻¹ + Z′×H⁻¹×Z)⁻¹×Z′×H⁻¹
        for i in 1:iN
            mFis[i,i]+= mHi[i,i]
        end

        # Kalman gain
        # Pₜ×Z′
        BLAS.gemm!('N', 'T', 1., mPs, mZ, .0, mTemp)
        # Pₜ×Z′×Fₜ⁻¹
        mul!(mKs, mTemp, mFis)
        # Kₜ = T×Pₜ×Z′×Fₜ⁻¹
        lmul!(mT, mKs)

        # Predicted state vector and state vector variance
        if (t < iT)
            # Store views 
            vau= view(ma,:,t+1)
            mPus= view(mP,:,t*iP+1:(t+1)*iP)

            # Kₜ×vₜ
            mul!(vau, mKs, vv)
            # Pₜ×Z′×Kₜ
            BLAS.gemm!('N', 'T', -1., mTemp, mKs, .0, mPus)
            # T×Pₜ×Z′×Kₜ
            lmul!(mT, mPus)
            # aₜ₊₁ = T×aₜ and Pₜ₊₁ = Q + T×Pₜ×T′
            for p in 1:iP
                dT= mT[p,p]
                vau[p]+= dT*va[p]
                mPus[p,p]+= mQ[p,p]
                for q in 1:iP
                    mPus[q,p]+= mT[q,q]*mPs[q,p]*dT
                end
            end
        end
    end  
    
    return nothing
end

function KalmanFilterEq(mY, mZ, mT, mR, mH, mQ, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm equation-by-equation to obtain
        predicted states, predicted states variance, forecasts errors, and
        forecast errors variances

    Inputs:
        mY          iN x iT matrix, data
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        mR          iP x iP matrix, system matrix R
        mH          iN x iN matrix, system matrix H
        mQ          iP x iP matrix, system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance

    Return value:
        ma          iP x iT*iN matrix, predicted state vectors mean
        mP          iP x iT*iP*iN matrix, predcited state vectors variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance
    """
    (iN,iT)= size(mY)
    iP= length(va1)

    # Initialize return containers
    ma= Array{Float64}(undef,(iP,iT*iN))
    mP= Array{Float64}(undef,(iP,iT*iP*iN))
    mv= Array{Float64}(undef,(iN,iT))
    mF= Array{Float64}(undef,(iN,iT))

    # Initialize temp. containers
    vK= Array{Float64}(undef,iP)
    mPs= Array{Float64}(undef,(iP,iP))

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # RQR matrix product
    mRQR= mR*mQ*mR'

    # Transpose
    mZt= mZ'
    mTt= mT'

    # Kalman filter equation-by-equation
    @inbounds for t in 1:iT
        # Stepsize
        iSt= (t-1)*iN*iP
        for i in 1:iN
            mPs= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt)
            # Forecast error
            mv[i,t]= mY[i,t] - view(mZt,:,i)'*view(ma,:,i+(t-1)*iN)
            # Forecast error variance
            mF[i,t]= view(mZt,:,i)'*mPs*view(mZt,:,i) + mH[i,i]
            # Kalman gain
            vK= mPs*view(mZt,:,i)/mF[i,t]
            if (i < iN)
                # Predicted state vector
                ma[:,i+1+(t-1)*iN]= view(ma,:,i+(t-1)*iN) + vK*mv[i,t]
                # Predicted state vector variance
                mP[:,i*iP+iSt+1:(i+1)*iP+iSt]= mPs - vK*vK'*mF[i,t]
            elseif (i == iN && t < iT)
                # Stepsize
                iSt1= t*iN*iP
                # Predicted state vector
                ma[:,t*iN+1]= mT*(view(ma,:,t*iN) + vK*mv[end,t])
                # Predicted state vector variance
                mP[:,iSt1+1:iP+iSt1]= mT*(view(mP,:,iSt1-iP+1:iSt1) - vK*vK'*mF[end,t])*mTt + mRQR
            end
        end
    end         

    return (ma, mP, mv, mF)
end

function KalmanFilterEq!(ma, mP, mv, mF, mY, mZ, mT, mH, mQ, mR, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm equation-by-equation to obtain
        predicted states, predicted states variance, forecasts errors, and
        forecast errors variances

    Inputs:
        mY          iN x iT matrix, data
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        mR          iP x iP matrix, system matrix R
        mH          iN x iN matrix, system matrix H
        mQ          iP x iP matrix, system matrix Q
        va1         iP x 1 vector, initial state vector mean
        mP1         iP x iP matrix, initial state variance

    Return value:
        ma          iP x iT*iN+1 matrix, predicted state vectors mean
        mP          iP x iP*(iT*iN+1) matrix, predcited state vectors variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance
    """
    (iN,iT)= size(mY)
    iP= length(va1)

    # Initialize temp. containers
    vK= Array{Float64}(undef,iP)
    mK2= Array{Float64}(undef,(iP,iP))
    # mPs= Array{Float64}(undef,(iP,iP))

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # RQR matrix product
    # mRQR= mR*mQ*mR'

    # Transpose
    mZt= transpose(mZ)
    mTt= transpose(mT)

    # Kalman filter equation-by-equation
    # @inbounds @fastmath for t in 1:iT
    #     # Stepsize
    #     iSt= (t-1)*iN*iP
    #     for i in 1:iN
    #         mPs= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt)
    #         # Forecast error
    #         mv[i,t]= mY[i,t] - view(mZt,:,i)'*view(ma,:,i+(t-1)*iN)
    #         # Forecast error variance
    #         mF[i,t]= view(mZt,:,i)'*mPs*view(mZt,:,i) + mH[i,i]
    #         # Kalman gain
    #         vK= mPs*view(mZt,:,i)/mF[i,t]
    #         if (i < iN)
    #             # Predicted state vector
    #             ma[:,i+1+(t-1)*iN]= view(ma,:,i+(t-1)*iN) + vK*mv[i,t]
    #             # Predicted state vector variance
    #             mP[:,i*iP+iSt+1:(i+1)*iP+iSt]= mPs - vK*vK'*mF[i,t]
    #         else
    #             # Stepsize
    #             iSt1= t*iN*iP
    #             # Predicted state vector
    #             ma[:,t*iN+1]= mT*(view(ma,:,t*iN) + vK*mv[end,t])
    #             # Predicted state vector variance
    #             mP[:,iSt1+1:iP+iSt1]= mT*(view(mP,:,iSt1-iP+1:iSt1) - vK*vK'*mF[end,t])*mTt + mQ
    #         end
    #     end
    # end  
    @inbounds @fastmath for t in 1:iT
        # Stepsize
        iSt= (t-1)*iN*iP
        for i in 1:iN
            # Forecast error
            # mv[i,t]= mY[i,t] - transpose(view(mZt,:,i))*view(ma,:,i+(t-1)*iN)
            dTemp= .0
            for p in 1:iP
                dTemp+= mZt[p,i]*ma[p,i+(t-1)*iN]
            end
            mv[i,t]= mY[i,t] - dTemp
            # Forecast error variance
            mul!(vK, view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt), view(mZt,:,i))
            # mF[i,t]= transpose(view(mZt,:,i))*vK + mH[i,i]
            dTemp= .0
            for p in 1:iP
                dTemp+= mZt[p,i]*vK[p]
            end
            mF[i,t]= dTemp + mH[i,i]
            # Kalman gain
            rmul!(vK, 1. / mF[i,t])
            # Outer product Kalman gain
            BLAS.gemm!('N', 'T', mF[i,t], vK, vK, .0, mK2)
            if (i < iN)
                # Predicted state vector
                rmul!(vK, mv[i,t])
                ma[:,i+1+(t-1)*iN]= view(ma,:,i+(t-1)*iN) .+ vK
                # ma[:,i+1+(t-1)*iN]= view(ma,:,i+(t-1)*iN) .+ vK*mv[i,t]
                # Predicted state vector variance
                mP[:,i*iP+iSt+1:(i+1)*iP+iSt]= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt) .- mK2
                # mP[:,i*iP+iSt+1:(i+1)*iP+iSt]= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt) .- vK*transpose(vK)*mF[i,t]
            else
                # Stepsize
                iSt1= t*iN*iP
                # Predicted state vector
                axpby!(1., ma[:,t*iN], mv[end,t], vK)
                # BLAS.gemv!('N', 1., mT, vK, .0, view(ma,:,t*iN+1))
                # ma[:,t*iN+1]= mT*(view(ma,:,t*iN) + vK*mv[end,t])
                # Predicted state vector variance
                axpby!(1., mP[:,iSt1-iP+1:iSt1], -1., mK2)
                for p in 1:iP
                    da= .0
                    for q in 1:iP
                        da+= mT[p,q]*vK[q]
                        dP= .0
                        for l in 1:iP
                            dP+= mTt[l,q]*mK2[l,p]
                        end
                        mP[q,iSt1+p]= dP
                    end
                    ma[p,t*iN+1]= da
                end
                # mul!(mP[:,iSt1+1:iP+iSt1], mT, mK2)
                mul!(mK2, view(mP,:,iSt1+1:iP+iSt1), mTt)
                mP[:,iSt1+1:iP+iSt1]= mK2 .+ mQ
                # mP[:,iSt1+1:iP+iSt1]= mT*(view(mP,:,iSt1-iP+1:iSt1) .- mK2)*mTt + mQ
            end
        end
    end   

    nothing        
end