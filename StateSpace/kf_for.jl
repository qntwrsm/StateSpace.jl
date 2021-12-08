"""
kf_for.jl

Purpose:
    Kalman filter in matrix form and equation-by-equation in for-loop
    "C++/Fortran style" programming

Version:
    0       Based on kf_for.py (only eq-by-eq)

Date:
    2020/07/27

@author: wqt200
"""
function KalmanFilterEq_for(mY, mZ, mT, mH, mQ, mR, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm equation-by-equation to obtain
        predicted states, predicted states variance, forecasts errors, and
        forecast errors variances by means of for-loop "C++/Fortran style"
        programming

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
        ma          iP x iT*iN+1 matrix, predicted state vectors mean
        mP          iP x iP*(iT*iN+1) matrix, predcited state vectors variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance
    """
    (iN,iT)= size(mY)
    iP= length(va1)
    
    # Initialize return containers
    ma= Array{Float64}(undef,(iP,iT*iN+1))
    mP= Array{Float64}(undef,(iP,iP*(iT*iN+1)))
    mv= Array{Float64}(undef,(iN,iT))
    mF= Array{Float64}(undef,(iN,iT))

    # Initialize temp. containers
    vTemp= Array{Float64}(undef,iP)

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # Transpose
    mZt= transpose(mZ)
    mTt= transpose(mT)
    mRt= transpose(mR)

    # RQR matrix product
    mRQR= Array{Float64}(undef,(iP,iP))
    @inbounds @fastmath for i in 1:iP
        vRi= view(mRt,:,i)
        for j in 1:iP
            vRj= view(mRt,:,j)
            mRQR[j,i]= dot(vRj, mQ, vRi)
        end
    end

    # Kalman filter equation-by-equation
    @inbounds @fastmath for t in 1:iT
        # Stepsize
        iSt= (t-1)*iN*iP
        for i in 1:iN
            # Initialize temp. containers
            dv= mY[i,t]
            dF= mH[i,i]

            # # Kalman gain and forecast error variance
            for j in 1:iP
                # Initialize temp. container
                dK= .0
                # Temp. container
                dZt= mZt[j,i]
                # Forecast error
                dv-= dZt*ma[j,i+(t-1)*iN]
                for k in 1:iP
                    # Kalman gain
                    dK+= mP[k,(i-1)*iP+iSt+j]*mZt[k,i]
                end
                # Forecast error variance
                dF+= dZt*dK
                # Store result
                vTemp[j]= dK
            end

            # Store results
            mv[i,t]= dv
            mF[i,t]= dF
            dvF= dv*inv(dF)

            # Non-transition
            for j in 1:iP
                # Temp. container
                dK= vTemp[j]
                # Predicted state vector
                ma[j,i+1+(t-1)*iN]= ma[j,i+(t-1)*iN] + dK*dvF
                for k in 1:iP
                    # Predicted state vector variance
                    mP[k,i*iP+iSt+j]= mP[k,(i-1)*iP+iSt+j] - vTemp[k]*dK*inv(dF)
                end
            end
        end

        # Transition
        # Stepsize
        iSt1= t*iN*iP
        for j in 1:iP
            # Initialize temp. container
            da= .0
            for k in 1:iP
                # Initialize temp. container
                dP= mRQR[k,j]
                # Predicted state vector
                da+= mTt[k,j]*ma[k,t*iN+1]
                for l in 1:iP
                    # Temp. container
                    dTt= mTt[l,k]
                    for m in 1:iP
                        # Predicted state vector variance
                        dP+= dTt*mP[m,iSt1+l]*mT[m,j]
                    end
                end
                # Store result
                mP1[k,j]= dP
            end
            # Store result
            vTemp[j]= da
        end

        # Copy results
        for j in 1:iP
            ma[j,t*iN+1]= vTemp[j]
            for k in 1:iP
                mP[k,iSt1+j]= mP1[k,j]
            end
        end
    end

    return (ma, mP, mv, mF)
end

function KalmanFilterEq_for!(ma, mP, mv, mF, mY, mZ, mT, mH, mQ, mR, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm equation-by-equation to obtain
        predicted states, predicted states variance, forecasts errors, and
        forecast errors variances by means of for-loop "C++/Fortran style"
        programming

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
        ma          iP x iT*iN+1 matrix, predicted state vectors mean
        mP          iP x iP*(iT*iN+1) matrix, predcited state vectors variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance
    """
    (iN,iT)= size(mY)
    iP= length(va1)

    # Initialize temp. containers
    vTemp= zeros(Float64, iP) 

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # Transpose
    mZt= transpose(mZ)
    mTt= transpose(mT)
    mRt= transpose(mR)

    # RQR matrix product
    mRQR= Array{Float64}(undef,(iP,iP))
    @inbounds @fastmath for i in 1:iP
        vRi= view(mRt,:,i)
        for j in 1:iP
            vRj= view(mRt,:,j)
            mRQR[j,i]= dot(vRj, mQ, vRi)
        end
    end

    # Kalman filter equation-by-equation
    @inbounds @fastmath for t in 1:iT
        # Stepsize
        iSt= (t-1)*iN*iP
        for i in 1:iN
            # Initialize temp. containers
            dv= mY[i,t]
            dF= mH[i,i]

            # # Kalman gain and forecast error variance
            for j in 1:iP
                # Initialize temp. container
                dK= .0
                # Temp. container
                dZt= mZt[j,i]
                # Forecast error
                dv-= dZt*ma[j,i+(t-1)*iN]
                for k in 1:iP
                    # Kalman gain
                    dK+= mP[k,(i-1)*iP+iSt+j]*mZt[k,i]
                end
                # Forecast error variance
                dF+= dZt*dK
                # Store result
                vTemp[j]= dK
            end

            # Store results
            mv[i,t]= dv
            mF[i,t]= dF
            dvF= dv*inv(dF)

            # Non-transition
            for j in 1:iP
                # Temp. container
                dK= vTemp[j]
                # Predicted state vector
                ma[j,i+1+(t-1)*iN]= ma[j,i+(t-1)*iN] + dK*dvF
                for k in 1:iP
                    # Predicted state vector variance
                    mP[k,i*iP+iSt+j]= mP[k,(i-1)*iP+iSt+j] - vTemp[k]*dK*inv(dF)
                end
            end
        end

        # Transition
        # Stepsize
        iSt1= t*iN*iP
        for j in 1:iP
            # Initialize temp. container
            da= .0
            for k in 1:iP
                # Initialize temp. container
                dP= mRQR[k,j]
                # Predicted state vector
                da+= mTt[k,j]*ma[k,t*iN+1]
                for l in 1:iP
                    # Temp. container
                    dTt= mTt[l,k]
                    for m in 1:iP
                        # Predicted state vector variance
                        dP+= dTt*mP[m,iSt1+l]*mT[m,j]
                    end
                end
                # Store result
                mP1[k,j]= dP
            end
            # Store result
            vTemp[j]= da
        end

        # Copy results
        for j in 1:iP
            ma[j,t*iN+1]= vTemp[j]
            for k in 1:iP
                mP[k,iSt1+j]= mP1[k,j]
            end
        end
    end

    nothing
end

function KalmanFilterEq_for!(ma, mP, mv, mF, mY, mZ, mT::Diagonal{Float64,Vector{Float64}}, mH::Diagonal{Float64,Vector{Float64}}, mQ::Diagonal{Float64,Vector{Float64}}, mR, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm equation-by-equation to obtain
        predicted states, predicted states variance, forecasts errors, and
        forecast errors variances by means of for-loop "C++/Fortran style"
        programming

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
        ma          iP x iT*iN+1 matrix, predicted state vectors mean
        mP          iP x iP*(iT*iN+1) matrix, predcited state vectors variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance
    """
    (iN,iT)= size(mY)
    iP= length(va1)

    # Initialize temp. containers
    vK= zeros(Float64, iP) 

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # Transpose
    mZt= transpose(mZ)
    mRt= transpose(mR)

    # RQR matrix product
    mRQR= Array{Float64}(undef,(iP,iP))
    @inbounds @fastmath for i in 1:iP
        for j in 1:iP
            dRQR= .0
            for k in 1:iP
                dRQR+= mRt[k,j]*mQ[k,k]*mRt[k,i]
            end
            mRQR[j,i]= dRQR
        end
    end

    # Kalman filter equation-by-equation
    @inbounds @fastmath for t in 1:iT
        # Stepsize
        iSt= (t-1)*iN*iP
        for i in 1:iN
            # Initialize temp. containers
            dv= mY[i,t]
            dF= mH[i,i]

            # Kalman gain and forecast error variance
            for j in 1:iP
                # Initialize temp. container
                dK= .0
                # Temp. container
                dZt= mZt[j,i]
                # Forecast error
                dv-= dZt*ma[j,i+(t-1)*iN]
                for k in 1:iP
                    # Kalman gain
                    dK+= mP[k,(i-1)*iP+iSt+j]*mZt[k,i]
                end
                # Forecast error variance
                dF+= dZt*dK
                # Store result
                vK[j]= dK
            end

            # Store results
            mv[i,t]= dv
            mF[i,t]= dF
            dvF= dv*inv(dF)

            # Non-transition
            for j in 1:iP
                # Temp. container
                dK= vK[j]
                # Predicted state vector
                ma[j,i+1+(t-1)*iN]= ma[j,i+(t-1)*iN] + dK*dvF
                for k in 1:iP
                    # Predicted state vector variance
                    mP[k,i*iP+iSt+j]= mP[k,(i-1)*iP+iSt+j] - vK[k]*dK*inv(dF)
                end
            end
        end

        # Transition
        # Stepsize
        iSt1= t*iN*iP
        for j in 1:iP
            dT= mT[j,j]
            # Predicted state vector
            ma[j,t*iN+1]= dT*ma[j,t*iN+1]
            for k in 1:iP
                # Predicted state vector variance
                mP[k,iSt1+j]= mRQR[k,j] + mT[k,k]*mP[k,iSt1+j]*dT
            end
        end
    end

    nothing
end

function KalmanFilterEq_for!(ma, mP, mv, mF, mY, mZ, mT, mH, mQ, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm equation-by-equation to obtain
        predicted states, predicted states variance, forecasts errors, and
        forecast errors variances by means of for-loop "C++/Fortran style"
        programming

    Inputs:
        mY          iN x iT matrix, data
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
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
    vTemp= zeros(Float64, iP) 

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # Transpose
    mZt= transpose(mZ)
    mTt= transpose(mT)

    # Kalman filter equation-by-equation
    @inbounds @fastmath for t in 1:iT
        # Stepsize
        iSt= (t-1)*iN*iP
        for i in 1:iN
            # Initialize temp. containers
            dv= mY[i,t]
            dF= mH[i,i]

            # # Kalman gain and forecast error variance
            for j in 1:iP
                # Initialize temp. container
                dK= .0
                # Temp. container
                dZt= mZt[j,i]
                # Forecast error
                dv-= dZt*ma[j,i+(t-1)*iN]
                for k in 1:iP
                    # Kalman gain
                    dK+= mP[k,(i-1)*iP+iSt+j]*mZt[k,i]
                end
                # Forecast error variance
                dF+= dZt*dK
                # Store result
                vTemp[j]= dK
            end

            # Store results
            mv[i,t]= dv
            mF[i,t]= dF
            dvF= dv*inv(dF)

            # Non-transition
            for j in 1:iP
                # Temp. container
                dK= vTemp[j]
                # Predicted state vector
                ma[j,i+1+(t-1)*iN]= ma[j,i+(t-1)*iN] + dK*dvF
                for k in 1:iP
                    # Predicted state vector variance
                    mP[k,i*iP+iSt+j]= mP[k,(i-1)*iP+iSt+j] - vTemp[k]*dK*inv(dF)
                end
            end
        end

        # Transition
        # Stepsize
        iSt1= t*iN*iP
        for j in 1:iP
            # Initialize temp. container
            da= .0
            for k in 1:iP
                # Initialize temp. container
                dP= mQ[k,j]
                # Predicted state vector
                da+= mTt[k,j]*ma[k,t*iN+1]
                for l in 1:iP
                    # Temp. container
                    dTt= mTt[l,k]
                    for m in 1:iP
                        # Predicted state vector variance
                        dP+= dTt*mP[m,iSt1+l]*mT[m,j]
                    end
                end
                # Store result
                mP1[k,j]= dP
            end
            # Store result
            vTemp[j]= da
        end

        # Copy results
        for j in 1:iP
            ma[j,t*iN+1]= vTemp[j]
            for k in 1:iP
                mP[k,iSt1+j]= mP1[k,j]
            end
        end
    end

    nothing
end

function KalmanFilterEq_for!(ma, mP, mv, mF, mY, mZ, mT::Diagonal{Float64,Vector{Float64}}, mH::Diagonal{Float64,Vector{Float64}}, mQ::Diagonal{Float64,Vector{Float64}}, va1, mP1)
    """
    Purpose:
        Implement the Kalman filter algorithm equation-by-equation to obtain
        predicted states, predicted states variance, forecasts errors, and
        forecast errors variances by means of for-loop "C++/Fortran style"
        programming

    Inputs:
        mY          iN x iT matrix, data
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
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
    vK= zeros(Float64, iP) 

    # Initialize the filter
    ma[:,1]= va1
    mP[:,1:iP]= mP1

    # Transpose
    mZt= transpose(mZ)

    # Kalman filter equation-by-equation
    @inbounds @fastmath for t in 1:iT
        # Stepsize
        iSt= (t-1)*iN*iP
        for i in 1:iN
            # Initialize temp. containers
            dv= mY[i,t]
            dF= mH[i,i]

            # Kalman gain and forecast error variance
            for j in 1:iP
                # Initialize temp. container
                dK= .0
                # Temp. container
                dZt= mZt[j,i]
                # Forecast error
                dv-= dZt*ma[j,i+(t-1)*iN]
                for k in 1:iP
                    # Kalman gain
                    dK+= mP[k,(i-1)*iP+iSt+j]*mZt[k,i]
                end
                # Forecast error variance
                dF+= dZt*dK
                # Store result
                vK[j]= dK
            end

            # Store results
            mv[i,t]= dv
            mF[i,t]= dF
            dvF= dv*inv(dF)

            # Non-transition
            for j in 1:iP
                # Temp. container
                dK= vK[j]
                # Predicted state vector
                ma[j,i+1+(t-1)*iN]= ma[j,i+(t-1)*iN] + dK*dvF
                for k in 1:iP
                    # Predicted state vector variance
                    mP[k,i*iP+iSt+j]= mP[k,(i-1)*iP+iSt+j] - vK[k]*dK*inv(dF)
                end
            end
        end

        # Transition
        # Stepsize
        iSt1= t*iN*iP
        for j in 1:iP
            dT= mT[j,j]
            # Predicted state vector
            ma[j,t*iN+1]= dT*ma[j,t*iN+1]
            for k in 1:iP
                # Predicted state vector variance
                mP[k,iSt1+j]= mT[k,k]*mP[k,iSt1+j]*dT
            end
            mP[j,iSt1+j]+= mQ[j,j]
        end
    end

    nothing
end