"""
ks_for.jl

Purpose:
    Kalman smoother in matrix form and equation-by-equation in for-loop
    "C++/Fortran style" programming

Version:
    0       Based on ks_for.py (only eq-by-eq)

Date:
    2020/07/29

@author: wqt200
"""
function KalmanSmootherCovEq_for(mZ, mT, ma, mP, mv, mF)
    """
    Purpose:
        Implement the Kalman state smoother algorithm equation-by-equation to
        obtain smoothed states, smoothed states variance, and smoothed states
        autocovariance by means of for-loop "C++/Fortran style" programming

    Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT*iN+1 vector, predicted state vector
        mP          iP x iP*(iT*iN+1) matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance

    Return value:
        mah         iP x iT matrix, smoothed state vectors mean
        mV          iP x iT*iP matrix, smoothed state vectors variance
        mVcov       iP x (iT-1)*iP matrix, smoothed state vectors autocovariance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize return containers
    mah= Array{Float64}(undef,(iP,iT))
    mV= Array{Float64}(undef,(iP,iT*iP))
    mVcov= Array{Float64}(undef,(iP,(iT-1)*iP))

    # Initialize temp. containers
    mLp= Array{Float64}(undef,(iP,iP))
    mTemp= Array{Float64}(undef,(iP,iP))
    mL= Array{Float64}(undef,(iP,iP))
    vTemp= Array{Float64}(undef,iP)

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))
    mNf= zeros(Float64,(iP,iP))

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # Transpose
    mZt= transpose(mZ)
    mTt= transpose(mT)

    # Kalman state smoother
    @inbounds @fastmath for t in iT:-1:1
        # Stepsize
        iSt= (t-1)*iN*iP
        iSt1= t*iN*iP

        # Initialize product of L matrices
        copyto!(mLp,mI)

        # Backward smoothing recursions
        for i in iN:-1:1
            # Temp. containers
            dv= mv[i,t]
            dF= mF[i,t]

            # Kalman gain
            @inbounds @fastmath for j in 1:iP
                # Initialize temp. container
                dK= .0
                for k in 1:iP
                    dK+= mP[k,(i-1)*iP+iSt+j]*mZt[k,i]
                end
                # Store result
                vTemp[j]= dK
            end

            # L matrix
            @inbounds @fastmath for j in 1:iP
                # Temp. container
                dTemp= mZt[j,i]/dF
                for k in 1:iP
                    mL[k,j]= mI[k,j] - vTemp[k]*dTemp
                end
            end

            # Update product of L matrices
            @inbounds @fastmath for j in 1:iP
                dTemp0= mZt[j,i]/dF
                for k in 1:iP
                    dTemp= .0
                    for l in 1:iP
                        dTemp+= mLp[k,l]*mL[l,j]
                    end
                    mTemp[k,j]= dTemp
                end
            end
            # Update
            copyto!(mLp,mTemp)

            # Backward smoothing recursion
            @inbounds @fastmath for j in 1:iP
                # Initialize temp. containers
                dTemp0= .0
                dTemp1= mZt[j,i]/dF
                for k in 1:iP
                    dTemp0+= mL[k,j]*vr[k]
                    # Initialize temp. container
                    dTemp2= mZt[k,i]*dTemp1
                    for l in 1:iP
                        # Temp. container
                        dL= mL[l,j]
                        for m in 1:iP
                            dTemp2+= mL[m,k]*mN[m,l]*dL
                        end
                    end
                    mTemp[k,j]= dTemp2
                end
                vTemp[j]= dTemp0 + dTemp1*dv
            end
            # Update
            copyto!(vr,vTemp)
            copyto!(mN,mTemp)
        end
        
        # Smoothed state and smoothed state variance
        for j in 1:iP
            # Initialize temp. container
            dah= ma[j,(t-1)*iN+1]
            for k in 1:iP
                # Smoothed state
                dah+= mP[k,iSt+j]*vr[k]
                # Initialize container
                dV= mP[k,iSt+j]
                for l in 1:iP
                    # Temp. container
                    dP= mP[l,iSt+j]
                    for m in 1:iP
                        # Smoothed state variance
                        dV-= mP[m,iSt+k]*mN[m,l]*dP
                    end
                end
                # Store result
                mV[k,(t-1)*iP+j]= dV
            end
            # Store result
            mah[j,t]= dah
        end

        # Smoothed state covariance
        if (t < iT)
            for j in 1:iP
                for k in 1:iP
                    # Initialize container
                    dTemp= .0
                    for l in 1:iP
                        # Temp. container
                        dP0= mP[l,iSt+j]
                        for m in 1:iP
                            dTemp+= mTt[m,k]*mLp[m,l]*dP0
                        end
                    end
                    mTemp[k,j]= dTemp
                end
                for k in 1:iP
                    # Initialize temp. container
                    dVcov= mTemp[k,j]
                    for p in 1:iP
                        # Temp. container
                        dP1= mP[p,iSt1+k]
                        for q in 1:iP
                            # Smoothed state covariance
                            dVcov-= dP1*mNf[p,q]*mTemp[q,j]
                        end
                    end
                    # Store result
                    mVcov[k,(t-1)*iP+j]= dVcov
                end
            end
        end

        # Store N
        copyto!(mNf,mN)

        # Update backward smoothing recursion
        if (t > 1)
            for j in 1:iP
                dTemp0= .0
                for k in 1:iP
                    dTemp1= .0
                    dTemp0+= mT[k,j]*vr[k]
                    for l in 1:iP
                        # Temp. container
                        dT= mT[l,j]
                        for m in 1:iP
                            dTemp1+= mT[m,k]*mN[m,l]*dT
                        end
                    end
                    mTemp[k,j]= dTemp1
                end
                vTemp[j]= dTemp0
            end
            # Update 
            copyto!(vr,vTemp)
            copyto!(mN,mTemp)
        end
    end

    return (mah, mV, mVcov)
end

function KalmanSmootherCovEq_for!(mah::Array{Float64,2}, mV::Array{Float64,2}, mVcov::Array{Float64,2}, mZ::Array{Float64,2}, mT::Array{Float64,2}, ma::Array{Float64,2}, mP::Array{Float64,2}, mv::Array{Float64,2}, mF::Array{Float64,2})
    """
    Purpose:
        Implement the Kalman state smoother algorithm equation-by-equation to
        obtain smoothed states, smoothed states variance, and smoothed states
        autocovariance by means of for-loop "C++/Fortran style" programming

    Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT*iN+1 vector, predicted state vector
        mP          iP x iP*(iT*iN+1) matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance

    Return value:
        mah         iP x iT matrix, smoothed state vectors mean
        mV          iP x iT*iP matrix, smoothed state vectors variance
        mVcov       iP x (iT-1)*iP matrix, smoothed state vectors autocovariance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize temp. containers
    mLp= zeros(Float64,(iP,iP))
    mTemp= zeros(Float64,(iP,iP))
    mL= zeros(Float64,(iP,iP))
    vTemp= zeros(Float64,iP)

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))
    mNf= zeros(Float64,(iP,iP))

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # Transpose
    mZt= transpose(mZ)
    mTt= transpose(mT)

    # Kalman state smoother
    @inbounds @fastmath for t in iT:-1:1
        # Stepsize
        iSt= (t-1)*iN*iP
        iSt1= t*iN*iP

        # Initialize product of L matrices
        copyto!(mLp, mI)

        # Backward smoothing recursions
        for i in iN:-1:1
            # Temp. containers
            dv= mv[i,t]
            dFi= inv(mF[i,t])

            # Kalman gain
            vZ= view(mZt,:,i)
            mPs= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt)
            for j in 1:iP
                # Initialize temp. container
                dK= .0
                @simd for k in 1:iP
                    dK+= mPs[k,j]*vZ[k]
                end
                # Store result
                vTemp[j]= dK
            end

            # L matrix
            for j in 1:iP
                dZF= vZ[j]*dFi
                @simd for k in 1:iP
                    mL[k,j]= mI[k,j] - vTemp[k]*dZF
                end
            end

            # Update product of L matrices
            for j in 1:iP
                for k in 1:iP
                    dTemp= .0
                    @simd for l in 1:iP
                        dTemp+= mLp[k,l]*mL[l,j]
                    end
                    mTemp[k,j]= dTemp
                end
            end
            copyto!(mLp, mTemp)

            # Backward smoothing recursion
            for j in 1:iP
                # Initialize temp. containers
                dTemp0= .0
                dTemp1= vZ[j]*dFi
                for k in 1:iP
                    dTemp0+= mL[k,j]*vr[k]
                    # Initialize temp. container
                    dTemp2= vZ[k]*dTemp1
                    for l in 1:iP
                        # Temp. container
                        dL= mL[l,j]
                        @simd for m in 1:iP
                            dTemp2+= mL[m,k]*mN[m,l]*dL
                        end
                    end
                    mTemp[k,j]= dTemp2
                end
                vTemp[j]= dTemp0 + dTemp1*dv
            end
            # Update
            for j in 1:iP
                vr[j]= vTemp[j]
                @simd for k in 1:iP
                    mN[k,j]= mTemp[k,j]
                end
            end
        end
        
        # Smoothed state and smoothed state variance
        mPs= view(mP,:,iSt+1:iSt+iP)
        for j in 1:iP
            # Initialize temp. container
            dah= ma[j,(t-1)*iN+1]
            for k in 1:iP
                # Initialize container
                dV= mPs[k,j]
                # Smoothed state
                dah+= dV*vr[k]
                for l in 1:iP
                    # Temp. container
                    dP= mPs[l,j]
                    @simd for m in 1:iP
                        # Smoothed state variance
                        dV-= mPs[m,k]*mN[m,l]*dP
                    end
                end
                # Store result
                mV[k,(t-1)*iP+j]= dV
            end
            # Store result
            mah[j,t]= dah
        end

        # Smoothed state covariance
        if (t < iT)
            mPs1= view(mP,:,iSt1+1:iSt1+iP)
            for j in 1:iP
                for k in 1:iP
                    # Initialize container
                    dTemp= .0
                    for l in 1:iP
                        # Temp. container
                        dP0= mPs[l,j]
                        @simd for m in 1:iP
                            dTemp+= mTt[m,k]*mLp[m,l]*dP0
                        end
                    end
                    mTemp[k,j]= dTemp
                end
                for k in 1:iP
                    # Initialize temp. container
                    dVcov= mTemp[k,j]
                    for p in 1:iP
                        # Temp. container
                        dP1= mPs1[p,k]
                        @simd for q in 1:iP
                            # Smoothed state covariance
                            dVcov-= dP1*mNf[p,q]*mTemp[q,j]
                        end
                    end
                    # Store result
                    mVcov[k,(t-1)*iP+j]= dVcov
                end
            end
        end

        # Store N and update backward smoothing recursion
        if (t > 1)
            for j in 1:iP
                dTemp0= .0
                for k in 1:iP
                    dTemp1= .0
                    dTemp0+= mT[k,j]*vr[k]
                    for l in 1:iP
                        # Temp. container
                        dT= mT[l,j]
                        @simd for m in 1:iP
                            dTemp1+= mT[m,k]*mN[m,l]*dT
                        end
                    end
                    mTemp[k,j]= dTemp1
                end
                vTemp[j]= dTemp0
            end
            # Update 
            # Update
            for j in 1:iP
                vr[j]= vTemp[j]
                @simd for k in 1:iP
                    mNf[k,j]= mN[k,j]
                    mN[k,j]= mTemp[k,j]
                end
            end
        end
    end

    nothing
end

function KalmanSmootherCovEq_for!(mah::Array{Float64,2}, mV::Array{Float64,2}, mVcov::Array{Float64,2}, mZ::Array{Float64,2}, mT::Diagonal{Float64,Vector{Float64}}, ma::Array{Float64,2}, mP::Array{Float64,2}, mv::Array{Float64,2}, mF::Array{Float64,2})
    """
    Purpose:
        Implement the Kalman state smoother algorithm equation-by-equation to
        obtain smoothed states, smoothed states variance, and smoothed states
        autocovariance by means of for-loop "C++/Fortran style" programming

    Inputs:
        mZ          iN x iP matrix, system matrix Z
        mT          iP x iP matrix, system matrix T
        ma          iP x iT*iN+1 vector, predicted state vector
        mP          iP x iP*(iT*iN+1) matrix, predicted state variance
        mv          iN x iT matrix, forecast error
        mF          iN x iT matrix, forecast error variance

    Return value:
        mah         iP x iT matrix, smoothed state vectors mean
        mV          iP x iT*iP matrix, smoothed state vectors variance
        mVcov       iP x (iT-1)*iP matrix, smoothed state vectors autocovariance
    """
    (iN,iT)= size(mv)
    iP= size(ma,1)

    # Initialize temp. containers
    mLp= zeros(Float64,(iP,iP))
    mTemp= zeros(Float64,(iP,iP))
    mL= zeros(Float64,(iP,iP))
    vTemp= zeros(Float64,iP)

    # Initialize state smoothing recursions
    vr= zeros(Float64,iP)
    mN= zeros(Float64,(iP,iP))
    mNf= zeros(Float64,(iP,iP))

    # Identity matrix
    mI= Array{Float64}(I,(iP,iP))

    # Transpose
    mZt= transpose(mZ)

    # Kalman state smoother
    @inbounds @fastmath for t in iT:-1:1
        # Stepsize
        iSt= (t-1)*iN*iP
        iSt1= t*iN*iP

        # Initialize product of L matrices
        copyto!(mLp, mI)

        # Backward smoothing recursions
        for i in iN:-1:1
            # Temp. containers
            dv= mv[i,t]
            dFi= inv(mF[i,t])

            # Kalman gain
            vZ= view(mZt,:,i)
            mPs= view(mP,:,(i-1)*iP+iSt+1:i*iP+iSt)
            for j in 1:iP
                # Initialize temp. container
                dK= .0
                @simd for k in 1:iP
                    dK+= mPs[k,j]*vZ[k]
                end
                # Store result
                vTemp[j]= dK
            end

            # L matrix
            for j in 1:iP
                dZF= vZ[j]*dFi
                @simd for k in 1:iP
                    mL[k,j]= mI[k,j] - vTemp[k]*dZF
                end
            end

            # Update product of L matrices
            for j in 1:iP
                for k in 1:iP
                    dTemp= .0
                    @simd for l in 1:iP
                        dTemp+= mLp[k,l]*mL[l,j]
                    end
                    mTemp[k,j]= dTemp
                end
            end
            copyto!(mLp, mTemp)

            # Backward smoothing recursion
            for j in 1:iP
                # Initialize temp. containers
                dTemp0= .0
                dTemp1= vZ[j]*dFi
                for k in 1:iP
                    dTemp0+= mL[k,j]*vr[k]
                    # Initialize temp. container
                    dTemp2= vZ[k]*dTemp1
                    for l in 1:iP
                        # Temp. container
                        dL= mL[l,j]
                        @simd for m in 1:iP
                            dTemp2+= mL[m,k]*mN[m,l]*dL
                        end
                    end
                    mTemp[k,j]= dTemp2
                end
                vTemp[j]= dTemp0 + dTemp1*dv
            end
            # Update
            for j in 1:iP
                vr[j]= vTemp[j]
                @simd for k in 1:iP
                    mN[k,j]= mTemp[k,j]
                end
            end
        end
        
        # Smoothed state and smoothed state variance
        mPs= view(mP,:,iSt+1:iSt+iP)
        for j in 1:iP
            # Initialize temp. container
            dah= ma[j,(t-1)*iN+1]
            for k in 1:iP
                # Initialize container
                dV= mPs[k,j]
                # Smoothed state
                dah+= dV*vr[k]
                for l in 1:iP
                    # Temp. container
                    dP= mPs[l,j]
                    @simd for m in 1:iP
                        # Smoothed state variance
                        dV-= mPs[m,k]*mN[m,l]*dP
                    end
                end
                # Store result
                mV[k,(t-1)*iP+j]= dV
            end
            # Store result
            mah[j,t]= dah
        end

        # Smoothed state covariance
        if (t < iT)
            mPs1= view(mP,:,iSt1+1:iSt1+iP)
            for j in 1:iP
                for k in 1:iP
                    # Initialize container
                    dTemp= .0
                    dT= mT[k,k]
                    @simd for l in 1:iP
                        # Temp. container
                        dTemp+= dT*mLp[k,l]*mPs[l,j]
                    end
                    mTemp[k,j]= dTemp
                end
                for k in 1:iP
                    # Initialize temp. container
                    dVcov= mTemp[k,j]
                    for p in 1:iP
                        # Temp. container
                        dP1= mPs1[p,k]
                        @simd for q in 1:iP
                            # Smoothed state covariance
                            dVcov-= dP1*mNf[p,q]*mTemp[q,j]
                        end
                    end
                    # Store result
                    mVcov[k,(t-1)*iP+j]= dVcov
                end
            end
        end

        # Store N and update backward smoothing recursion
        if (t > 1)
            for j in 1:iP
                dT= mT[j,j]
                vr[j]= dT*vr[j]
                @simd for k in 1:iP
                    dN= mN[k,j]
                    mNf[k,j]= dN
                    mN[k,j]= mT[k,k]*dN*dT
                end
            end
        end
    end

    nothing
end