#=
system.jl

    State space system abstract type and constructors

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2022/02/17
=#
"""
    StateSpaceSystem

Abstract type for state space systems.
"""
abstract type StateSpaceSystem end

"""
    LinearTimeInvariant <: StateSpaceSystem

Constructor for a linear time invariant instance of the state space system type
with system matrices ``Z``, ``T``, ``d``, ``c``, ``H``, and ``Q`` and
initialization of the Kalman filter ``a₁`` and ``P₁``.
"""
struct LinearTimeInvariant{Ty, TZ, TT, Td, Tc, TH, TQ, Ta, TP} <: StateSpaceSystem
	y::Ty	# data
	Z::TZ	# system matrix Z
	T::TT	# system matrix T
	d::Td	# mean adjustment observation
	c::Tc	# mean adjustment state
	H::TH	# system matrix H
	Q::TQ	# system matrix Q
	a1::Ta	# initial state
	P1::TP	# initial state variance
end
# Constructor
function LinearTimeInvariant{Ty, TZ, TT, Td, Tc, TH, TQ, Ta, TP}(n::Integer, p::Integer, T_len::Integer) where {Ty, TZ, TT, Td, Tc, TH, TQ, Ta, TP}
	# data
	y= Ty(undef, n, T_len)

    # Initialize system components
    Z= TZ(undef, n, p)
    T= TT <: Diagonal ? TT(undef, p) : TT(undef, p, p)
	d= Td(undef, n)
	c= Tc(undef, p)
    H= TH <: Diagonal ? TH(undef, n) : Symmetric(Matrix{eltype(TH)}(undef, n, n))
    Q= TQ <: Diagonal ? TQ(undef, p) : Symmetric(Matrix{eltype(TQ)}(undef, p, p))

	# Initial conditions
	a1= similar(Ta, p)
	P1= TP <: Diagonal ? TP(undef, p) : Symmetric(Matrix{eltype(TP)}(undef, p, p))

    return LinearTimeInvariant(y, Z, T, d, c, H, Q, a1, P1)
end

"""
    LinearTimeVariant <: StateSpaceSystem

Constructor for a linear time variant instance of the state space system type
with system matrices ``Zₜ``, ``Tₜ``, ``dₜ``, ``cₜ``, ``Hₜ``, and ``Qₜ`` and
initialization of the Kalman filter ``a₁`` and ``P₁``.
"""
struct LinearTimeVariant{Ty, TZ, TT, Td, Tc, TH, TQ, Ta, TP} <: StateSpaceSystem
	y::Ty	# data
	Z::TZ	# system matrices Zₜ
	T::TT	# system matrices Tₜ
	d::Td	# mean adjustments observation
	c::Tc	# mean adjustments state
	H::TH	# system matrices Hₜ
	Q::TQ	# system matrices Qₜ
	a1::Ta	# initial state
	P1::TP	# initial state variance
end
# Constructor
function LinearTimeVariant{Ty, TZ, TT, Td, Tc, TH, TQ, Ta, TP}(n::Integer, p::Integer, T_len::Integer) where {Ty, TZ, TT, Td, Tc, TH, TQ, Ta, TP}
	# data
	y= Ty(undef, n, T_len)

    # Initialize system components
    Z= [TZ(undef, n, p) for _ in 1:T_len]
    T= TT <: Diagonal ? [TT(undef, p) for _ in 1:T_len] : 
                        [TT(undef, p, p) for _ in 1:T_len]
	d= Td(undef, n, T_len)
	c= Tc(undef, p, T_len)
    H= TH <: Diagonal ? [TH(undef, n) for _ in 1:T_len] : 
                        [Symmetric(Matrix{eltype(TH)}(undef, n, n)) for _ in 1:T_len]
    Q= TQ <: Diagonal ? [TQ(undef, p) for _ in 1:T_len] : 
                        [Symmetric(Matrix{eltype(TQ)}(undef, p, p)) for _ in 1:T_len]

	# Initial conditions
	a1= similar(Ta, p)
	P1= TP <: Diagonal ? TP(undef, p) : Symmetric(Matrix{eltype(TP)}(undef, p, p))

    return LinearTimeVariant(y, Z, T, d, c, H, Q, a1, P1)
end