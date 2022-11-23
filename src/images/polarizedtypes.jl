export StokesVector, CoherencyMatrix, evpa, m̆, SingleStokes, CircBasis, LinBasis

"""
    $(TYPEDEF)
Static vector that holds the stokes parameters of a polarized
complex visibility

To convert between a `StokesVector` and `CoherencyMatrix` use the `convert`
function

```julia
convert(::CoherencyMatrix, StokesVector(1.0, 0.1, 0.1, 0.4))
```
"""
struct StokesVector{T} <: FieldVector{4,T}
    I::T
    Q::T
    U::T
    V::T
end

StaticArrays.similar_type(::Type{StokesVector}, ::Type{T}, s::Size{(4,)}) where {T} = StokesVector{T}



abstract type PolBasis end

"""
    CircBasis <: PolBasis

Measurement uses the circular polarization basis, which is typically used for circular
feed interferometers.
"""
struct CircBasis <: PolBasis end

"""
    LinBasis <: PolBasis

Measurement uses the linear polarization basis, which is typically used for linear
feed interferometers.
"""
struct LinBasis <: PolBasis end

"""
    CoherencyMatrix{T,B1,B2}

Coherency matrix for a single baseline with bases `B1` and `B2`. The two bases correspond
to the type of feeds used for each telescope and should be subtypes of `PolBasis`. To see which
bases are implemented type `subtypes(Rimes.PolBasis)` in the REPL.

For a circular basis the layout of the coherency matrix is
```
RR RL
LR RR
```
which can be constructed using
```julia-repl
c = CoherencyMatrix(RR, LR, RL, LL, CircBasis())
```

For a linear basis the layout of the coherency matrix is
```
XX XY
YX YY
```
which can be constructed using
```julia-repl
c = CoherencyMatrix(XX, YX, XY, YY, CircBasis())
```

For a mixed (e.g., circular and linear basis) the layout of the coherency matrix is
```
RX RY
LX LY
```

or e.g., linear and circular the layout of the coherency matrix is
```
XR XL
YR YL
```

which can be construct using
```julia-repl
c = CoherencyMatrix(XX, YX, XY, YY, CircBasis(), LinBasis())
# or
c = CoherencyMatrix(XX, YX, XY, YY, (CircBasis(), LinBasis()))
```


# Examples

There are a number of different ways to construct a coherency matrix:
```julia-repl
m1 = CoherencyMatrix(1.0, 0.0, 0.0, 1.0, CircBasis())
m2 = CoherencyMatrix(1.0, 0.0, 0.1, 1.0, LinBasis())
m3 = CoherencyMatrix(1.0, 0.0, 0.1, 1.0, CircBasis(), CircBasis())
m4 = CoherencyMatrix(1.0, 0.0, 0.1, 1.0, CircBasis(), LinBasis())
m4 = CoherencyMatrix(1.0, 0.0, 0.1, 1.0, (CircBasis(), LinBasis()))
```

# Warning
Note that because the CoherencyMatrix is a `FieldMatrix` the elements are specified
in a column-major order when constructing.
"""
struct CoherencyMatrix{B1,B2,T} <: StaticArrays.FieldMatrix{2,2,T}
    e11::T
    e21::T
    e12::T
    e22::T
end

function CoherencyMatrix(e11, e21, e12, e22, basis::NTuple{2,PolBasis})
    T = promote_type(e11, e12, e21, e22)
    return CoherencyMatrix{T, typeof(basis[1]), typeof(basis[2])}(T(e11), T(e21), T(e12), T(e22))
end

function CoherencyMatrix(e11, e21, e12, e22, basis::PolBasis)
    return CoherencyMatrix(e11, e21, e12, e22, (basis, basis))
end

function CoherencyMatrix(e11, e21, e12, e22, basis1::PolBasis, basis2::PolBasis)
    return CoherencyMatrix(e11, e21, e12, e22, (basis1, basis2))
end

function CoherencyMatrix(mat::AbstractMatrix, basis::Vararg{Any, N}) where {N}
    return CoherencyMatrix(mat[1], mat[2], mat[3], mat[4], basis...)
end

# Needed to ensure everything is constructed nicely
StaticArrays.similar_type(::Type{CoherencyMatrix{B1,B2}}, ::Type{T}, s::Size{(2,2)}) where {B1,B2,T} = CoherencyMatrix{B1,B2,T}

@inline function coherencymatrix(s::StokesVector, basis1::PolBasis, basis2::PolBasis)
    return convert(CoherencyMatrix{typeof(basis1), typeof(basis2)}, s)
end

@inline function Base.convert(::Type{<:CoherencyMatrix{CircBasis,CircBasis}}, s::StokesVector)
    (;I,Q,U,V) = s
    RR = (I + V)/2
    LR = (Q - 1im*U)/2
    RL = (Q + 1im*U)/2
    LL = (I - V)/2
    return CoherencyMatrix(RR, LR, RL, LL, CircBasis(), CircBasis())
end

@inline function Base.convert(::Type{<:CoherencyMatrix{LinBasis,LinBasis}}, s::StokesVector)
    (;I,Q,U,V) = s
    XX = (I + Q)/2
    YX = (U - 1im*V)/2
    XY = (U + 1im*V)/2
    YY = (I - Q)/2
    return CoherencyMatrix(XX, YX, XY, YY, LinBasis(), LinBasis())
end

@inline function Base.convert(::Type{<:CoherencyMatrix{CircBasis,LinBasis}}, s::StokesVector)
    (;I,Q,U,V) = s
    prefac = inv(2*sqrt(oftype(2, eltype(s.I))))
    RX = prefac*(I + Q - 1im*U - V)
    LX = prefac*(I + Q + 1im*U + V)
    RY = prefac*(-1im*I - 1im*Q + U + 1im*V)
    LY = prefac*(1im*I -1im*Q + U + 1im*V)
    return CoherencyMatrix(RX, LX, RY, LY, CircBasis(), LinBasis())
end

@inline function Base.convert(::Type{<:CoherencyMatrix{LinBasis,CircBasis}}, s::StokesVector)
    (;I,Q,U,V) = s
    prefac = inv(2*sqrt(oftype(2, eltype(s.I))))
    XR = prefac*(I + Q + 1im*U - V)
    YR = prefac*(I + Q -1im*U + V)
    XL = prefac*(1im*I - 1im*Q + U - 1im*V)
    YL = prefac*(-1im*I + 1im*Q + U - 1im*V)
    return CoherencyMatrix(XR, YR, XL, YL, LinBasis(), CircBasis())
end


@inline function Base.convert(::Type{StokesVector}, c::CoherencyMatrix{CircBasis, CircBasis})
    I = c.e11 + c.e22
    Q = c.e21 + c.e12
    U = 1im*(c.e21 - c.e12)
    V = c.e11 - c.e22
    return StokesVector(I, Q, U, V)
end

@inline function Base.convert(::Type{StokesVector}, c::CoherencyMatrix{LinBasis, LinBasis})
    I = c.e11 + c.e22
    Q = c.e11 - c.e22
    U = c.e21 + c.e12
    V = 1im*(c.e21 - c.e12)
    return StokesVector(I, Q, U, V)
end



"""
    $(SIGNATURES)

Computes `linearpol` from a set of stokes parameters `s`.
"""
function linearpol(s::StokesVector)
    return s.Q + 1im*s.U
end

"""
    $(SIGNATURES)
Compute the fractional linear polarization of a stokes vector
or coherency matrix
"""
m̆(m::StokesVector{T}) where {T} = (m.Q + 1im*m.U)/(m.I + eps(T))
m̆(m::CoherencyMatrix{CircBasis,CircBasis}) = 2*m.e12/(m.e11+m.e22)

"""
    $(SIGNATURES)
Compute the evpa of a stokes vector or cohereny matrix.
"""
evpa(m::StokesVector) = 1/2*atan(m.U,m.Q)
evpa(m::StokesVector{<:Complex}) = 1/2*angle(m.U/m.Q)
evpa(m::CoherencyMatrix) = evpa(convert(StokesVector, m))


"""
    $(TYPEDEF)
Helper function that converts a model from something that compute polarized images
to just a single stokes parameter. This is useful if you just want to fit a single
stokes parameter.
"""
struct SingleStokes{M, S} <: ComradeBase.AbstractModel
    model::M
end
SingleStokes(m::M, param::Symbol) where {M} = SingleStokes{M, param}(m)

visanalytic(::Type{SingleStokes{M,S}}) where {M,S} = visanalytic(M)
imanalytic(::Type{SingleStokes{M,S}})  where {M,S} = imanalytic(M)
isprimitive(::Type{SingleStokes{M,S}}) where {M,S} = isprimitive(M)
@inline intensity_point(s::SingleStokes{M,S}, x,y) where {M,S} = getproperty(intensity_point(s.model, x,y), S)





# struct PolarizedMap{SI<:AbstractIntensityMap,
#                     SQ<:AbstractIntensityMap,
#                     SU<:AbstractIntensityMap,
#                     SV<:AbstractIntensityMap} <: AbstractPolarizedMap{SI,SQ,SU,SV}
#     I::SI
#     Q::SQ
#     U::SU
#     V::SV
#     function PolarizedMap(I::SI,Q::SQ,U::SU,V::SV) where {SI, SQ, SU, SV}
#         @assert size(I) == size(Q) == size(U) == size(V) "Image sizes must be equal in polarized map"
#         @assert fov(I) == fov(Q) == fov(U) == fov(V) "Image fov must be equal in polarized map"
#         new{SI,SQ,SU,SV}(I,Q,U,V)
#     end
# end


# Base.Base.@propagate_inbounds function Base.getindex(pimg::PolarizedMap, i...)
# return StokesVector(pimg.I[i...], pimg.Q[i...], pimg.U[i...], pimg.V[i...])
# end

# @inline stokes_parameter(pimg::PolarizedMap, p::Symbol) = getproperty(pimg, p)
