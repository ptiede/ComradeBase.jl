export StokesParams, CoherencyMatrix, evpa, m̆, SingleStokes, CirBasis, LinBasis, coherencymatrix, stokesparams

"""
    $(TYPEDEF)
Static vector that holds the stokes parameters of a polarized
complex visibility

To convert between a `StokesParams` and `CoherencyMatrix` use the `convert`
function

```julia
convert(::CoherencyMatrix, StokesVector(1.0, 0.1, 0.1, 0.4))
```
"""
struct StokesParams{T} <: FieldVector{4,T}
    I::T
    Q::T
    U::T
    V::T
end

#const NdPi{Na, T, N} = NamedDimsArray{Na, T, N, <:StructArray{T, N}}
#StokesIntensityMap{T,N,Na} = KeyedArray{StokesParams{T}, N, <:NdPi{Na, StokesParams{T}, N}}

baseimage(s::IntensityMap) = parent(parent(s))

"""
    stackstokes(I, Q, U, V)

Create an array of full stokes parameters. The image is stored as a `StructArray` of
@ref[StokesParams]. Each of the four
"""
function StokesIntensityMap(I::IntensityMap{T}, Q::IntensityMap{T}, U::IntensityMap{T}, V::IntensityMap{T}) where {T}
    @assert check_grid(I,Q,U,V) "Intensity grids are not the same across the 4 stokes parameters"
    polimg = StructArray{StokesParams{T}}(I=baseimage(I), Q=baseimage(Q), U=baseimage(U), V=baseimage(V))
    return IntensityMap(polimg, named_axiskeys(I))
end

# simple check to ensure that the four grids are equal across stokes parameters
function check_grid(I,Q,U,V)
    named_axiskeys(I) == named_axiskeys(Q) == named_axiskeys(U) == named_axiskeys(V)
end

function AxisKeys.named_axiskeys(simg::StokesIntensityMap)
    return named_axiskeys(simg.I)
end

function Base.summary(io::IO, x::StokesIntensityMap)
    return _summary(io, x)
end

function _summary(io, x::StokesIntensityMap{T,N,Na}) where {T,N,Na}
    println(io, ndims(x.I), "-dimensional")
    println(io, "StokesIntensityMap{$T, $N, $Na}")
    for d in 1:ndims(x)
        print(io, d==1 ? "↓" : d==2 ? "→" : d==3 ? "◪" : "▨", "   ")
        c = AxisKeys.colour(x, d)
        AxisKeys.hasnames(x.I) && printstyled(io, AxisKeys.dimnames(x.I,d), " ∈ ", color=c)
        printstyled(io, length(axiskeys(x.I,d)), "-element ", AxisKeys.shorttype(axiskeys(x.I,d)), "\n", color=c)
    end
    println(io, "Polarizations ", propertynames(x))
end

Base.show(io::IO, img::StokesIntensityMap) = summary(io, img)





StaticArrays.similar_type(::Type{StokesParams}, ::Type{T}, s::Size{(4,)}) where {T} = StokesParams{T}

abstract type ElectricFieldBasis end

struct R <: ElectricFieldBasis end
struct L <: ElectricFieldBasis end
struct X <: ElectricFieldBasis end
struct Y <: ElectricFieldBasis end

struct PolBasis{B1, B2} end


const POLBASES = Union{
                    PolBasis{X,Y}, PolBasis{X, Missing}, PolBasis{Missing, Y},
                    PolBasis{R,L}, PolBasis{R, Missing}, PolBasis{Missing, L},
                    PolBasis{X,R}, PolBasis{Y,R}, PolBasis{X,L}, PolBasis{Y,L},
                    PolBasis{R,X}, PolBasis{R,Y}, PolBasis{L,X}, PolBasis{L,Y},
                    }


"""
    CirBasis <: PolBasis

Measurement uses the circular polarization basis, which is typically used for circular
feed interferometers.
"""
const CirBasis = PolBasis{R,L}

"""
    LinBasis <: PolBasis

Measurement uses the linear polarization basis, which is typically used for linear
feed interferometers.
"""
const LinBasis = PolBasis{X, Y}



"""
    CoherencyMatrix{B1,B2,T}

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
c = CoherencyMatrix(RR, LR, RL, LL, CirBasis())
```

For a linear basis the layout of the coherency matrix is
```
XX XY
YX YY
```
which can be constructed using
```julia-repl
c = CoherencyMatrix(XX, YX, XY, YY, CirBasis())
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
c = CoherencyMatrix(XX, YX, XY, YY, CirBasis(), LinBasis())
# or
c = Coherency in a basis given by `B`. There are a two main bases we
use `:RL` for a circular basis, and `:XY` for linear basis.Matrix(XX, YX, XY, YY, (CirBasis(), LinBasis()))
```


# Examples

There are a number of different ways to construct a coherency matrix:
```julia-repl
m1 = CoherencyMatrix(1.0, 0.0, 0.0, 1.0, CirBasis())
m2 = CoherencyMatrix(1.0, 0.0, 0.1, 1.0, LinBasis())
m3 = CoherencyMatrix(1.0, 0.0, 0.1, 1.0, CirBasis(), CirBasis())
m4 = CoherencyMatrix(1.0, 0.0, 0.1, 1.0, CirBasis(), LinBasis())
m4 = CoherencyMatrix(1.0, 0.0, 0.1, 1.0, (CirBasis(), LinBasis()))
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
    T = promote_type(typeof(e11), typeof(e12), typeof(e21), typeof(e22))
    return CoherencyMatrix{typeof(basis[1]), typeof(basis[2]),T}(T(e11), T(e21), T(e12), T(e22))
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

function CoherencyMatrix(s::StokesParams, b1::PolBasis, b2::PolBasis)
    return convert(CoherencyMatrix{typeof(b1), typeof(b2)}, s)
end

function CoherencyMatrix{B1,B2}(s::StokesParams) where {B1, B2}
    return convert(CoherencyMatrix{B1, B2}, s)
end

@inline function Base.convert(::Type{<:CoherencyMatrix{CirBasis, CirBasis}}, s::StokesParams)
    (;I,Q,U,V) = s
    RR = complex((I + V)/2)
    LR = (Q - 1im*U)/2
    RL = (Q + 1im*U)/2
    LL = complex((I - V)/2)
    return CoherencyMatrix(RR, LR, RL, LL, CirBasis(), CirBasis())
end

@inline function Base.convert(::Type{<:CoherencyMatrix{LinBasis, LinBasis}}, s::StokesParams)
    (;I,Q,U,V) = s
    XX = (I + Q)/2
    YX = (U - 1im*V)/2
    XY = (U + 1im*V)/2
    YY = (I - Q)/2
    return CoherencyMatrix(XX, YX, XY, YY, LinBasis(), LinBasis())
end

@inline function Base.convert(::Type{<:CoherencyMatrix{CirBasis, LinBasis}}, s::StokesParams)
    (;I,Q,U,V) = s
    prefac = oftype(s.I, inv(2*sqrt(2)))
    RX = prefac*(I + Q - 1im*U - V)
    LX = prefac*(I + Q + 1im*U + V)
    RY = prefac*(-1im*I - 1im*Q + U + 1im*V)
    LY = prefac*(1im*I -1im*Q + U + 1im*V)
    return CoherencyMatrix(RX, LX, RY, LY, CirBasis(), LinBasis())
end

@inline function Base.convert(::Type{<:CoherencyMatrix{LinBasis, CirBasis}}, s::StokesParams)
    (;I,Q,U,V) = s
    prefac = oftype(s.I, inv(2*sqrt(2)))
    XR = prefac*(I + Q + 1im*U - V)
    YR = prefac*(I + Q -1im*U + V)
    XL = prefac*(1im*I - 1im*Q + U - 1im*V)
    YL = prefac*(-1im*I + 1im*Q + U - 1im*V)
    return CoherencyMatrix(XR, YR, XL, YL, LinBasis(), CirBasis())
end


@inline function StokesParams(c::CoherencyMatrix{CirBasis, CirBasis})
    I = c.e11 + c.e22
    Q = c.e21 + c.e12
    U = 1im*(c.e21 - c.e12)
    V = c.e11 - c.e22
    return StokesParams(I, Q, U, V)
end

@inline function StokesParams(c::CoherencyMatrix{LinBasis, LinBasis})
    I = c.e11 + c.e22
    Q = c.e11 - c.e22
    U = c.e21 + c.e12
    V = 1im*(c.e21 - c.e12)
    return StokesParams(I, Q, U, V)
end



"""
    $(SIGNATURES)

Computes `linearpol` from a set of stokes parameters `s`.
"""
function linearpol(s::StokesParams)
    return s.Q + 1im*s.U
end

"""
    $(SIGNATURES)
Compute the fractional linear polarization of a stokes vector
or coherency matrix
"""
m̆(m::StokesParams{T}) where {T} = (m.Q + 1im*m.U)/(m.I + eps(T))
m̆(m::CoherencyMatrix{CirBasis,CirBasis}) = 2*m.e12/(m.e11+m.e22)

"""
    $(SIGNATURES)
Compute the evpa of a stokes vector or cohereny matrix.
"""
evpa(m::StokesParams) = 1/2*atan(m.U,m.Q)
evpa(m::StokesParams{<:Complex}) = 1/2*angle(m.U/m.Q)


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
# return StokesParams(pimg.I[i...], pimg.Q[i...], pimg.U[i...], pimg.V[i...])
# end

# @inline stokes_parameter(pimg::PolarizedMap, p::Symbol) = getproperty(pimg, p)
