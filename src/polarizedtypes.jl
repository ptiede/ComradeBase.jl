export StokesParams, CoherencyMatrix, evpa, m̆, SingleStokes, R, L, X, Y, PolBasis, CirBasis, LinBasis,
       basis_components, basis_transform, coherencymatrix, stokesparams


"""
    $(TYPEDEF)

An abstract type whose subtypes denote a specific electric field basis.
"""
abstract type ElectricFieldBasis end

"""
    $(TYPEDEF)

The right circular electric field basis, i.e. a right-handed circular feed.
"""
struct R <: ElectricFieldBasis end

"""
    $(TYPEDEF)

The left circular electric field basis, i.e. a left-handed circular feed.
"""
struct L <: ElectricFieldBasis end

"""
    $(TYPEDEF)

The horizontal or X electric feed basis, i.e. the horizontal linear feed.
"""
struct X <: ElectricFieldBasis end

"""
    $(TYPEDEF)

The vertical or Y electric feed basis, i.e. the vertical linear feed.
"""
struct Y <: ElectricFieldBasis end


"""
    $(TYPEDEF)

Denotes a general polarization basis, with basis vectors (B1,B2) which are typically
`<: Union{ElectricFieldBasis, Missing}`
"""
struct PolBasis{B1<:Union{ElectricFieldBasis, Missing}, B2<:Union{ElectricFieldBasis, Missing}} end


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
const LinBasis = PolBasis{X,Y}

"""
    basis_components([T=Float64,], e::ElectricFieldBasis, b::PolBasis)

Returns a static vector that contains the components of the electric field basis vector `e`
in terms of the polarization basis `b`. The first argument is optionally the eltype of the
static vector.

# Examples
```julia
julia> basis_components(Float64, R(), PolBasis{X,Y}())
2-element StaticArraysCore.SVector{2, ComplexF64} with indices SOneTo(2):
 0.7071067811865475 + 0.0im
                0.0 - 0.7071067811865475im

julia> basis_components(R(), PolBasis{X,Y}())
2-element StaticArraysCore.SVector{2, ComplexF64} with indices SOneTo(2):
 0.7071067811865475 + 0.0im
                0.0 - 0.7071067811865475im


julia> basis_components(Float64, X(), PolBasis{X,Y}())
2-element StaticArraysCore.SVector{2, ComplexF64} with indices SOneTo(2):
 1.0 + 0.0im
 0.0 + 0.0im
```
"""
function basis_components end

@inline innerprod(::Type{T}, ::B, ::B) where {T, B<:ElectricFieldBasis} = complex(one(T), zero(T))
@inline innerprod(::Type{T}, ::R, ::L) where {T} = complex(zero(T))
@inline innerprod(::Type{T}, ::L, ::R) where {T} = complex(zero(T))

@inline innerprod(::Type{T}, ::X, ::Y) where {T} = complex(zero(T))
@inline innerprod(::Type{T}, ::Y, ::X) where {T} = complex(zero(T))


@inline innerprod(::Type{T}, ::X, ::R) where {T} = complex(inv(sqrt(T(2))), zero(T))
@inline innerprod(::Type{T}, ::X, ::L) where {T} = complex(inv(sqrt(T(2))), zero(T))

@inline innerprod(::Type{T}, ::Y, ::R) where {T} = complex(zero(T), inv(sqrt(T(2))))
@inline innerprod(::Type{T}, ::Y, ::L) where {T} = complex(zero(T), -inv(sqrt(T(2))))

# use the conjugate symmetry of the inner product
@inline innerprod(::Type{T}, c::Union{R,L}, l::Union{X,Y}) where {T} = conj(innerprod(T, l, c))

# Now handle missing
@inline innerprod(::Type{T}, c::Missing, l::ElectricFieldBasis) where {T} = missing
@inline innerprod(::Type{T}, l::ElectricFieldBasis, c::Missing) where {T} = missing


@inline basis_components(::Type{T}, b1::Union{ElectricFieldBasis,Missing}, ::PolBasis{B1,B2}) where {T, B1,B2} = SVector{2}(innerprod(T, b1, B1()), innerprod(T, b1, B2()))
@inline basis_components(v::Union{ElectricFieldBasis, Missing}, b::PolBasis) = basis_components(Float64, v, b)

"""
    basis_transform([T=Float64,], b1::PolBasis, b2::PolBasis)
    basis_transform([T=Float64,], b1::PolBasis=>b2::PolBasis)

Produces the transformation matrix that transforms the vector components from basis `b1` to basis `b2`.
This means that if for example `E` is the circular basis then `basis_transform(CirBasis=>LinBasis)E` is in the
linear basis. In other words the **columns** of the transformation matrix are the coordinate vectors
of the new basis vectors in the old basis.

# Example
```julia-repl
julia> basis_transform(CirBasis()=>LinBasis())
2×2 StaticArraysCore.SMatrix{2, 2, ComplexF64, 4} with indices SOneTo(2)×SOneTo(2):
 0.707107-0.0im       0.707107-0.0im
      0.0-0.707107im       0.0+0.707107im
```
"""
function basis_transform end

@inline basis_transform(::Type{T}, b1::PolBasis{E1,E2}, b2::PolBasis) where {E1,E2,T} = hcat(basis_components(T, E1(), b2), basis_components(T, E2(), b2))
@inline basis_transform(b1::PolBasis, b2::PolBasis) = basis_transform(Float64, b1, b2)

@inline basis_transform(::Type{T}, p::Pair{B1,B2}) where {T, B1<:PolBasis, B2<:PolBasis} = basis_transform(T, B1(), B2())
@inline basis_transform(::Pair{B1,B2}) where {B1<:PolBasis, B2<:PolBasis} = basis_transform(Float64, B1(), B2())



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

const NdPi{Na, T, N} = NamedDimsArray{Na, T, N, <:StructArray{T, N}}
StokesIntensityMap{T,N,Na} = KeyedArray{StokesParams{T}, N, <:NdPi{Na, StokesParams{T}, N}}

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
    println(io, ndims(x), "-dimensional")
    println(io, "StokesIntensityMap{$T, $N, $Na}")
    for d in 1:ndims(x)
        print(io, d==1 ? "↓" : d==2 ? "→" : d==3 ? "◪" : "▨", "   ")
        c = AxisKeys.colour(x, d)
        AxisKeys.hasnames(x) && printstyled(io, AxisKeys.dimnames(x,d), " ∈ ", color=c)
        printstyled(io, length(axiskeys(x,d)), "-element ", AxisKeys.shorttype(axiskeys(x,d)), "\n", color=c)
    end
    println(io, "Polarizations ", propertynames(parent(x).data))
end

Base.show(io::IO, img::StokesIntensityMap) = summary(io, img)


StaticArrays.similar_type(::Type{StokesParams}, ::Type{T}, s::Size{(4,)}) where {T} = StokesParams{T}


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
# Circular and linear feeds i.e., R⊗X
c = CoherencyMatrix(RX, LX, RY, LY, LinBasis(), CirBasis())
# Linear and circular feeds i.e., X⊗R
c = CoherencyMatrix(XR, YR, XL, YL, LinBasis(), CirBasis())
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

# Needed to ensure everything is constructed nicely
StaticArrays.similar_type(::Type{CoherencyMatrix{B1,B2}}, ::Type{T}, s::Size{(2,2)}) where {B1,B2,T} = CoherencyMatrix{B1,B2,T}



@inline function CoherencyMatrix(e11, e21, e12, e22, basis::NTuple{2,PolBasis})
    T = promote_type(typeof(e11), typeof(e12), typeof(e21), typeof(e22))
    return CoherencyMatrix{typeof(basis[1]), typeof(basis[2]),T}(T(e11), T(e21), T(e12), T(e22))
end

@inline function CoherencyMatrix(e11, e21, e12, e22, basis::PolBasis)
    return CoherencyMatrix(e11, e21, e12, e22, (basis, basis))
end

@inline function CoherencyMatrix(e11, e21, e12, e22, basis1::PolBasis, basis2::PolBasis)
    return CoherencyMatrix(e11, e21, e12, e22, (basis1, basis2))
end

@inline function CoherencyMatrix(mat::AbstractMatrix, basis::Vararg{Any, N}) where {N}
    return CoherencyMatrix(mat[1], mat[2], mat[3], mat[4], basis...)
end


@inline function CoherencyMatrix(s::StokesParams, b1::PolBasis, b2::PolBasis)
    return convert(CoherencyMatrix{typeof(b1), typeof(b2)}, s)
end

@inline function CoherencyMatrix(s::StokesParams, b::PolBasis)
    return convert(CoherencyMatrix{typeof(b), typeof(b)}, s)
end

@inline function CoherencyMatrix{B1,B2}(s::StokesParams) where {B1, B2}
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



@inline function StokesParams(c::CoherencyMatrix{CirBasis, CirBasis})
    I = c.e11 + c.e22
    Q = c.e21 + c.e12
    U = 1im*(c.e21 - c.e12)
    V = c.e11 - c.e22
    return StokesParams(I, Q, U, V)
end

# @inline function StokesParams(c::CoherencyMatrix{LinBasis, LinBasis})
#     I = c.e11 + c.e22
#     Q = c.e11 - c.e22
#     U = c.e21 + c.e12
#     V = 1im*(c.e21 - c.e12)
#     return StokesParams(I, Q, U, V)
# end

@inline function StokesParams(c::CoherencyMatrix{B1, B2}) where {B1, B2}
    t1 = basis_transform(B1()=>CirBasis())
    # Flip because these are the dual elements
    t2 = basis_transform(CirBasis()=>B2())
    c_cir = CoherencyMatrix(t1*c*t2, CirBasis())
    return StokesParams(c_cir)
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
# m̆(m::CoherencyMatrix{CirBasis,CirBasis}) = 2*m.e12/(m.e11+m.e22)

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
