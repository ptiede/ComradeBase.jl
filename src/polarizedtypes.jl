export StokesParams, CoherencyMatrix, evpa, m̆, SingleStokes, RPol, LPol, XPol, YPol, PolBasis, CirBasis, LinBasis,
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
struct RPol <: ElectricFieldBasis end

"""
    $(TYPEDEF)

The left circular electric field basis, i.e. a left-handed circular feed.
"""
struct LPol <: ElectricFieldBasis end

"""
    $(TYPEDEF)

The horizontal or X electric feed basis, i.e. the horizontal linear feed.
"""
struct XPol <: ElectricFieldBasis end

"""
    $(TYPEDEF)

The vertical or Y electric feed basis, i.e. the vertical linear feed.
"""
struct YPol <: ElectricFieldBasis end


abstract type StokesBasis end
struct IPol end
struct QPol end
struct UPol end
struct VPol end


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
const CirBasis = PolBasis{RPol,LPol}

"""
    LinBasis <: PolBasis

Measurement uses the linear polarization basis, which is typically used for linear
feed interferometers.
"""
const LinBasis = PolBasis{XPol,YPol}

"""
    basis_components([T=Float64,], e::ElectricFieldBasis, b::PolBasis)

Returns a static vector that contains the components of the electric field basis vector `e`
in terms of the polarization basis `b`. The first argument is optionally the eltype of the
static vector.

# Examples
```julia
julia> basis_components(Float64, R(), PolBasis{XPol,Y}())
2-element StaticArraysCore.SVector{2, ComplexF64} with indices SOneTo(2):
 0.7071067811865475 + 0.0im
                0.0 - 0.7071067811865475im

julia> basis_components(R(), PolBasis{XPol,Y}())
2-element StaticArraysCore.SVector{2, ComplexF64} with indices SOneTo(2):
 0.7071067811865475 + 0.0im
                0.0 - 0.7071067811865475im


julia> basis_components(Float64, X(), PolBasis{XPol,Y}())
2-element StaticArraysCore.SVector{2, ComplexF64} with indices SOneTo(2):
 1.0 + 0.0im
 0.0 + 0.0im
```
"""
function basis_components end


"""
    innerprod(::Type{T}, XPol(), YPol())

Computes the complex inner product of two elements of a complex Hilbert space `X` and `Y`
where base element of the output is T.
"""
function innerprod end

# Define that XPol,YPol and RPol,LPol are orthonormal bases
@inline innerprod(::Type{T}, ::B, ::B) where {T, B<:ElectricFieldBasis} = one(T)
@inline innerprod(::Type{T}, ::RPol, ::LPol) where {T} = complex(zero(T))
@inline innerprod(::Type{T}, ::LPol, ::RPol) where {T} = complex(zero(T))
@inline innerprod(::Type{T}, ::XPol, ::YPol) where {T} = complex(zero(T))
@inline innerprod(::Type{T}, ::YPol, ::XPol) where {T} = complex(zero(T))

# Now define the projections of linear onto circular
@inline innerprod(::Type{T}, ::XPol, ::RPol) where {T} = complex(inv(sqrt(T(2))), zero(T))
@inline innerprod(::Type{T}, ::XPol, ::LPol) where {T} = complex(inv(sqrt(T(2))), zero(T))

@inline innerprod(::Type{T}, ::YPol, ::RPol) where {T} = complex(zero(T), -inv(sqrt(T(2))))
@inline innerprod(::Type{T}, ::YPol, ::LPol) where {T} = complex(zero(T), inv(sqrt(T(2))))

# use the conjugate symmetry of the inner product to define projections of circular onto linear.
@inline innerprod(::Type{T}, c::Union{RPol,LPol}, l::Union{XPol,YPol}) where {T} = conj(innerprod(T, l, c))

# Now handle missing basis vectors (important when you are missing a feed)
@inline innerprod(::Type{T}, c::Missing, l::ElectricFieldBasis) where {T} = missing
@inline innerprod(::Type{T}, l::ElectricFieldBasis, c::Missing) where {T} = missing

# Now give me the components of electic fields in both linear and circular bases.
@inline basis_components(::Type{T}, b1::Union{ElectricFieldBasis,Missing}, ::PolBasis{B1,B2}) where {T, B1,B2} = SVector{2}(innerprod(T, B1(), b1), innerprod(T, B2(), b1))
@inline basis_components(v::Union{ElectricFieldBasis, Missing}, b::PolBasis) = basis_components(Float64, v, b)

#This handles non-orthogonal bases
# @inline basis_components(::Type{T}, b1::B1, ::PolBasis{B1, B2}) where {T, B1<:ElectricFieldBasis, B2<:ElectricFieldBasis} = SVector{2}(complex(one(T)), complex(zero(T)))
# @inline basis_components(::Type{T}, b1::B2, ::PolBasis{B1, B2}) where {T, B1<:ElectricFieldBasis, B2<:ElectricFieldBasis} = SVector{2}(complex(zero(T)), complex(one(T)))
# @inline basis_components(::Type{T}, b1::B1, ::PolBasis{B1, B1}) where {T, B1<:ElectricFieldBasis} = SVector{2}(complex(one(T)), complex(one(T)))


for (E1, E2) in [(:XPol,:RPol), (:RPol, :XPol), (:RPol,:YPol), (:YPol,:RPol), (:XPol,:LPol), (:LPol,:XPol), (:YPol,:LPol), (:LPol,:YPol)]
    @eval begin
        PolBasis{$E1,$E2}() = throw(AssertionError("Non-orthogonal bases not implemented"))
    end
end

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
# @inline basis_transform(::Type{T}, b1::B, b2::B) where {T, B<:PolBasis} = SMatrix{2,2,Complex{T}}(1.0, 0.0, 0.0, 1.0)
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


StaticArrays.similar_type(::Type{StokesParams}, ::Type{T}, s::Size{(4,)}) where {T} = StokesParams{T}


"""
    $(TYPEDEF)

Coherency matrix for a single baseline with bases `B1` and `B2`. The two bases correspond
to the type of feeds used for each telescope and should be subtypes of `PolBasis`. To see which
bases are implemented type `subtypes(Rimes.PolBasis)` in the REPL.

For a circular basis the layout of the coherency matrix is
```
RR* RL*
LR* RR*
```
which can be constructed using
```julia-repl
c = CoherencyMatrix(RR, LR, RL, LL, CirBasis())
```

For a linear basis the layout of the coherency matrix is
```
XX* XY*
YX* YY*
```
which can be constructed using
```julia-repl
c = CoherencyMatrix(XX, YX, XY, YY, CirBasis())
```

For a mixed (e.g., circular and linear basis) the layout of the coherency matrix is
```
RX* RY*
LX* LY*
```

or e.g., linear and circular the layout of the coherency matrix is
```
XR* XL*
YR* YL*
```

These coherency matrices can be constructed using:
```julia-repl
# Circular and linear feeds i.e., |R><X|
c = CoherencyMatrix(RX, LX, RY, LY, LinBasis(), CirBasis())
# Linear and circular feeds i.e., |X><R|
c = CoherencyMatrix(XR, YR, XL, YL, LinBasis(), CirBasis())
```

"""
struct CoherencyMatrix{B1,B2,T} <: StaticArrays.FieldMatrix{2,2,T}
    e11::T
    e21::T
    e12::T
    e22::T
end

ChainRulesCore.ProjectTo(x::CoherencyMatrix{B1, B2, <: Number}) where {B1, B2} = ProjectTo{CoherencyMatrix}(; element = ProjectTo(eltype(x)), basis1=B1(), basis2=B2())
function (project::ProjectTo{CoherencyMatrix})(dx::AbstractMatrix)
    @assert size(dx) == (2,2) "Issue in Coherency pullback the matrix is not 2x2"
    return CoherencyMatrix(dx, project.basis1, project.basis2)
end

# function ChainRulesCore.rrule(::Type{<:CoherencyMatrix}, e11, e21, e12, e22, basis::NTuple{2, <:PolBasis})
#     c = CoherencyMatrix(e11, e21, e12, e22, basis)
#     pr = ProjectTo(e11)
#     function _CoherencyMatrix_sep_pullback(Δ)
#         return NoTangent(), pr(Δ[1,1]), pr(Δ[2,1]), pr(Δ[1,2]), pr(Δ[2,2]), NoTangent()
#     end
#     return c, _CoherencyMatrix_sep_pullback
# end

# Needed to ensure everything is constructed nicely
StaticArrays.similar_type(::Type{CoherencyMatrix{B1,B2}}, ::Type{T}, s::Size{(2,2)}) where {B1,B2,T} = CoherencyMatrix{B1,B2,T}


"""
    CoherencyMatrix(e11, e21, e12, e22, basis::NTuple{2, PolBasis})

Constructs the coherency matrix with components
   e11 e12
   e21 e22
relative to the tensor product basis, `|basis[1]><basis[2]|`. Note that basis[1] and basis[2]
could be different.

For instance
```julia
c = Coherency(1.0, 0.0, 0.0, 1.0, CirBasis(), LinBasis())
```
elements correspond to
    RX* RY*
    LX* LY*
"""
@inline function CoherencyMatrix(e11::Number, e21::Number, e12::Number, e22::Number, basis::NTuple{2,PolBasis})
    T = promote_type(typeof(e11), typeof(e12), typeof(e21), typeof(e22))
    return CoherencyMatrix{typeof(basis[1]), typeof(basis[2]),T}(T(e11), T(e21), T(e12), T(e22))
end

"""
    CoherencyMatrix(e11, e21, e12, e22, basis::PolBasis)

Constructs the coherency matrix with components
   e11 e12
   e21 e22
relative to the tensor product basis, `basis` given by `|basis><basis|`.

For instance
```julia
c = Coherency(1.0, 0.0, 0.0, 1.0, CirBasis())
```
elements correspond to
    RR* RL*
    LR* LL*

"""
@inline function CoherencyMatrix(e11::Number, e21::Number, e12::Number, e22::Number, basis::PolBasis)
    return CoherencyMatrix(e11, e21, e12, e22, (basis, basis))
end

"""
    CoherencyMatrix(e11, e21, e12, e22, basis1::PolBasis basis2::PolBasis)

Constructs the coherency matrix with components
   e11 e12
   e21 e22
relative to the tensor product basis, `basis` given by `|basis1><basis2|`.

For instance
```julia
c = Coherency(1.0, 0.0, 0.0, 1.0, CirBasis(), LinBasis())
```
elements correspond to
    RX* RY*
    LX* LY*

"""
@inline function CoherencyMatrix(e11::Number, e21::Number, e12::Number, e22::Number, basis1::PolBasis, basis2::PolBasis)
    return CoherencyMatrix(e11, e21, e12, e22, (basis1, basis2))
end

@inline function CoherencyMatrix(mat::AbstractMatrix, basis1::PolBasis, basis2::PolBasis)
    return CoherencyMatrix(mat[1], mat[2], mat[3], mat[4], basis1, basis2)
end

@inline function CoherencyMatrix(mat::AbstractMatrix, basis::PolBasis)
    return CoherencyMatrix(mat[1], mat[2], mat[3], mat[4], basis)
end



"""
    CoherencyMatrix(s::StokesParams, basis1::PolBasis)
    CoherencyMatrix(s::StokesParams, basis1::PolBasis, basis2::PolBasis)
    CoherencyMatrix(s::StokesParams, basis1::PolBasis, basis2::PolBasis, refbasis=CirBasis())

Constructs the coherency matrix from the set of stokes parameters `s`.
This is specialized on `basis1` and `basis2` which form the tensor product basis
`|basis1><basis2|`, or if a single basis is given then by `|basis><basis|`.

For example
```julia
CoherencyMatrix(s, CircBasis())
```
will give the coherency matrix

   |I+V   Q+iU|
   |Q-iU  I-V |

while
```julia
CoherencyMatrix(s, LinBasis())
```
will give
    |I+Q   U+iV|
    |U-iV  I-Q |

# Notes

Internally this function first converts to a reference basis and then the final basis.
You can select the reference basis used with the optional argument refbasis. By default
we use the circular basis as our reference. Note that this is only important for mixed bases,
e.g., if `basis1` and `basis2` are different. If `basis1==basis2` then the reference basis
is never used.
"""
@inline function CoherencyMatrix(s::StokesParams, b1::PolBasis, b2::PolBasis, refbasis::Union{LinBasis, CirBasis}=CirBasis())
    t1 = basis_transform(refbasis=>b1)
    # Flip because these are the dual elements
    t2 = basis_transform(b2=>refbasis)
    # Use circular basis as a reference
    c_cir = CoherencyMatrix(s, refbasis)
    return CoherencyMatrix(t1*c_cir*t2, b1, b2)
end

function CoherencyMatrix(s::StokesParams, b1::T, b2::T, refbasis=CirBasis()) where {T<:PolBasis}
    return CoherencyMatrix(s, b1)
end

@inline function CoherencyMatrix(s::StokesParams, b::PolBasis)
    return CoherencyMatrix{typeof(b), typeof(b)}(s)
end

@inline function CoherencyMatrix{CirBasis,CirBasis}(s::StokesParams)
    (;I,Q,U,V) = s
    RR = complex((I + V))
    LR = (Q - 1im*U)
    RL = (Q + 1im*U)
    LL = complex((I - V))
    return CoherencyMatrix(RR, LR, RL, LL, CirBasis(), CirBasis())
end


@inline function CoherencyMatrix{LinBasis, LinBasis}(s::StokesParams)
    (;I,Q,U,V) = s
    XX = (I + Q)
    YX = (U - 1im*V)
    XY = (U + 1im*V)
    YY = (I - Q)
    return CoherencyMatrix(XX, YX, XY, YY, LinBasis(), LinBasis())
end



@inline function StokesParams(c::CoherencyMatrix{CirBasis, CirBasis})
    I = (c.e11 + c.e22)/2
    Q = (c.e21 + c.e12)/2
    U = 1im*(c.e21 - c.e12)/2
    V = (c.e11 - c.e22)/2
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


export linearpol, mbreve, m̆, evpa
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
m̆(m::StokesParams{Complex{T}}) where {T} = (m.Q + 1im*m.U)/(m.I + eps(T))
m̆(m::CoherencyMatrix{CirBasis,CirBasis}) = 2*m.e12/(m.e11+m.e22)
# m̆(m::CoherencyMatrix{CirBasis,CirBasis}) = 2*m.e12/(m.e11+m.e22)
mbreve(m::Union{StokesParams, CoherencyMatrix}) = m̆(m)

"""
    $(SIGNATURES)
Compute the evpa of a stokes vector or cohereny matrix.
"""
evpa(m::StokesParams) = 1/2*atan(m.U, m.Q)
evpa(m::StokesParams{<:Complex}) = 1/2*angle(m.U/m.Q)
evpa(m::CoherencyMatrix{CirBasis, CirBasis}) = angle(m.e12)


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
