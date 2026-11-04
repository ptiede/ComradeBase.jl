"""
    coherency(c::CoherencyMap, a, b)

Returns the coherency element `e_ab` of `c` (`a`, `b` ∈ 1:2, the feed of antenna a and of
antenna b) as an unpolarized `IntensityMap` over the same domain. For `FieldDimArray` and
`StructArray` data this is a view of the data; other arrays are copied.
"""
@inline function coherency(c::CoherencyMap, a::Integer, b::Integer)
    (a in 1:2 && b in 1:2) || throw(ArgumentError("feed indices must be 1 or 2; got ($a, $b)"))
    return IntensityMap(_element(baseimage(c), a + 2 * (b - 1)), axisdims(c), refdims(c), DD.name(c))
end

_element(x::FieldDimArray, k) = fieldview(x, k)
_element(x::StructArray, k) = StructArrays.component(x, k)
_element(x::AbstractArray, k) = getindex.(x, k)

"""
    coherencymap(img::StokesMap, basis)

Returns the [`CoherencyMap`](@ref) over the domain of `img` whose elements are the coherency
matrices of the Stokes parameters of `img` in `basis`, used for both feeds. `basis` is
`CirBasis()` (`e11 = I + V`, `e21 = Q - iU`, `e12 = Q + iU`, `e22 = I - V`) or `LinBasis()`
(`e11 = I + Q`, `e21 = U - iV`, `e12 = U + iV`, `e22 = I - Q`). The element type is complex.
[`stokesmap`](@ref) is the inverse.
"""
function coherencymap(img::StokesMap, basis)
    _check_basis(basis)
    return _coherencypoint.(Ref(basis), img)
end

"""
    stokesmap(c::CoherencyMap, basis)

Returns the [`StokesMap`](@ref) over the domain of `c` holding the Stokes parameters of the
coherency matrices of `c` in `basis`, `CirBasis()` or `LinBasis()`. This is the inverse of
[`coherencymap`](@ref); the element type of `c` is kept, so the result is complex.
"""
function stokesmap(c::CoherencyMap, basis)
    _check_basis(basis)
    return _stokespoint.(Ref(basis), c)
end

"""
    stokesmap(I::IntensityMap, Q, U, V)
    stokesmap(I::AbstractArray, Q, U, V, domain)

Returns the [`StokesMap`](@ref) whose Stokes parameters are the maps `I`, `Q`, `U` and `V`,
which must share one domain, or the arrays `I`, `Q`, `U` and `V` on `domain`. The components
are copied into one dense array whose last dimension holds I, Q, U and V.
"""
@inline function stokesmap(I::IntensityMap, Q::IntensityMap, U::IntensityMap, V::IntensityMap)
    axisdims(I) == axisdims(Q) == axisdims(U) == axisdims(V) ||
        throw(ArgumentError("the Stokes I, Q, U and V maps must share one domain"))
    data = _stokesparams(baseimage(I), baseimage(Q), baseimage(U), baseimage(V))
    return IntensityMap(data, axisdims(I), refdims(I), DD.name(I))
end

@inline function stokesmap(
        I::AbstractArray, Q::AbstractArray, U::AbstractArray, V::AbstractArray,
        domain::AbstractSingleDomain
    )
    return IntensityMap(_stokesparams(I, Q, U, V), domain)
end

_stokesparams(I, Q, U, V) = FieldDimArray{StokesParams}(cat(I, Q, U, V; dims = Val(ndims(I) + 1)))

_check_basis(::Union{CirBasis, LinBasis}) = nothing
@noinline function _check_basis(basis)
    throw(
        ArgumentError(
            "the polarization basis $basis is not supported; the supported bases are CirBasis() and LinBasis()"
        )
    )
end

# Multiplication by the imaginary unit, for real or complex `x`, in the precision of `x`.
_muli(x) = complex(-imag(x), real(x))

function _coherencypoint(::CirBasis, s)
    return SMatrix{2, 2}(complex(s.I + s.V), s.Q - _muli(s.U), s.Q + _muli(s.U), complex(s.I - s.V))
end
function _coherencypoint(::LinBasis, s)
    return SMatrix{2, 2}(complex(s.I + s.Q), s.U - _muli(s.V), s.U + _muli(s.V), complex(s.I - s.Q))
end
function _stokespoint(::CirBasis, c)
    return StokesParams((c[1, 1] + c[2, 2]) / 2, (c[2, 1] + c[1, 2]) / 2, _muli(c[2, 1] - c[1, 2]) / 2, (c[1, 1] - c[2, 2]) / 2)
end
function _stokespoint(::LinBasis, c)
    return StokesParams((c[1, 1] + c[2, 2]) / 2, (c[1, 1] - c[2, 2]) / 2, (c[2, 1] + c[1, 2]) / 2, _muli(c[2, 1] - c[1, 2]) / 2)
end
