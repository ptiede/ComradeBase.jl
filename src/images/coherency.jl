"""
    coherency(c::CoherencyMap, a, b)

Returns the coherency element `e_ab` of `c` (`a`, `b` ∈ 1:2, the feed of antenna a and of
antenna b) as an unpolarized `IntensityMap` over the same domain that is a view of `c`.
"""
@inline function coherency(c::CoherencyMap, a, b)
    return view(c, Fa(a), Fb(b))
end

"""
    coherencymap(img::StokesMap, basis)

Returns the [`CoherencyMap`](@ref) over the domain of `img` whose elements are the coherency
matrices of the Stokes parameters of `img` in `basis`, used for both feeds. `basis` is
`CirBasis()` (`e11 = I + V`, `e21 = Q - iU`, `e12 = Q + iU`, `e22 = I - V`) or `LinBasis()`
(`e11 = I + Q`, `e21 = U - iV`, `e12 = U + iV`, `e22 = I - Q`). The element type is complex.
[`stokesmap`](@ref) is the inverse and [`coherencymap!`](@ref) the in-place form.
"""
function coherencymap(img::StokesMap, basis)
    _check_basis(basis)
    P = baseimage(img)
    dest = _rewrap(img, similar(P, complex(_numbertype(P)), size(P)))
    _convert!(_coherencypoint, basis, _stokesslabs(dest), _stokesslabs(img), executor(img))
    storage = reshape(baseimage(dest), (size(axisdims(img))..., 2, 2))
    return IntensityMap(storage, axisdims(img), Fa(), Fb(); refdims = refdims(img), name = DD.name(img))
end

"""
    stokesmap(c::CoherencyMap, basis)

Returns the [`StokesMap`](@ref) over the domain of `c` holding the Stokes parameters of the
coherency matrices of `c` in `basis`, `CirBasis()` or `LinBasis()`. This is the inverse of
[`coherencymap`](@ref); the element type of `c` is kept, so the result is complex.
[`stokesmap!`](@ref) is the in-place form.
"""
function stokesmap(c::CoherencyMap, basis)
    _check_basis(basis)
    P = baseimage(c)
    storage = similar(P, typeof(zero(_numbertype(P)) / 2), (size(axisdims(c))..., 4))
    dest = IntensityMap(storage, axisdims(c), Stokes(); refdims = refdims(c), name = DD.name(c))
    _convert!(_stokespoint, basis, _stokesslabs(dest), _coherencyslabs(c), executor(c))
    return dest
end

"""
    coherencymap!(img::StokesMap, basis)

The in-place form of [`coherencymap`](@ref): overwrites the storage of `img` with the
coherency elements in `basis` and returns a [`CoherencyMap`](@ref) whose storage is a
`reshape` of that storage (Stokes slot `k` becomes the feed pair `(a, b)` with
`k = a + 2(b - 1)`). The storage of `img` is consumed: `img` must not be used as a Stokes map
afterwards. The storage must have a complex element type; for real storage use
[`coherencymap`](@ref).
"""
function coherencymap!(img::StokesMap, basis)
    _check_basis(basis)
    _check_complex(baseimage(img), "coherencymap!", "coherencymap")
    _convert!(_coherencypoint, basis, _stokesslabs(img), executor(img))
    storage = reshape(baseimage(img), (size(axisdims(img))..., 2, 2))
    return IntensityMap(storage, axisdims(img), Fa(), Fb(); refdims = refdims(img), name = DD.name(img))
end

"""
    stokesmap!(c::CoherencyMap, basis)

The in-place form of [`stokesmap`](@ref): overwrites the storage of `c` with the Stokes
parameters in `basis` and returns a [`StokesMap`](@ref) whose storage is a `reshape` of that
storage. The storage of `c` is consumed: `c` must not be used as a coherency map afterwards.
The storage must have a complex element type. Under Reactant this throws an `ArgumentError`;
use [`stokesmap`](@ref) there.
"""
function stokesmap!(c::CoherencyMap, basis)
    _check_basis(basis)
    _check_complex(baseimage(c), "stokesmap!", "stokesmap")
    _check_stokesmap!(executor(c))
    _convert!(_stokespoint, basis, _coherencyslabs(c), executor(c))
    storage = reshape(baseimage(c), (size(axisdims(c))..., 4))
    return IntensityMap(storage, axisdims(c), Stokes(); refdims = refdims(c), name = DD.name(c))
end

_check_basis(::Union{CirBasis, LinBasis}) = nothing
@noinline function _check_basis(basis)
    throw(
        ArgumentError(
            "the polarization basis $basis is not supported; the supported bases are CirBasis() and LinBasis()"
        )
    )
end

_check_stokesmap!(executor) = nothing
function _check_stokesmap!(::ReactantEx)
    throw(
        ArgumentError(
            "`stokesmap!` does not run under Reactant, because XLA crashes when compiling it for a sharded complex map; use `stokesmap` instead"
        )
    )
end

# The number type of the storage; Reactant arrays have traced element types.
_numbertype(P) = eltype(P)

function _check_complex(P, f, outofplace)
    _numbertype(P) <: Complex || throw(
        ArgumentError(
            "`$f` needs complex storage, but the map has element type $(eltype(P)); use `$outofplace` instead"
        )
    )
    return nothing
end

# The four components of a map as views of its storage, in storage order.
_stokesslabs(img) = (parent(stokes(img, :I)), parent(stokes(img, :Q)), parent(stokes(img, :U)), parent(stokes(img, :V)))
_coherencyslabs(c) = (parent(coherency(c, 1, 1)), parent(coherency(c, 2, 1)), parent(coherency(c, 1, 2)), parent(coherency(c, 2, 2)))

# Writes `f(basis, values...)` of the four arrays `src` at every index into the four arrays
# `dest`, which may be `src`: a point is read before it is written. Executors `@threaded`
# does not handle run the loop serially.
function _convert!(f::F, basis, dest, src, executor) where {F}
    s1, s2, s3, s4 = src
    d1, d2, d3, d4 = dest
    @threaded _threadsex(executor) for i in eachindex(s1, s2, s3, s4, d1, d2, d3, d4)
        d1[i], d2[i], d3[i], d4[i] = f(basis, s1[i], s2[i], s3[i], s4[i])
    end
    return nothing
end

_threadsex(ex::ThreadsEx{S}) where {S} = S in schedulers ? ex : Serial()
_threadsex(ex) = Serial()

function _broadcastconvert!(f::F, basis, dest, src) where {F}
    for k in 1:4
        dest[k] .= ComponentFn(f, k).(Ref(basis), src...)
    end
    return nothing
end

# `dest` are the four slabs of one array. Under Reactant, writes into the slabs of a 2-d array
# lower to `scatter`; building the whole array with `cat` does not, and computes all four
# components before writing, so `dest` may be `src`.
function _convert!(f::F, basis, dest, src, ::ReactantEx) where {F}
    P = parent(first(dest))
    c1, c2 = ComponentFn(f, 1).(Ref(basis), src...), ComponentFn(f, 2).(Ref(basis), src...)
    c3, c4 = ComponentFn(f, 3).(Ref(basis), src...), ComponentFn(f, 4).(Ref(basis), src...)
    P .= cat(c1, c2, c3, c4; dims = Val(ndims(P)))
    return nothing
end

# In place on the slabs `s`. A broadcasting executor writes one component at a time, so the
# KernelAbstractions method reads from copies of `s`.
_convert!(f::F, basis, s, executor) where {F} = _convert!(f, basis, s, s, executor)

# Multiplication by the imaginary unit, for real or complex `x`, in the precision of `x`.
_muli(x) = complex(-imag(x), real(x))

# Point formulas; values are in storage order (I, Q, U, V) and (e11, e21, e12, e22).
_coherencypoint(::CirBasis, I, Q, U, V) = SVector(complex(I + V), Q - _muli(U), Q + _muli(U), complex(I - V))
_coherencypoint(::LinBasis, I, Q, U, V) = SVector(complex(I + Q), U - _muli(V), U + _muli(V), complex(I - Q))
function _stokespoint(::CirBasis, e11, e21, e12, e22)
    return SVector((e11 + e22) / 2, (e21 + e12) / 2, _muli(e21 - e12) / 2, (e11 - e22) / 2)
end
function _stokespoint(::LinBasis, e11, e21, e12, e22)
    return SVector((e11 + e22) / 2, (e11 - e22) / 2, (e21 + e12) / 2, _muli(e21 - e12) / 2)
end
