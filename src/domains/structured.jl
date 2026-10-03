export StructuredDomain, Pt

DD.@dim Pt "point"

"""
    speed_of_light

The speed of light in vacuum in m/s. [`StructuredDomain`](@ref) uses it to convert
baseline coordinates `u`, `v` in meters at frequencies `Fr` in Hz to `U`, `V` in
wavelengths.
"""
const speed_of_light = 299_792_458.0

"""
    StructuredDomain(dims::Tuple; executor=Serial(), header=ComradeBase.NoHeader(), coords...)
    StructuredDomain(coords::NamedTuple; executor=Serial(), header=ComradeBase.NoHeader())

A visibility domain made of DimensionalData dims and coordinate arrays that each span a
subset of those dims.

`dims` starts with the point index dim `Pt`, followed by optional plane dims such as `Ti`
and `Fr` whose lookups give the plane labels. `Pt(n)` with an integer `n` is the index dim
`Pt(Base.OneTo(n))`.

Each keyword in `coords` is a coordinate array. Its span (the dims it varies along) is
either inferred from its size, when exactly one ordered subset of `dims` matches it, or
given explicitly as `array => (:Pt, :Ti)`, listing dim names in the order they appear in
`dims`. A coordinate's `axes` must equal the axes of the dims it spans, in that order.

The coordinates `U`, `V` (baselines in wavelengths), `Ti`, `Fr`, `valid` (a `Bool` mask)
have a fixed meaning; `u`, `v` are baselines in meters, which need an `Fr` dim (in Hz) and
give `U = u * Fr / c` and `V = v * Fr / c`. Any other coordinate is passed through to
[`domainpoints`](@ref) unchanged.

The `NamedTuple` form builds the domain `(Pt(n),)` with every coordinate a length-`n`
vector spanning `Pt`.

Properties: `d.U`, `d.V` (derived from `u`, `v` when needed, spanning the coordinate's dims
and `Fr`), every coordinate by name, and `d.Ti`, `d.Fr` as the dim lookups when no
coordinate of that name is given.

# Examples

```julia
d = StructuredDomain((Pt(100),); U = randn(100), V = randn(100))
d = StructuredDomain(
    (Pt(100), Ti([0.0, 1.0, 2.0]), Fr([230e9, 345e9]));
    u = randn(100, 3), v = randn(100, 3), Ti = rand(100, 3)
)
```
"""
struct StructuredDomain{D <: Tuple, C <: NamedTuple, S <: NamedTuple, E, H <: AMeta} <:
    AbstractSingleDomain{D, E}
    dims::D
    coords::C
    spans::S
    executor::E
    header::H
    function StructuredDomain{D, C, S, E, H}(
            dims, coords, spans, executor, header
        ) where {D, C, S, E, H}
        d = new{D, C, S, E, H}(dims, coords, spans, executor, header)
        _check_structured(d)
        return d
    end
end

function StructuredDomain(dims::Tuple, coords::NamedTuple, spans::NamedTuple, executor, header)
    return StructuredDomain{
        typeof(dims), typeof(coords), typeof(spans), typeof(executor), typeof(header),
    }(dims, coords, spans, executor, header)
end

function StructuredDomain(dims::Tuple; executor = Serial(), header = NoHeader(), coords...)
    fdims = _format_structured_dims(dims)
    cs = values(coords)
    ks = keys(cs)
    arrays = NamedTuple{ks}(map(_coord_array, values(cs)))
    spans = NamedTuple{ks}(map((k, c) -> _coord_span(k, c, fdims), ks, values(cs)))
    return StructuredDomain(fdims, arrays, spans, executor, header)
end

function StructuredDomain(nt::NamedTuple; executor = Serial(), header = NoHeader())
    isempty(nt) && throw(ArgumentError("StructuredDomain needs at least one coordinate"))
    n = length(first(values(nt)))
    return StructuredDomain((Pt(n),); executor, header, map(a -> a => (:Pt,), nt)...)
end

EnzymeRules.inactive_type(::Type{<:StructuredDomain}) = true

function _format_structured_dims(dims::Tuple)
    isempty(dims) && throw(ArgumentError("StructuredDomain needs dims starting with `Pt`"))
    for d in dims
        d isa DD.Dimension || throw(
            ArgumentError(
                "StructuredDomain dims must be DimensionalData dimensions, got a $(typeof(d))"
            )
        )
    end
    first(dims) isa Pt || throw(
        ArgumentError("the first dim of a StructuredDomain must be `Pt`, got `$(DD.name(first(dims)))`")
    )
    ds = map(_index_lookup, dims)
    for d in ds
        parent(d) isa AbstractVector || throw(
            ArgumentError(
                "dim `$(DD.name(d))` needs a vector lookup whose length gives its axis, got $(typeof(parent(d)))"
            )
        )
    end
    return DD.format(ds)
end

_index_lookup(d::DD.Dimension) = d
_index_lookup(d::Pt) = parent(d) isa Integer ? Pt(Base.OneTo(parent(d))) : d

_coord_array(p::Pair) = first(p)
_coord_array(a) = a

_coord_span(k, p::Pair, dims) = _explicit_span(k, last(p))
_coord_span(k, a::AbstractArray, dims) = _infer_span(k, a, dims)
function _coord_span(k, x, dims)
    throw(
        ArgumentError(
            "coordinate `$k` must be an array or `array => (dim names...)`, got a $(typeof(x))"
        )
    )
end

_explicit_span(k, s::Symbol) = (s,)
_explicit_span(k, s::Tuple{Vararg{Symbol}}) = s
function _explicit_span(k, s)
    throw(
        ArgumentError(
            "the span of coordinate `$k` must be a dim name or a tuple of dim names, e.g. `(:Pt, :Ti)`, got $(repr(s))"
        )
    )
end

function _infer_span(k, a::AbstractArray, dims)
    names = map(DD.name, dims)
    sizes = map(length, dims)
    candidates = Tuple{Vararg{Symbol}}[]
    for mask in 0:(2^length(dims) - 1)
        sel = [i for i in eachindex(dims) if isodd(mask >> (i - 1))]
        if length(sel) == ndims(a) && all(j -> size(a, j) == sizes[sel[j]], eachindex(sel))
            push!(candidates, Tuple(names[i] for i in sel))
        end
    end
    isempty(candidates) && throw(
        DimensionMismatch(
            "coordinate `$k` has size $(size(a)), which matches no ordered subset of the dims $names with sizes $sizes"
        )
    )
    length(candidates) > 1 && throw(
        ArgumentError(
            "the span of coordinate `$k` with size $(size(a)) is ambiguous; it could span any of $(Tuple(candidates)). Give it explicitly, e.g. `$k = $k => $(first(candidates))`"
        )
    )
    return only(candidates)
end

function _check_structured(d::StructuredDomain)
    ds = dims(d)
    names = keys(d)
    isempty(ds) && throw(ArgumentError("StructuredDomain needs dims starting with `Pt`"))
    first(ds) isa Pt || throw(
        ArgumentError("the first dim of a StructuredDomain must be `Pt`, got `$(first(names))`")
    )
    allunique(names) || throw(ArgumentError("StructuredDomain dim names must be distinct, got $names"))
    coords = getfield(d, :coords)
    spans = getfield(d, :spans)
    keys(coords) == keys(spans) || throw(
        ArgumentError("coordinate names $(keys(coords)) and span names $(keys(spans)) differ")
    )
    for k in keys(coords)
        _check_coord(k, coords[k], spans[k], ds, names)
    end
    for (a, b) in ((:U, :u), (:V, :v))
        haskey(coords, a) && haskey(coords, b) && throw(
            ArgumentError("give either `$a` in wavelengths or `$b` in meters, not both")
        )
    end
    if (haskey(coords, :u) || haskey(coords, :v)) && !(:Fr in names)
        throw(
            ArgumentError(
                "u and v in meters need a frequency axis to be converted to wavelengths; add an `Fr` dim (in Hz) or give `U` and `V` in wavelengths. Dims are $names"
            )
        )
    end
    if haskey(coords, :valid)
        eltype(coords.valid) <: Bool || throw(
            ArgumentError("coordinate `valid` must have a Bool element type, got $(eltype(coords.valid))")
        )
    end
    return nothing
end

function _check_coord(k, a, span, ds, names)
    a isa AbstractArray || throw(ArgumentError("coordinate `$k` must be an array, got a $(typeof(a))"))
    Base.require_one_based_indexing(a)
    for s in span
        s in names || throw(
            ArgumentError("coordinate `$k` spans `$s`, which is not a dim of this domain; dims are $names")
        )
    end
    pos = map(s -> findfirst(==(s), names), span)
    issorted(pos; lt = <=) || throw(
        ArgumentError(
            "coordinate `$k` spans $span; list each dim once, in the domain's dim order $names"
        )
    )
    dimaxes = map(p -> axes(parent(ds[p]), 1), pos)
    axes(a) == dimaxes || throw(
        DimensionMismatch(
            "coordinate `$k` has axes $(axes(a)), but the dims $span it spans have axes $dimaxes"
        )
    )
    return nothing
end

@inline Base.keys(d::StructuredDomain) = map(DD.name, dims(d))
Base.axes(d::StructuredDomain) = map(dm -> axes(parent(dm), 1), dims(d))

"""
    coordspans(d::StructuredDomain)

Returns a `NamedTuple` mapping each coordinate given to `d` to the tuple of dim names it
spans.
"""
coordspans(d::StructuredDomain) = getfield(d, :spans)

"""
    coords(d::StructuredDomain)

Returns the `NamedTuple` of coordinate arrays given to `d`, without derived coordinates.
"""
coords(d::StructuredDomain) = getfield(d, :coords)

function DD.rebuild(
        d::StructuredDomain, dims, coords = getfield(d, :coords),
        spans = getfield(d, :spans), executor = getfield(d, :executor),
        header = getfield(d, :header)
    )
    return StructuredDomain(dims, coords, spans, executor, header)
end

function DD.rebuild(
        d::StructuredDomain; dims = getfield(d, :dims), coords = getfield(d, :coords),
        spans = getfield(d, :spans), executor = getfield(d, :executor),
        header = getfield(d, :header)
    )
    return rebuild(d, dims, coords, spans, executor, header)
end

struct CoordColumn{A, P}
    data::A
    pos::P
end

struct WavelengthColumn{A, P, F}
    data::A
    pos::P
    freq::F
    fpos::Int
end

Base.@propagate_inbounds _value(c::CoordColumn, I) = c.data[map(p -> I[p], c.pos)...]
Base.@propagate_inbounds function _value(c::WavelengthColumn, I)
    return c.data[map(p -> I[p], c.pos)...] * c.freq[I[c.fpos]] / speed_of_light
end

_coleltype(c::CoordColumn) = eltype(c.data)
function _coleltype(c::WavelengthColumn)
    return Base.promote_op((x, f) -> x * f / speed_of_light, eltype(c.data), eltype(c.freq))
end

_colpositions(c::CoordColumn) = c.pos
_colpositions(c::WavelengthColumn) = Tuple(sort!(unique!([c.pos..., c.fpos])))

_point_name(k::Symbol) = k === :u ? :U : k === :v ? :V : k

function _columns(d::StructuredDomain)
    names = keys(d)
    ds = dims(d)
    cs = coords(d)
    sp = coordspans(d)
    position(s) = something(findfirst(==(s), names))
    cols = Pair{Symbol, Any}[]
    for k in keys(cs)
        pos = map(position, sp[k])
        col = if k === :u || k === :v
            p = position(:Fr)
            WavelengthColumn(cs[k], pos, basedim(ds[p]), p)
        else
            CoordColumn(cs[k], pos)
        end
        push!(cols, _point_name(k) => col)
    end
    for k in (:Ti, :Fr)
        if k in names && !haskey(cs, k)
            p = position(k)
            push!(cols, k => CoordColumn(basedim(ds[p]), (p,)))
        end
    end
    return (; cols...)
end

function _shaped(c::CoordColumn, target, sz)
    return reshape(c.data, map(t -> t in c.pos ? sz[t] : 1, target))
end

function _shaped(c::WavelengthColumn, target, sz)
    data = reshape(c.data, map(t -> t in c.pos ? sz[t] : 1, target))
    freq = reshape(c.freq, map(t -> t == c.fpos ? sz[t] : 1, target))
    return data .* freq ./ speed_of_light
end

_materialize(c::CoordColumn, sz) = c.data
_materialize(c::WavelengthColumn, sz) = _shaped(c, _colpositions(c), sz)

Base.propertynames(d::StructuredDomain) = keys(_columns(d))
function Base.getproperty(d::StructuredDomain, p::Symbol)
    cols = _columns(d)
    haskey(cols, p) || throw(
        ArgumentError("StructuredDomain has no property `$p`; properties are $(keys(cols))")
    )
    return _materialize(cols[p], size(d))
end

"""
    StructuredPoints

The lazy array of points returned by `domainpoints(::StructuredDomain)`. Element `I` is a
`NamedTuple` of the coordinates at that point, each read from its array at the indices
of the dims it spans.
"""
struct StructuredPoints{T, N, C <: NamedTuple} <: AbstractArray{T, N}
    columns::C
    size::Dims{N}
    function StructuredPoints(columns::NamedTuple, size::Dims{N}) where {N}
        T = NamedTuple{keys(columns), Tuple{map(_coleltype, values(columns))...}}
        return new{T, N, typeof(columns)}(columns, size)
    end
end

Base.size(p::StructuredPoints) = p.size

Base.@propagate_inbounds function Base.getindex(
        p::StructuredPoints{T, N}, I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(p, I...)
    return map(c -> _value(c, I), p.columns)
end

"""
    domainpoints(d::StructuredDomain)

Returns a lazy array with `size(d)` whose element `I` is a `NamedTuple` of the point's
`U`, `V` (derived from `u`, `v` and `Fr` when given in meters), `Ti` (the coordinate when
given, otherwise the `Ti` lookup value), `Fr` (the lookup value) and every other
coordinate.
"""
domainpoints(d::StructuredDomain) = StructuredPoints(_columns(d), size(d))

"""
    shapedcoords(d::StructuredDomain)

Returns a `NamedTuple` with the same names as the elements of `domainpoints(d)`, holding
each coordinate reshaped to `ndims(d)` dims with singleton dims where the coordinate does
not vary. Broadcasting over its values gives the full grid of points.
"""
function shapedcoords(d::StructuredDomain)
    target = ntuple(identity, ndims(d))
    return map(c -> _shaped(c, target, size(d)), _columns(d))
end
EnzymeRules.inactive(::typeof(shapedcoords), args...) = nothing

function Base.summary(io::IO, d::StructuredDomain)
    return print(io, "StructuredDomain with dims $(keys(d)) and size $(size(d))")
end

function Base.show(io::IO, mime::MIME"text/plain", d::StructuredDomain)
    println(io, "StructuredDomain(")
    println(io, "executor: $(executor(d))")
    println(io, "header: $(header(d))")
    println(io, "Dimensions: ")
    show(io, mime, dims(d))
    println(io)
    println(io, "Coordinates: ")
    for (k, s) in pairs(coordspans(d))
        println(io, "  $k :: $(summary(coords(d)[k])) spanning $s")
    end
    return print(io, ")")
end
