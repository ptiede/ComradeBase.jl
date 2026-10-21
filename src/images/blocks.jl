"""
    _blockdomain(vis::StructuredDomain, pdims::Tuple, nblocks::Tuple)

Returns `vis` laid out for image planes split into blocks: plane dim `pdims[k]` into
`nblocks[k]` blocks of consecutive planes. Each point belongs to the block of the planes its
coordinate named like `pdims[k]` falls in. Blocks are numbered row-major over `pdims`, and
the points of block `b` fill the `b`-th of `prod(nblocks)` equal ranges of `Node`, keeping
their order, with padding slots at the end of each range. The `Node` lookup holds each
slot's original `Node` value, `0` for padding. The `valid` coordinate spans `Node` and is
`false` in padding slots, which also have zero baselines and a block coordinate inside the
block's first plane.
"""
function _blockdomain(vis::StructuredDomain, pdims::Tuple, nblocks::Tuple)
    nodes = parent(DD.lookup(first(dims(vis))))
    any(iszero, nodes) && throw(
        ArgumentError("the visibility domain's `Node` lookup contains 0, which marks padding slots; blocking needs a domain whose points are all real")
    )
    vis = _nodevalid(vis)
    cs = coords(vis)
    Base.require_one_based_indexing(nodes, values(cs)..., map(DD.lookup, pdims)...)
    sp = coordspans(vis)
    names = map(DD.name, pdims)
    perblock = map(pdims, nblocks) do pd, n
        k = DD.name(pd)
        haskey(cs, k) || throw(
            ArgumentError("the image's `$k` planes are split over devices, but the visibility domain has no `$k` coordinate to assign its points to them")
        )
        sp[k] == (:Node,) || throw(
            ArgumentError("the image's `$k` planes are split over devices, so the visibility domain's `$k` coordinate must span only `Node`, got $(sp[k])")
        )
        length(pd) % n == 0 || throw(
            ArgumentError("the image's `$k` dim has $(length(pd)) planes, which do not split evenly over its $n shards")
        )
        length(pd) ÷ n
    end
    planeblocks = map((pd, q) -> cld.(frameindex(pd, cs[DD.name(pd)]), q), pdims, perblock)
    L = LinearIndices(reverse(nblocks))
    block = map(j -> L[reverse(map(b -> b[j], planeblocks))...], eachindex(first(planeblocks)))
    issorted(block) || throw(
        ArgumentError("splitting the image's $(join(map(n -> "`$n`", names), ", ")) planes over devices needs the visibility points grouped by the device that holds their planes; sort the data by $(join(map(n -> "`$n`", names), " and then "))")
    )

    counts = zeros(Int, length(L))
    for b in block
        counts[b] += 1
    end
    Mb = maximum(counts)
    slots = zeros(Int, Mb * length(L))
    j = firstindex(block) - 1
    for b in eachindex(counts), i in 1:counts[b]
        slots[(b - 1) * Mb + i] = j += 1
    end
    pads = findall(==(0), slots)
    src = copy(slots)
    prev = firstindex(block)
    for s in eachindex(src)
        src[s] == 0 ? (src[s] = prev) : (prev = src[s])
    end
    blockindex = CartesianIndices(reverse(nblocks))
    # The offset of the first plane of the block holding slot `s`, along plane dim `q`.
    firstplane(s, q) = (reverse(Tuple(blockindex[cld(s, Mb)]))[q] - 1) * perblock[q]

    newcs = map(keys(cs), values(cs), values(sp)) do k, c, span
        p = findfirst(==(:Node), span)
        p === nothing && return c
        a = c[ntuple(i -> i == p ? src : Colon(), ndims(c))...]
        if k in (:U, :V, :W, :u, :v, :w)
            fill!(selectdim(a, p, pads), zero(eltype(a)))
        elseif k === :valid
            fill!(selectdim(a, p, pads), false)
        elseif k in names
            q = findfirst(==(k), names)
            l = DD.lookup(DD.format(pdims[q]))
            inside = _insideplanes(DD.Lookups.sampling(l), l)
            for s in pads
                a[s] = inside[firstplane(s, q) + 1]
            end
        end
        return a
    end
    nodedim = Node([s == 0 ? zero(eltype(nodes)) : nodes[s] for s in slots])
    newdims = _format_structured_dims((nodedim, Base.tail(dims(vis))...))
    return StructuredDomain(newdims, NamedTuple{keys(cs)}(newcs), sp, executor(vis), header(vis))
end

# For each plane of `l`, a coordinate inside it and no other: the midpoint of its interval,
# or its value.
_insideplanes(::DD.Lookups.Intervals, l) = map(b -> sum(b) / 2, DD.Lookups.intervalbounds(l))
_insideplanes(::Any, l) = collect(l)

# `d` with a `valid` coordinate spanning `Node` and the dims the given one spans: the given
# one, repeated along `Node`, or all `true`.
function _nodevalid(d::StructuredDomain)
    cs = coords(d)
    old = haskey(cs, :valid) ? coordspans(d).valid : ()
    :Node in old && return d
    span = filter(n -> n === :Node || n in old, keys(d))
    sz = map(n -> size(d)[findfirst(==(n), keys(d))], span)
    valid = Array{Bool}(undef, sz)
    valid .= haskey(cs, :valid) ? reshape(cs.valid, map((n, l) -> n in old ? l : 1, span, sz)) : true
    return rebuild(d; coords = merge(cs, (; valid)), spans = merge(coordspans(d), (; valid = span)))
end

# The values of `x` laid out on the points of `d`, with `pad` in the padding slots.
function _blockvalues(x::StructuredMap, d::StructuredDomain, pad)
    xd = axisdims(x)
    map(DD.name, dims(xd)) == keys(d) && Base.tail(dims(xd)) == Base.tail(dims(d)) || throw(
        DimensionMismatch("the map's dims $(keys(xd)) must equal the domain's dims $(keys(d)), apart from the `Node` lookup")
    )
    slots = parent(DD.lookup(first(dims(d))))
    real = findall(!=(0), slots)
    collect(parent(DD.lookup(first(dims(xd))))) == slots[real] || throw(
        ArgumentError("the map is not defined on the points of the domain: its `Node` lookup must equal the domain's `Node` lookup without the padding slots (0)")
    )
    v = baseimage(x)
    Base.require_one_based_indexing(v)
    y = similar(v, (length(slots), Base.tail(size(v))...))
    copyto!(selectdim(y, 1, real), v)
    fill!(selectdim(y, 1, findall(==(0), slots)), pad)
    return y
end
