export frames, frameindex

"""
    frames(D, starts, stops)
    frames(D, edges)

Returns a dim of type `D` (e.g. `Ti` or `Fr`) whose planes are the intervals
`[starts[i], stops[i]]`, with an `Intervals(Center())` lookup at the interval centers and
`Explicit` bounds. Gaps between intervals are allowed; each interval must have
`start < stop`, and the intervals must be sorted and must not overlap (touching is
allowed). With `edges`, the intervals are the contiguous `[edges[i], edges[i+1]]`.

```julia
julia> scans = frames(Ti, [0.0, 1.5, 4.0], [1.0, 3.0, 5.0])

julia> bands = frames(Fr, [226e9, 228e9, 230e9])
```
"""
function frames(::Type{D}, starts::AbstractVector, stops::AbstractVector) where {D <: DD.Dimension}
    for (a, b) in zip(starts, stops)
        a < b || throw(ArgumentError("each interval needs start < stop, got [$a, $b]"))
    end
    lo = collect(starts)
    hi = collect(stops)
    for i in firstindex(lo):(lastindex(lo) - 1)
        hi[i] <= lo[i + 1] ||
            throw(ArgumentError("intervals must be sorted and must not overlap, got [$(lo[i]), $(hi[i])] before [$(lo[i + 1]), $(hi[i + 1])]"))
    end
    centers = (lo .+ hi) ./ 2
    bounds = permutedims(hcat(lo, hi))
    return D(
        DD.Lookups.Sampled(
            centers; order = DD.Lookups.ForwardOrdered(),
            span = DD.Lookups.Explicit(bounds), sampling = DD.Lookups.Intervals(DD.Lookups.Center())
        )
    )
end

function frames(::Type{D}, edges::AbstractVector) where {D <: DD.Dimension}
    return frames(D, edges[begin:(end - 1)], edges[(begin + 1):end])
end

"""
    frameindex(lookup, coords)

Returns, for each coordinate in `coords`, the index of the plane of `lookup` (a
DimensionalData lookup or dim) that it belongs to, with the axes of `coords`.

  - `Intervals` sampling: the plane whose interval `[start, stop]` contains the coordinate.
    A coordinate on the boundary shared by two touching intervals belongs to the later one.
  - any other lookup: the plane whose value equals the coordinate. Repeated values throw.

A coordinate that matches no plane throws an `ArgumentError` giving the number of such
coordinates and the first one. Comparisons happen in the coordinates' precision, so a
`Float32` time at a `Float64` interval end can fall outside it. Equality matching is meant
for planes built from the data's own values, e.g. a grid `g ⊗ dims(visdomain, Fr)`; use
[`frames`](@ref) for times.
"""
frameindex(d::DD.Dimension, coords::AbstractArray) = frameindex(DD.lookup(DD.format(d)), coords)
frameindex(l::DD.Lookups.Lookup, coords::AbstractArray) = _frameindex(DD.Lookups.sampling(l), l, coords)

function _frameindex(::DD.Lookups.Intervals, l, coords)
    planes = collect(eachindex(l))
    bounds = DD.Lookups.intervalbounds(l)
    order = sortperm(bounds; by = first)
    lo = [first(bounds[k]) for k in order]
    hi = [last(bounds[k]) for k in order]
    return _assignframes(coords) do c
        k = searchsortedlast(lo, c)
        (k >= firstindex(lo) && c <= hi[k]) ? planes[order[k]] : nothing
    end
end

function _frameindex(::Any, l, coords)
    plane = Dict{eltype(l), Int}()
    for (i, v) in pairs(l)
        k = _eqkey(v)
        haskey(plane, k) && throw(ArgumentError("the lookup repeats the value $v, so a coordinate equal to it has no unique plane"))
        plane[k] = i
    end
    return _assignframes(c -> get(plane, _eqkey(c), nothing), coords)
end

# `Dict` compares with `isequal`, which separates -0.0 from 0.0.
_eqkey(v::Number) = v + zero(v)
_eqkey(v) = v

function _assignframes(find, coords)
    inds = similar(coords, Int)
    nmiss = 0
    firstmiss = nothing
    for i in eachindex(coords, inds)
        k = find(coords[i])
        if k === nothing
            nmiss += 1
            firstmiss === nothing && (firstmiss = coords[i])
        else
            inds[i] = k
        end
    end
    nmiss == 0 ||
        throw(ArgumentError("$nmiss of $(length(coords)) coordinates match no plane of the lookup; the first is $firstmiss"))
    return inds
end
