export getparam, @unpack_params, build_param, apply_param, paramfield

"""
    abstract type DomainParams{T}

A parameter family that is evaluated at a point in the time/frequency domain rather than
being a fixed value. This extends models defined in the image and visibility domains so
that their parameters may also vary across time and frequency.

`T` is the element type of the value the family produces at a point: a single parameter
value such as a `Number` or a `StokesParams`, never a container. See [`paramtype`](@ref).

A family is a *transformation* of a base value, not a value on its own: it says how a
parameter departs from a reference as time and frequency change. Subtype `DomainParams` and
define [`apply_param`](@ref), optionally splitting off the part that depends on the domain
alone as [`paramfield`](@ref):

```julia
struct MyDomainParam{T} <: DomainParams{T}
    scale::T
end
paramfield(param::MyDomainParam, p) = param.scale .* p.Fr
apply_param(base, param::MyDomainParam, field, p) = base .* field
```

where `p` is the point the family is evaluated at. Splitting out `paramfield` lets the chain
evaluate it once per frequency rather than once per point of the result; a family with no
domain-only part defines `apply_param` alone and ignores `field`.

A family becomes an evaluable parameter only when it is paired with a base value, and
several may be chained to compose in order — the modeling package supplies the container
that does this (`MultiDomainParams` in VLBISkyModels). Evaluate the result with
[`build_param`](@ref) at a point `p`, or read it off a model with [`getparam`](@ref) or the
[`@unpack_params`](@ref) macro. A bare family has no value: `build_param` on one is an
error, because the base it transforms is missing.
"""
abstract type DomainParams{T} end

"""
    paramtype(::Type)

The element type of a single parameter value: what a [`DomainParams`](@ref) produces when
evaluated at a point. A type that is not a `DomainParams` returns its own element type.

A parameter is either a single value or a field of values over the image grid. An
`AbstractArray` is a field, so it unwraps to its element type; a `StaticArray` is a single
value, since a polarized parameter (`StokesParams <: FieldVector`) is itself a static
vector. This is the only place that distinction is made — every operation that must tell a
field from a value dispatches on it and nothing else.

So `paramtype(Matrix{Float64}) === Float64`, while
`paramtype(StokesParams{Float64}) === StokesParams{Float64}` and
`paramtype(Matrix{StokesParams{Float64}}) === StokesParams{Float64}`.
"""
@inline paramtype(::Type{T}) where {T} = T
@inline paramtype(::Type{<:DomainParams{T}}) where {T} = paramtype(T)
@inline paramtype(::Type{<:AbstractArray{T}}) where {T} = paramtype(T)
@inline paramtype(::Type{T}) where {T <: StaticArray} = T

"""
    getparam(m, s::Symbol, p)

Gets the parameter value `s` from the model `m` evaluated at the domain `p`. 
This is similar to getproperty, but allows for the parameter to be a function of the 
domain. Essentially is `m.s <: DomainParams` then `m.s` is evaluated at the parameter `p`.
If `m.s` is not a subtype of `DomainParams` then `m.s` is returned.

!!! warn
    Developers should not typically overload this function and instead
    target [`apply_param`](@ref).

!!! warn
    This feature is experimental and is not considered part of the public stable API.

"""
@inline function getparam(m, s::Symbol, p)
    ps = getproperty(m, s)
    return build_param(ps, p)
end
@inline function getparam(m, ::Val{s}, p) where {s}
    return getparam(m, s, p)
end

"""
    build_param(param, p)

The value of `param` at the point `p` in the (X/U, Y/V, Ti, Fr) domain. Anything that is not
a [`DomainParams`](@ref) is already a value and is returned unchanged.

This is closed: families do not extend it. A `DomainParams` acquires a value only once it is
paired with a base, so evaluating a bare one is an error — see [`apply_param`](@ref).
"""
@inline function build_param(param::Any, p)
    return param
end

function build_param(param::NTuple, p)
    return map(x -> build_param(x, p), param)
end

function build_param(param::AbstractArray{<:DomainParams}, p)
    return map(x -> build_param(x, p), param)
end

# Without this a family falls through to the pass-through above and silently returns itself.
function build_param(param::DomainParams, p)
    throw(
        ArgumentError(
            "$(typeof(param)) transforms a base value and has none of its own; pair it " *
                "with one, e.g. `MultiDomainParams(base, param)`."
        )
    )
end

"""
    paramfield(param::DomainParams, p)

The part of `param` that depends on the domain alone, with no reference to a base — for a
spectral model, the factor as a function of frequency. Returns `nothing` by default, for a
family with no such part.

A chain evaluates this once per model and hands the result to [`apply_param`](@ref), so work
that depends on only some of the domain axes is done once per axis point rather than once
per point of the full result. The whole frequency axis may arrive in `p` at once, and that
difference is typically one or two orders of magnitude on a cube — splitting it out here is
what makes it automatic rather than something each family has to remember.

Returning a lazy `Base.Broadcasted` opts back out of the caching, which is worth doing only
when the field is already as large as the result and would gain nothing from being reused.
"""
paramfield(param::DomainParams, p) = nothing

"""
    apply_param(base, param::DomainParams, field, p)

Apply `param` to the running value `base` at the point `p`, returning the transformed value.
`field` is what [`paramfield`](@ref) produced for this model and point. This is one link of a
chain: what it returns becomes the `base` of the next link, and it must not alias `base`.

Together with `paramfield` this is all a family defines. Write it with ordinary broadcasting;
a family with no domain-only part ignores `field`:

```julia
apply_param(base, param::MyDrift, _, p) = base .+ param.v .* p.Ti
```

There is no general relation between this and [`build_param`](@ref): the identity element of
the transformation is family-specific, so a value cannot be derived from a transformation
without one.
"""
function apply_param end

function (m::DomainParams{T})(p) where {T}
    return build_param(m, p)
end

"""
    @unpack_params a,b,c,... = m(p)

Extracts the parameters `a,b,c,...` from the model `m` evaluated at the domain `p`.
This is a macro that essentially lowers to 
```julia
a = getparam(m, :a, p)
b = getparam(m, :b, p)
...
```
For any model that may depend on a `DomainParams` type this macro should be used to 
extract the parameters. 

!!! warn
    This feature is experimental and is not considered part of the public stable API.

"""
macro unpack_params(args)
    args.head != :(=) &&
        throw(ArgumentError("Expression needs to be of the form a, b, = c(p)"))
    items, suitcase = args.args
    items = isa(items, Symbol) ? [items] : items.args
    hasproperty(suitcase, :head) ||
        throw(ArgumentError("RHS of expression must be of form m(p)"))
    suitcase.head != :call && throw(ArgumentError("RHS of expression must be of form m(p)"))
    m, p = suitcase.args[1], suitcase.args[2]
    paraminstance = gensym()
    kp = [
        :($key = getparam($paraminstance, Val{$(Expr(:quote, key))}(), $p))
            for key in items
    ]
    kpblock = Expr(:block, kp...)
    expr = quote
        local $paraminstance = $m
        $kpblock
        $paraminstance
    end
    return esc(expr)
end
