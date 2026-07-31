export getparam, @unpack_params, build_param

"""
    abstract type DomainParams{T}

A parameter family that is evaluated at a point in the time/frequency domain rather than
being a fixed value. This extends models defined in the image and visibility domains so
that their parameters may also vary across time and frequency.

`T` is the element type of the value the family produces at a point: a single parameter
value such as a `Number` or a `StokesParams`, never a container. See [`paramtype`](@ref).

To define your own family, subtype `DomainParams` and define [`build_param`](@ref):

```julia
struct MyDomainParam{T} <: DomainParams{T}
    scale::T
end
build_param(param::MyDomainParam, p) = param.scale * p.Fr
```

where `p` is the point the family is evaluated at. To use the family inside a chain that
transforms a base value, also define the three-argument
`build_param(base, param::MyDomainParam, p)`.

To evaluate the family at a point `p` use `build_param(param, p)` or just `param(p)`. A
model parameterized by a `DomainParams` should read its parameters with [`getparam`](@ref)
or the [`@unpack_params`](@ref) macro.
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
    target [`build_param`](@ref).

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
    build_param(base, param, p)

Construct the value of `param` at the point `p` in the (X/U, Y/V, Ti, Fr) domain. A value
that is not a [`DomainParams`](@ref) is returned unchanged.

The two-argument form is required for any `<:DomainParams` and returns the parameter on its
own. The three-argument form transforms an externally supplied `base` value and is required
for any family used inside a chain that applies several models in turn; it must not alias
`base`.
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
