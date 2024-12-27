export ConstParam, TaylorSpectral, getparam, @unpack_params

abstract type DomainParams end

abstract type FrequencyParams <: DomainParams end
abstract type TimeParams <: DomainParams end

struct TaylorSpectral{N,P,T<:NTuple{N},F<:Real} <: FrequencyParams
    param::P
    index::T
    freq0::F
end

"""
    TaylorSpectral(param, index::NTuple{N}, freq0::Real) -> TaylorSpectral{N}

Creates a frequency model that expands the parameter in a Taylor series defined by 
    `param * exp(∑ index[i] * log(Fr / freq0)^i)`.
i.e. an expansion in log(Fr / freq0) where Fr is the frequency of the observation, 
`freq0` is the reference frequency, `param` is the parameter value at `freq0`.

The `N` in index defines the order of the Taylor expansion. If `index` is a `<:Real`
then the expansion is of order 1.
"""
TaylorSpectral(param, index::Real, freq0) = TaylorSpectral(param, (index,), freq0)

"""
    getparam(m, s::Symbol, p)

Gets the parameter value `s` from the model `m` evaluated at the domain `p`. 
This is similar to getproperty, but allows for the parameter to be a function of the 
domain. Essentially is `m.s <: DomainParams` then `m.s` is evaluated at the parameter `p`.
If `m.s` is not a subtype of `DomainParams` then `m.s` is returned.
"""
@inline function getparam(m, s::Symbol, p)
    ps = getproperty(m, s)
    return build_param(ps, p)
end
@inline function getparam(m, ::Val{s}, p) where {s}
    return getparam(m, s, p)
end

@fastmath @inline function build_param(model::TaylorSpectral{N}, p) where {N}
    lf = log(p.Fr / model.freq0)
    arg = reduce(+, ntuple(n -> @inbounds(model.index[n]) * lf^n, Val(N)))
    param = model.param * exp(arg)
    return param
end

@inline function build_param(param::Any, p)
    return param
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
"""
macro unpack_params(args)
    args.head != :(=) &&
        throw(ArgumentError("Expression needs to be of the form a, b, = c(p)"))
    items, suitcase = args.args
    items = isa(items, Symbol) ? [items] : items.args
    suitcase.head != :call && throw(ArgumentError("RHS of expression must be of form c(p)"))
    m, p = suitcase.args[1], suitcase.args[2]
    paraminstance = gensym()
    kp = [:($key = getparam($paraminstance, Val{$(Expr(:quote, key))}(), $p))
          for key in items]
    kpblock = Expr(:block, kp...)
    expr = quote
        local $paraminstance = $m
        $kpblock
        $paraminstance
    end
    return esc(expr)
end
