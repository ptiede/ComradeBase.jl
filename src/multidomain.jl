export ConstParam, TaylorSpectral, getparam, @unpack_params

abstract type DomainParams end

abstract type FrequencyParams <: DomainParams end
abstract type TimeParams <: DomainParams end

struct TaylorSpectral{P,T,F<:Real} <: FrequencyParams
    param::P
    index::T
    freq0::F
end

@inline function getparam(m, s::Symbol, p)
    ps = getproperty(m, s)
    return build_param(ps, p)
end
@inline function getparam(m, ::Val{s}, p) where {s}
    return getparam(m, s, p)
end

@fastmath @inline function build_param(model::TaylorSpectral, p)
    param = model.param * (p.Fr / model.freq0)^model.index
    return param
end

@inline function build_param(param::Any, p)
    return param
end

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
