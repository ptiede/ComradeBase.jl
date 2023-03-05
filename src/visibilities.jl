function extract_pos(p::NamedTuple)
    return p.U, p.V, p.T, p.F
end

function extract_pos(p::NamedTuple{(:U,:V)})
    return p.U, p.V, zero(eltype(p.U)), zero(eltype(p.V))
end

"""
    visibilities(m, p)

Computes the visibilities of the model `m` using the coordinates `p`. The coordinates `p`
are expected to have the properties `U`, `V`, and sometimes `T` and `F`.
"""
@inline function visibilities(m::M, p::NamedTuple) where {M<:AbstractModel}
    U, V, T, F = extract_pos(p)
    return _visibilities(visanalytic(M), m, U, V, T, F)
end
@inline _visibilities(::IsAnalytic,  m::AbstractModel, U, V, T, F)  = visibilities_analytic(m, U, V, T, F)
@inline _visibilities(::NotAnalytic, m::AbstractModel, U, V, T, F) = visibilities_numeric(m, U, V, T, F)

function visibilities_analytic(m::AbstractModel, u, v, t, f)
    vis = visibility_point.(Ref(m), u, v, t, f)
    return vis
end


"""
    visibilities!(vis, m, p)

Computes the visibilities `vis` in place of the model `m` using the coordinates `p`. The coordinates `p`
are expected to have the properties `U`, `V`, and sometimes `T` and `F`.
"""
@inline function visibilities!(vis::AbstractArray, m::M, p::NamedTuple) where {M<:AbstractModel}
    U, V, T, F = extract_pos(p)
    return _visibilities!(visanalytic(M), vis, m, U, V, T, F)
end
@inline _visibilities!(::IsAnalytic , vis::AbstractArray, m::AbstractModel, U, V, T, F)  = visibilities_analytic!(vis, m, U, V, T, F)
@inline _visibilities!(::NotAnalytic, vis::AbstractArray, m::AbstractModel, U, V, T, F)  = visibilities_numeric!(vis, m, U, V, T, F)

function visibilities_analytic!(vis::AbstractArray, m::AbstractModel, u, v, t, f)
    vis .= visibility_point.(Ref(m), u, v, t, f)
    return nothing
end
