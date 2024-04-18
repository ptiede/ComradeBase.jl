export visibilities, visibilities!,
      logclosure_amplitude, logclosure_amplitudemap,
      amplitude, amplitudemap,
      closure_phase, closure_phasemap,
      bispectrum, bispectrummap

function extract_pos(p::NamedTuple)
    return p.U, p.V, p.T, p.F
end

function extract_pos(p::NamedTuple{(:U,:V)})
    return p.U, p.V, zero(eltype(p.U)), zero(eltype(p.V))
end

"""
    visibilitymap(m, p)

Computes the visibilities of the model `m` using the coordinates `p`. The coordinates `p`
are expected to have the properties `U`, `V`, and sometimes `T` and `F`.
"""
@inline function visibilitymap(m::M, p) where {M<:AbstractModel}
    return create_map(_visibilitymap(visanalytic(M), m, p), p)
end
@inline _visibilitymap(::IsAnalytic,  m::AbstractModel, p)  = visibilitymap_analytic(m, p)
@inline _visibilitymap(::NotAnalytic, m::AbstractModel, p)  = visibilitymap_numeric(m, p)



"""
    visibilitymap!(vis, m, p)

Computes the visibilities `vis` in place of the model `m` using the coordinates `p`. The coordinates `p`
are expected to have the properties `U`, `V`, and sometimes `T` and `F`.
"""
@inline function visibilitymap!(vis, m::M) where {M<:AbstractModel}
    return _visibilitymap!(visanalytic(M), vis, m)
end
@inline _visibilitymap!(::IsAnalytic , vis, m::AbstractModel)  = visibilitymap_analytic!(vis, m)
@inline _visibilitymap!(::NotAnalytic, vis, m::AbstractModel)  = visibilitymap_numeric!(vis, m)

function visibilitymap_analytic(m::AbstractModel, p::AbstractGrid)
    g = imagegrid(p)
    return  visibility_point.(Ref(m), g)
end

function visibilitymap_analytic!(vis, m::AbstractModel)
    d = axisdims(vis)
    g = imagegrid(d)
    vis .= visibility_point.(Ref(m), g)
    return nothing
end

function visibilitymap_analytic(m::AbstractModel, p::AbstractGrid{D, <:ThreadsEx}) where {D}
    vis = allocate_map(p)
    visibilitymap_analytic!(vis, m)
    return vis
end

function visibilitymap_analytic!(vis::UnstructuredMap{T, <:AbstractVector, <:UnstructuredGrid{D, <:ThreadsEx{S}}}, m::AbstractModel) where {T,D,S}
    d = axisdims(vis)
    g = imagegrid(d)
    _threads_visibilitymap!(vis, m, g, Val(S))
    return nothing
end

function visibilitymap_analytic!(
    vis::IntensityMap{T,N,D,<:AbstractArray{T,N},<:ComradeBase.AbstractRectiGrid{D, <:ThreadsEx{S}}},
    m::AbstractModel) where {T,N,D,S}
    d = axisdims(vis)
    g = imagegrid(d)
    _threads_visibilitymap!(vis, m, g, Val(S))
    return nothing
end





"""
    visibility(mimg, p)

Computes the complex visibility of model `m` at coordinates `p`. `p` corresponds to
the coordinates of the model. These need to have the properties `U`, `V` and sometimes
`Ti` for time and `Fr` for frequency.

# Notes
If you want to compute the visibilities at a large number of positions
consider using the [`visibilities`](@ref visibilities).
"""
@inline function visibility(mimg::M, p) where {M}
    #first we split based on whether the model is primitive
    T = typeof(p.U)
    _visibility(isprimitive(M), mimg, p.U, p.V, zero(T), zero(T))
end


"""
    amplitude(model, p)

Computes the visibility amplitude of model `m` at the coordinate `p`.
The coordinate `p`
is expected to have the properties `U`, `V`, and sometimes `Ti` and `Fr`.

If you want to compute the amplitudemap at a large number of positions
consider using the `amplitudemap` function.
"""
@inline function amplitude(model, p)
    return abs(visibility(model, p))
end

"""
    bispectrum(model, p1, p2, p3)

Computes the complex bispectrum of model `m` at the uv-triangle
p1 -> p2 -> p3

If you want to compute the bispectrum over a number of triangles
consider using the `bispectrummap` function.
"""
@inline function bispectrum(model, p1, p2, p3)
    return visibility(model, p1)*visibility(model, p2)*visibility(model, p3)
end

"""
    closure_phase(model, p1, p2, p3, p4)

Computes the closure phase of model `m` at the uv-triangle
u1,v1 -> u2,v2 -> u3,v3

If you want to compute closure phases over a number of triangles
consider using the `closure_phasemap` function.
"""
@inline function closure_phase(model, p1, p2, p3)
    return angle(bispectrum(model, p1, p2, p3))
end

"""
    logclosure_amplitude(model, p1, p2, p3, p4)

Computes the log-closure amplitude of model `m` at the uv-quadrangle
u1,v1 -> u2,v2 -> u3,v3 -> u4,v4 using the formula

```math
C = \\log\\left|\\frac{V(u1,v1)V(u2,v2)}{V(u3,v3)V(u4,v4)}\\right|
```

If you want to compute log closure amplitudemap over a number of triangles
consider using the `logclosure_amplitudemap` function.
"""
@inline function logclosure_amplitude(model, p1, p2, p3, p4)
    a1 = amplitude(model, p1)
    a2 = amplitude(model, p2)
    a3 = amplitude(model, p3)
    a4 = amplitude(model, p4)

    return log(a1*a2/(a3*a4))
end


#=
    Welcome to the trait jungle. Below is
    how we specify how to evaluate the model
=#
@inline function _visibility(::NotPrimitive, m, u, v, time, freq)
    return visibility_point(m, u, v, time, freq)
end

@inline function _visibility(::IsPrimitive, m::M, u, v, time, freq) where {M}
    _visibility_primitive(visanalytic(M), m, u, v, time, freq)
end


@inline function _visibility_primitive(::IsAnalytic, mimg, u, v, time, freq)
    return visibility_point(mimg, u, v, time, freq)
end




"""
    amplitudemap(m::AbstractModel, u::AbstractArray, v::AbstractArray)

Computes the visibility amplitudemap of the model `m` at the coordinates `p`.
The coordinates `p` are expected to have the properties `U`, `V`,
and sometimes `Ti` and `Fr`.
"""
function amplitudemap(m, p::NamedTuple{(:U, :V, :T, :F)})
    _amplitudemap(m, p.U, p.V, p.T, p.F)
end

function amplitudemap(m, p::NamedTuple{(:U, :V)})
    T = eltype(p.U)
    _amplitudemap(m, p.U, p.V, zero(T), zero(T))
end



function _amplitudemap(m::S, u, v, time, freq) where {S}
    _amplitudemap(visanalytic(S), m, u, v, time, freq)
end

function _amplitudemap(::IsAnalytic, m, u, v, time, freq)
    abs.(visibility_point.(Ref(m), u, v, time, freq))
end

function _amplitudemap(::NotAnalytic, m, u, v, time, freq)
    abs.(visibilitymap_numeric(m, u, v, time, freq))
end


"""
    bispectrummap(m, p1, p2, p3)

Computes the closure phases of the model `m` at the
triangles p1, p2, p3, where `pi` are coordinates.
"""
function bispectrummap(m,
                    p1,
                    p2,
                    p3,
                    )

    _bispectrummap(m, p1, p2, p3)
end

# internal method used for trait dispatch
function _bispectrummap(m::M,
                    p1,
                    p2,
                    p3
                    ) where {M}
    _bispectrummap(visanalytic(M), m, p1, p2, p3)
end

# internal method used for trait dispatch for analytic visibilities
function _bispectrummap(::IsAnalytic, m,
                    p1,
                    p2,
                    p3,
                   )
    return bispectrum.(Ref(m), StructArray(p1), StructArray(p2), StructArray(p3))
end

# internal method used for trait dispatch for non-analytic visibilities
function _bispectrummap(::NotAnalytic, m,
                    p1,p2,p3
                   )
    vis1 = visibilitymap(m, p1)
    vis2 = visibilitymap(m, p2)
    vis3 = visibilitymap(m, p3)
    return @. vis1*vis2*vis3
end

"""
    closure_phasemap(m,
                   p1::AbstractArray
                   p2::AbstractArray
                   p3::AbstractArray
                   )

Computes the closure phases of the model `m` at the
triangles p1, p2, p3, where `pi` are coordinates.
"""
@inline function closure_phasemap(m::AbstractModel,
                        p1,p2,p3
                        )
    _closure_phasemap(m, p1, p2, p3)
end

# internal method used for trait dispatch
@inline function _closure_phasemap(m::M, p1, p2, p3) where {M<:AbstractModel}
    _closure_phasemap(visanalytic(M), m, p1, p2, p3)
end

# internal method used for trait dispatch for analytic visibilities
@inline function _closure_phasemap(::IsAnalytic, m,
                        p1::NamedTuple,
                        p2::NamedTuple,
                        p3::NamedTuple
                       )
    return closure_phase.(Ref(m), StructArray(p1), StructArray(p2), StructArray(p3))
end

# internal method used for trait dispatch for non-analytic visibilities
function _closure_phasemap(::NotAnalytic, m, p1,p2, p3)
    return angle.(bispectrummap(m, p1, p2, p3))
end

"""
    logclosure_amplitudemap(m::AbstractModel,
                          p1,
                          p2,
                          p3,
                          p4
                         )

Computes the log closure amplitudemap of the model `m` at the
quadrangles p1, p2, p3, p4.
"""
function logclosure_amplitudemap(m::AbstractModel,
                               p1,
                               p2,
                               p3,
                               p4
                              )
    _logclosure_amplitudemap(m, p1, p2, p3, p4)
end


# internal method used for trait dispatch
@inline function _logclosure_amplitudemap(m::M,
                        p1,
                        p2,
                        p3,
                        p4
                       ) where {M<:AbstractModel}
    _logclosure_amplitudemap(visanalytic(M), m, p1, p2, p3, p4)
end

# internal method used for trait dispatch for analytic visibilities
@inline function _logclosure_amplitudemap(::IsAnalytic, m, p1::NamedTuple, p2::NamedTuple, p3::NamedTuple, p4::NamedTuple)
    return logclosure_amplitude.(Ref(m), StructArray(p1), StructArray(p2), StructArray(p3), StructArray(p4))
end

# internal method used for trait dispatch for non-analytic visibilities
@inline function _logclosure_amplitudemap(::NotAnalytic, m, p1, p2, p3, p4)
    amp1 = amplitudemap(m, p1)
    amp2 = amplitudemap(m, p2)
    amp3 = amplitudemap(m, p3)
    amp4 = amplitudemap(m, p4)
    return @. log(amp1*amp2*inv(amp3*amp4))
end
