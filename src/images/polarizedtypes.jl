export StokesVector, CoherencyMatrix, evpa, m̆, SingleStokes

struct StokesVector{T} <: FieldVector{4,T}
    I::T
    Q::T
    U::T
    V::T
end

struct CoherencyMatrix{T} <: FieldMatrix{2,2,T}
    rr::T
    lr::T
    rl::T
    ll::T
end


@inline function Base.convert(::Type{CoherencyMatrix}, p::StokesVector)
    rr = p.I + p.V
    ll = p.I - p.V
    rl = p.Q + 1im*p.U
    lr = p.Q - 1im*p.U
    return CoherencyMatrix(rr, lr, rl, ll)
end

@inline function Base.convert(::Type{StokesVector}, p::CoherencyMatrix)
    i = (p.rr + p.ll)/2
    v = (p.rr - p.ll)/2
    q = (p.rl + p.lr)/2
    u = (p.rl - p.lr)/(2im)
    return StokesVector(i, q, u, v)
end


m̆(m::StokesVector) = (m.Q + 1im*m.U)/m.I
m̆(m::CoherencyMatrix) = 2*m.rl/(m.rr+m.ll)


evpa(m::StokesVector) = 1/2*atan(m.U,m.Q)
evpa(m::StokesVector{<:Complex}) = 1/2*angle(m.U/m.Q)
evpa(m::CoherencyMatrix) = evpa(convert(StokesVector, m))


"""
    $(TYPEDEF)
Helper function that converts a model from something that compute polarized images
to just a single stokes parameter. This is useful if you just want to fit a single
stokes parameter.
"""
struct SingleStokes{M, S} <: ComradeBase.AbstractModel
    model::M
end
SingleStokes(m::M, param::Symbol) where {M} = SingleStokes{M, param}(m)

visanalytic(::Type{SingleStokes{M,S}}) where {M,S} = visanalytic(M)
imanalytic(::Type{SingleStokes{M,S}})  where {M,S} = imanalytic(M)
@inline intensity_point(s::SingleStokes{M,S}, x,y) where {M,S} = getproperty(intensity_point(s.model, x,y), S)





# struct PolarizedMap{SI<:AbstractIntensityMap,
#                     SQ<:AbstractIntensityMap,
#                     SU<:AbstractIntensityMap,
#                     SV<:AbstractIntensityMap} <: AbstractPolarizedMap{SI,SQ,SU,SV}
#     I::SI
#     Q::SQ
#     U::SU
#     V::SV
#     function PolarizedMap(I::SI,Q::SQ,U::SU,V::SV) where {SI, SQ, SU, SV}
#         @assert size(I) == size(Q) == size(U) == size(V) "Image sizes must be equal in polarized map"
#         @assert fov(I) == fov(Q) == fov(U) == fov(V) "Image fov must be equal in polarized map"
#         new{SI,SQ,SU,SV}(I,Q,U,V)
#     end
# end


# Base.Base.@propagate_inbounds function Base.getindex(pimg::PolarizedMap, i...)
# return StokesVector(pimg.I[i...], pimg.Q[i...], pimg.U[i...], pimg.V[i...])
# end

# @inline stokes_parameter(pimg::PolarizedMap, p::Symbol) = getproperty(pimg, p)
