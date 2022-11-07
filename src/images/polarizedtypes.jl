export StokesParameters, CoherencyMatrix, evpa, m̆, SingleStokes

"""
    $(TYPEDEF)
Static vector that holds the stokes parameters of a polarized
complex visibility
"""
struct StokesParameters{T} <: FieldVector{4,T}
    I::T
    Q::T
    U::T
    V::T
end



const StokesIntensityMap{T,N,Na} = StructArray{<:StokesParameters, N, <:NamedTuple{(:I,:Q,:U,:V), <:NTuple{4, <:IntensityMap{T, N, Na}}}} where {N}


"""
    stackstokes(I, Q, U, V)

Create an array of full stokes parameters. The image is stored as a `StructArray` of
@ref[StokesParameters]. Each of the four
"""
function stackstokes(I::IntensityMap{T}, Q::IntensityMap{T}, U::IntensityMap{T}, V::IntensityMap{T}) where {T}
    @assert check_grid(I,Q,U,V) "Intensity grids are not the same across the 4 stokes parameters"
    return StructArray{StokesParameters{T}}(;I,Q,U,V)
end

# simple check to ensure that the four grids are equal across stokes parameters
function check_grid(I,Q,U,V)
    named_axiskeys(I) == named_axiskeys(Q) == named_axiskeys(U) == named_axiskeys(V)
end

function AxisKeys.named_axiskeys(simg::StokesIntensityMap)
    return named_axiskeys(simg.I)
end

function Base.summary(io::IO, x::StokesIntensityMap)
    return _summary(io, x)
end

function _summary(io, x::StokesIntensityMap{T,N,Na}) where {T,N,Na}
    println(io, ndims(x.I), "-dimensional")
    println(io, "StokesIntensityMap{$T, $N, $Na}")
    for d in 1:ndims(x)
        print(io, d==1 ? "↓" : d==2 ? "→" : d==3 ? "◪" : "▨", "   ")
        c = AxisKeys.colour(x, d)
        AxisKeys.hasnames(x.I) && printstyled(io, AxisKeys.dimnames(x.I,d), " ∈ ", color=c)
        printstyled(io, length(axiskeys(x.I,d)), "-element ", AxisKeys.shorttype(axiskeys(x.I,d)), "\n", color=c)
    end
    println(io, "Polarizations ", propertynames(x))
end

Base.show(io::IO, img::StokesIntensityMap) = summary(io, img)


linearpol(s::StokesParameters) = s.Q + im*s.U



"""
    $(TYPEDEF)
Static matrix that holds construct the coherency matrix of a polarized
complex visibility in a basis given by `B`. There are a two main bases we
use `:RL` for a circular basis, and `:XY` for linear basis.


```julia
convert(::StokesParameters, CoherencyMatrix(1.0, 0.1, 0.1, 0.4))
```
"""
struct CoherencyMatrix{B,T} <: FieldMatrix{2,2,T}
    c11::T
    c21::T
    c12::T
    c22::T
end




@inline function Base.convert(::Type{CoherencyMatrix{:RL}}, p::StokesParameters)
    rr = p.I + p.V
    ll = p.I - p.V
    rl = p.Q + 1im*p.U
    lr = p.Q - 1im*p.U
    return CoherencyMatrix(rr, lr, rl, ll)
end

@inline function Base.convert(::Type{StokesParameters}, p::CoherencyMatrix{:RL})
    i = (p.c11 + p.c22)/2
    v = (p.c11 - p.c22)/2
    q = (p.c21 + p.c12)/2
    u = (p.c21 - p.c12)/(2im)
    return StokesParameters(i, q, u, v)
end

"""
    $(SIGNATURES)
Compute the fractional linear polarization of a stokes vector
or coherency matrix
"""
m̆(m::StokesParameters) = (m.Q + 1im*m.U)/(m.I + eps())

"""
    $(SIGNATURES)
Compute the evpa of a stokes vector or cohereny matrix.
"""
evpa(m::StokesParameters) = 1/2*atan(m.U,m.Q)
evpa(m::StokesParameters{<:Complex}) = 1/2*angle(m.U/m.Q)


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
isprimitive(::Type{SingleStokes{M,S}}) where {M,S} = isprimitive(M)
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
# return StokesParameters(pimg.I[i...], pimg.Q[i...], pimg.U[i...], pimg.V[i...])
# end

# @inline stokes_parameter(pimg::PolarizedMap, p::Symbol) = getproperty(pimg, p)
