function ChainRulesCore.ProjectTo(img::IntensityMap)
    return ProjectTo{IntensityMap}(;data=ProjectTo(DD.data(img)),
                                    grid=axisdims(img),
                                    refdims=DD.refdims(img),
                                    name = DD.name(img))
end

(project::ProjectTo{IntensityMap})(dx) = IntensityMap(dx, project.grid, project.refdims, project.name)
(project::ProjectTo{IntensityMap})(dx::AbstractZero) = dx

__getdata(img::IntensityMap) = DimensionalData.data(img)
__getdata(img::Tangent) = img.data
__getdata(img::ChainRulesCore.Thunk) = img
__getdata(img::UnstructuredMap) = baseimage(img)

function ChainRulesCore.rrule(::Type{IntensityMap}, data::AbstractArray, keys...)
    img = IntensityMap(data, keys...)
    pd = ProjectTo(data)
    function _IntensityMap_pullback(Δ)
        return (NoTangent(), @thunk(pd(__getdata(Δ))), map(i->NoTangent(), keys)...)
    end
    return img, _IntensityMap_pullback
end



_baseim_pb(Δ, pr) = (NoTangent(), pr(Δ))
_baseim_pb(Δ::Tangent, pr) = _baseim_pb(Δ.data, pr)
_baseim_pb(Δ::AbstractThunk, pr) = _baseim_pb(unthunk(Δ), pr)

function ChainRulesCore.rrule(::typeof(baseimage), img)
    pb(Δ) = _baseim_pb(Δ, ProjectTo(img))
    return baseimage(img), pb
end

function ChainRulesCore.ProjectTo(img::UnstructuredMap)
    return ProjectTo{UnstructuredMap}(;data=ProjectTo(parent(img)),
                                       dims=axisdims(img),
                                       )
end
(project::ProjectTo{UnstructuredMap})(dx) = UnstructuredMap(dx, project.dims)
(project::ProjectTo{UnstructuredMap})(dx::AbstractZero) = dx


function ChainRulesCore.rrule(::Type{UnstructuredMap}, data::AbstractArray, dims)
    img = UnstructuredMap(data, dims)
    pd = ProjectTo(data)
    function _UnstructuredMap_pullback(Δ)
        return (NoTangent(), @thunk(pd(__getdata(Δ))), NoTangent())
    end
    return img, _UnstructuredMap_pullback
end
