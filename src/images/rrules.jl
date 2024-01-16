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
