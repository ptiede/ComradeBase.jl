module ComradeBase

using ChainRulesCore
using DimensionalData
const DD = DimensionalData
using DocStringExtensions
using FITSIO
using StaticArrays
using StructArrays
using Statistics
using Reexport
@reexport using PolarizedTypes
using PrecompileTools

export  visibility,
        intensitymap, intensitymap!,
        visibilitymap, visibilitymap!,
        StokesParams, CoherencyMatrix,
        flux, fieldofview, imagepixels, pixelsizes, IntensityMap,
        named_dims, IntensityMapTypes

include("interface.jl")
include("domain.jl")
include("unstructured_map.jl")
include("images/images.jl")

const FluxMap2{T, N, E} = Union{IntensityMap{T,N,<:Any,E}, UnstructuredMap{T,<:AbstractVector,E}}


include("visibilities.jl")
include("rrules.jl")

@setup_workload begin
    fovx = 10.0
    fovy = 12.0
    nx = 10
    ny = 10
    # @compile_workload begin
    #     p = imagepixels(fovx, fovy, nx, ny)
    #     g = RectiGrid(p)
    #     gs = domainpoints(p)
    #     imgI = IntensityMap(rand(10, 10), g)
    #     imgI.^2

    #     pimg = StokesIntensityMap(imgI, imgI, imgI, imgI)
    # end
end

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5" include(joinpath(@__DIR__, "..", "ext", "ComradeBaseOhMyThreadsExt.jl"))
    end
end



end
