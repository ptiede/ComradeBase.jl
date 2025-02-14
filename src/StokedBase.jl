module StokedBase

# using ChainRulesCore
using EnzymeCore: EnzymeRules
using DimensionalData
const DD = DimensionalData
using DocStringExtensions
using StaticArrays
using StructArrays
using Reexport
@reexport using PolarizedTypes
using PrecompileTools

export visibility,
       intensitymap, intensitymap!,
       visibilitymap, visibilitymap!,
       StokesParams, CoherencyMatrix,
       flux, fieldofview, imagepixels, pixelsizes, IntensityMap,
       named_dims

include("interface.jl")
include("domain.jl")
include("unstructured/domain.jl")
include("unstructured/map.jl")
include("images/images.jl")

const FluxMap2{T,N,E} = Union{IntensityMap{T,N,<:Any,E},
                              UnstructuredMap{T,<:AbstractVector,E}}

include("visibilities.jl")
# include("rrules.jl")

@setup_workload begin
    fovx = 10.0
    fovy = 12.0
    nx = 10
    ny = 10
    @compile_workload begin
        p = imagepixels(fovx, fovy, nx, ny)
        g = RectiGrid(p)
        gs = domainpoints(p)
        imgI = IntensityMap(rand(10, 10), g)
        imgI .^ 2
    end
end

end
