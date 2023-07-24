module ComradeBase

using AxisKeys
using ChainRulesCore
using DocStringExtensions
using FITSIO
using StaticArrays
using StructArrays
using RectiGrids
using Statistics
using Reexport
@reexport using PolarizedTypes
using PrecompileTools

export  visibility, intensitymap, intensitymap!,
        StokesParams, CoherencyMatrix, evpa, m̆, SingleStokes,
        flux, fieldofview, imagepixels, pixelsizes, IntensityMap,
        named_dims, IntensityMapTypes

include("interface.jl")
include("images/images.jl")
include("visibilities.jl")

@setup_workload begin
    fovx = 10.0
    fovy = 12.0
    nx = 10
    ny = 10
    @compile_workload begin
        p = imagepixels(fovx, fovy, nx, ny)
        g = GriddedKeys(p)
        imgI = IntensityMap(rand(10, 10), g)
        imgI.^2

        pimg = StokesIntensityMap(imgI, imgI, imgI, imgI)
    end
end


end
