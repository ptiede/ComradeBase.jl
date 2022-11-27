module ComradeBase

using AxisKeys
using ChainRulesCore
using DocStringExtensions
using RectiGrids
using StaticArrays
using StructArrays

export  visibility, intensitymap, intensitymap!,
        StokesParams, CoherencyMatrix, evpa, m̆, SingleStokes,
        flux, fov, imagepixels, pixelsizes, IntensityMap,
        named_axiskeys

include("interface.jl")
include("polarizedtypes.jl")
include("images/images.jl")


end
