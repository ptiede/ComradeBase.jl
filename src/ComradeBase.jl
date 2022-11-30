module ComradeBase

using ChainRulesCore
using DimensionalData
using DocStringExtensions
using StaticArrays
using StructArrays

export  visibility, intensitymap, intensitymap!,
        StokesParams, CoherencyMatrix, evpa, m̆, SingleStokes,
        flux, fieldofview, imagepixels, pixelsizes, IntensityMap,
        named_dims

include("interface.jl")
include("polarizedtypes.jl")
include("images/images.jl")


end
