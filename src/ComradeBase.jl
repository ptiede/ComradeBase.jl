module ComradeBase

using AxisKeys
using ChainRulesCore
using DocStringExtensions
using RectiGrids
using StaticArrays
using StructArrays

export  visibility, intensitymap, intensitymap!,
        StokesVector, CoherencyMatrix, evpa, m̆, SingleStokes,
        flux, fov, imagepixels, pixelsizes, IntensityMap

include("interface.jl")
include("images/images.jl")
#include("threaded.jl")

#include("polarized.jl")


end
