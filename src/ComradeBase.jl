module ComradeBase

using DocStringExtensions
using ChainRulesCore
using StaticArrays
using StructArrays

export  visibility, intensitymap, intensitymap!,
        StokesVector, CoherencyMatrix, evpa, m̆, SingleStokes,
        flux, fov, imagepixels, pixelsizes, IntensityMap

include("interface.jl")
include("images/images.jl")
#include("polarized.jl")


end