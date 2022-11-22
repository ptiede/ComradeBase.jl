module ComradeBase

using DocStringExtensions
using ChainRulesCore
using FLoops
using StaticArrays

export  visibility, intensitymap, intensitymap!,
        StokesVector, CoherencyMatrix, evpa, m̆, SingleStokes,
        flux, fov, imagepixels, pixelsizes, IntensityMap

include("interface.jl")
include("images/images.jl")
include("polarizedtypes.jl")


end
