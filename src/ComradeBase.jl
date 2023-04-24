module ComradeBase

using AxisKeys
using ChainRulesCore
using DocStringExtensions
using StaticArrays
using StructArrays
using RectiGrids
using Statistics
using PrecompileTools

export  visibility, intensitymap, intensitymap!,
        StokesParams, CoherencyMatrix, evpa, m̆, SingleStokes,
        flux, fieldofview, imagepixels, pixelsizes, IntensityMap,
        named_dims, IntensityMapTypes

include("interface.jl")
include("polarizedtypes.jl")
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

        # Now polarization stuff
        s = StokesParams(1.0, 0.5, 0.5, 0.5)
        c1 = CoherencyMatrix(s, CirBasis(), CirBasis())
        c2 = CoherencyMatrix(s, LinBasis(), LinBasis())
        c3 = CoherencyMatrix(s, CirBasis(), LinBasis())
        c4 = CoherencyMatrix(s, LinBasis(), CirBasis())

        StokesParams(c1)
        StokesParams(c2)
        StokesParams(c3)
        StokesParams(c4)
    end
end


end
