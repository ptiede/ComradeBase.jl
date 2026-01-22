export IntensityMap, SpatialIntensityMap,
    DataArr, SpatialDataArr, DataNames,
    named_axisdims, imagepixels, pixelsizes, domainpoints,
    phasecenter, baseimage, stokes

include("dim_image.jl")

export flux, centroid, second_moment, named_axisdims, axisdims,
    imagepixels, pixelsizes, domainpoints, phasecenter
include("methods.jl")
