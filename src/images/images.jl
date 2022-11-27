


export IntensityMap, ImageDimensions, named_dims, dims, grid,
       named_axiskeys, imagepixels, pixelsizes, fov
include("keyed_image.jl")

export StokesIntensityMap
include("stokes_image.jl")

export flux, centroid, second_moment
include("moments.jl")

export ContinuousImage
include("continuous_image.jl")
#include("polarizedmap.jl")
