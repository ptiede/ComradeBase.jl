module ComradeBase

# using ChainRulesCore
using EnzymeCore: EnzymeRules
using DimensionalData
const DD = DimensionalData
using DocStringExtensions
using StaticArrays
using StructArrays
using Reexport
using Accessors: @set
@reexport using PolarizedTypes
@reexport using FieldDimArrays
using PrecompileTools

export visibility,
    intensitymap, intensitymap!,
    visibilitymap, visibilitymap!,
    StokesParams, CoherencyMatrix, CirBasis, LinBasis,
    flux, fieldofview, imagepixels, pixelsizes, IntensityMap,
    named_dims


include("interface.jl")
include("domains/domain.jl")
include("models/models.jl")
include("images/images.jl")

@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public,
            :AbstractModel, :AbstractPolarizedModel,
            :intensity_point, :visibility_point,
            :visanalytic, :imanalytic, :IsAnalytic, :NotAnalytic,
            :ispolarized, :radialextent,
            :intensitymap_analytic, :intensitymap_analytic!,
            :intensitymap_numeric, :intensitymap_numeric!,
            :visibilitymap_analytic, :visibilitymap_analytic!,
            :visibilitymap_numeric, :visibilitymap_numeric!,
            :intensitymap_analytic_executor!,
            :AbstractDomain, :AbstractSingleDomain, :AbstractDualDomain, :AbstractRectiGrid,
            :StructuredMap,
            :create_map, :create_imgmap, :create_vismap,
            :allocate_map, :allocate_imgmap, :allocate_vismap,
            :basedim, :NoHeader, :MinimalHeader,
            :DomainParams, :paramtype,
            :rgetindex, :rsetindex!, :pointbroadcasted,
        )
    )
end

"""
    rgetindex(A, i...)

Returns `A[i...]`. Packages extend it for arrays that need a different indexing path at
traced indices, so model code indexes through it rather than calling `getindex` directly.
"""
Base.@propagate_inbounds function rgetindex(I, i...)
    return I[i...]
end

"""
    rsetindex!(A, v, i...)

Sets `A[i...] = v`; the in-place counterpart of [`rgetindex`](@ref).
"""
Base.@propagate_inbounds function rsetindex!(I, v, i...)
    return I[i...] = v
end


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
