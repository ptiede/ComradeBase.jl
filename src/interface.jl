
"""
    AbstractModel

The Comrade abstract model type. To instantiate your own model type you should
subtybe from this model. Additionally you need to implement the following
methods to satify the interface:

**Mandatory Methods**
- [`visanalytic`](@ref): defines whether the model visibilities can be computed analytically. If yes
   then this should return `IsAnalytic()` and the user *must* to define `visibility_point`.
   If not analytic then `visanalytic` should return `NotAnalytic()`.
- [`imanalytic`](@ref): defines whether the model intensities can be computed pointwise. If yes
    then this should return `IsAnalytic()` and the user *must* to define `intensity_point`.
    If not analytic then `imanalytic` should return `NotAnalytic()`.
- [`radialextent`](@ref): Provides a estimate of the radial extent of the model in the image domain.
   This is used for estimating the size of the image, and for plotting.
- [`flux`](@ref): Returns the total flux of the model.
- [`intensity_point`](@ref): Defines how to compute model intensities pointwise. Note this is
  must be defined if `imanalytic(::Type{YourModel})==IsAnalytic()`.
- [`visibility_point`](@ref): Defines how to compute model visibilties pointwise. Note this is
    must be defined if `visanalytic(::Type{YourModel})==IsAnalytic()`.
**Optional Methods:**
- [`ispolarized`](@ref): Specified whether a model is intrinsically polarized (returns `IsPolarized()`) or is not (returns `NotPolarized()`), by default a model is `NotPolarized()`
- [`visibilitymap_analytic`](@ref): Vectorized version of `visibility_point` for models where `visanalytic` returns `IsAnalytic()`
- [`visibilitymap_numeric`](@ref): Vectorized version of `visibility_point` for models where `visanalytic` returns `NotAnalytic()` typically these are numerical FT's
- [`intensitymap_analytic`](@ref): Computes the entire image for models where `imanalytic` returns `IsAnalytic()`
- [`intensitymap_numeric`](@ref): Computes the entire image for models where `imanalytic` returns `NotAnalytic()`
- [`intensitymap_analytic!`](@ref): Inplace version of `intensitymap`
- [`intensitymap_numeric!`](@ref): Inplace version of `intensitymap`


"""
abstract type AbstractModel end



export stokes, IsPolarized, NotPolarized

# Trait to signal whether a model is polarized or not
struct IsPolarized end
struct NotPolarized end

Base.@constprop  :aggressive Base.:*(::IsPolarized, ::IsPolarized) = IsPolarized()
Base.@constprop  :aggressive Base.:*(::IsPolarized, ::NotPolarized) = IsPolarized()
Base.@constprop  :aggressive Base.:*(::NotPolarized, ::IsPolarized) = IsPolarized()
Base.@constprop  :aggressive Base.:*(::NotPolarized, ::NotPolarized) = NotPolarized()

"""
    ispolarized(::Type)

Trait function that defines whether a model is polarized or not.
"""
ispolarized(::Type{<:AbstractModel}) = NotPolarized()


"""
        $(TYPEDEF)

A generic polarized model. To implement the use needs to follow the [`AbstractModel`](@ref)
implementation instructions. In addtion there is an optional method `stokes(model, p::Symbol)`
which extracts the specific stokes parameter of the model. The default that the different
stokes parameters are stored as fields of the model. To overwrite this behavior overload the
function.
"""
abstract type AbstractPolarizedModel <: AbstractModel end
ispolarized(::Type{<:AbstractPolarizedModel}) = IsPolarized()

"""
    stokes(m::AbstractPolarizedModel, p::Symbol)

Extract the specific stokes component `p` from the polarized model `m`
"""
stokes(m::AbstractPolarizedModel, v::Symbol) = getfield(m, v)


"""
    DensityAnalytic

Internal type for specifying the nature of the model functions.
Whether they can be easily evaluated pointwise analytic. This
is an internal type that may change.
"""
abstract type DensityAnalytic end

"""
    $(TYPEDEF)

Defines a trait that a states that a model is analytic.
This is usually used with an abstract model where we use
it to specify whether a model has a analytic fourier transform
and/or image.
"""
struct IsAnalytic <: DensityAnalytic end

"""
    $(TYPEDEF)

Defines a trait that a states that a model is analytic.
This is usually used with an abstract model where we use
it to specify whether a model has does not have a easy analytic
fourier transform and/or intensity function.
"""
struct NotAnalytic <: DensityAnalytic end

"""
    visanalytic(::Type{<:AbstractModel})

Determines whether the model is pointwise analytic in Fourier domain, i.e. we can evaluate
its fourier transform at an arbritrary point.

If `IsAnalytic()` then it will try to call `visibility_point` to calculate the complex visibilities.
Otherwise it fallback to using the FFT that works for all models that can compute an image.

"""
@inline visanalytic(::Type{<:AbstractModel}) = NotAnalytic()

"""
    imanalytic(::Type{<:AbstractModel})

Determines whether the model is pointwise analytic in the image domain, i.e. we can evaluate
its intensity at an arbritrary point.

If `IsAnalytic()` then it will try to call `intensity_point` to calculate the intensity.
"""
@inline imanalytic(::Type{<:AbstractModel}) = IsAnalytic()



#=
    This is internal function definitions for how to
    compose whether a model is analytic. We need this
    for composite models.
=#
Base.@constprop  :aggressive Base.:*(::IsAnalytic, ::IsAnalytic) = IsAnalytic()
Base.@constprop  :aggressive Base.:*(::IsAnalytic, ::NotAnalytic) = NotAnalytic()
Base.@constprop  :aggressive Base.:*(::NotAnalytic, ::IsAnalytic) = NotAnalytic()
Base.@constprop  :aggressive Base.:*(::NotAnalytic, ::NotAnalytic) = NotAnalytic()


# Traits are not differentiable
ChainRulesCore.@non_differentiable visanalytic(::Type)
ChainRulesCore.@non_differentiable imanalytic(::Type)
ChainRulesCore.@non_differentiable ispolarized(::Type)

"""
    visibility_point(model::AbstractModel, p)


Function that computes the pointwise visibility. This must be implemented
in the model interface if `visanalytic(::Type{MyModel}) == IsAnalytic()`
"""
function visibility_point end

"""
    intensity_point(model::AbstractModel, p)

Function that computes the pointwise intensity if the model has the trait
in the image domain `IsAnalytic()`. Otherwise it will use construct the image in visibility
space and invert it.
"""
function intensity_point end


"""
    intensitymap!(buffer::AbstractDimArray, model::AbstractModel)

Computes the intensity map of `model` by modifying the `buffer`
"""
function intensitymap! end


"""
    intensitymap(model::AbstractModel, p::AbstractGrid)

Computes the intensity map of model. For the inplace version see [`intensitymap!`](@ref)
"""
function intensitymap end

"""
    intensitymap_numeric(m::AbstractModel, p::AbstractGrid)

Computes the `IntensityMap` of a model `m` at the image positions `p` using a numerical method.
This has to be specified uniquely for every model `m` if `imanalytic(typeof(m)) === NotAnalytic()`.
See `Comrade.jl` for example implementations.
"""
function intensitymap_numeric end

"""
    intensitymap_analytic(m::AbstractModel, p::AbstractGrid)

Computes the `IntensityMap` of a model `m` using the image dimensions `p`
by broadcasting over the analytic [`intensity_point`](@ref) method.
"""
function intensitymap_analytic end

"""
    intensitymap_numeric!(img::IntensityMap, m::AbstractModel)
    intensitymap_numeric!(img::StokesIntensityMap, m::AbstractModel)

Updates the `img` using the model `m`  using a numerical method.
This has to be specified uniquely for every model `m` if `imanalytic(typeof(m)) === NotAnalytic()`.
See `Comrade.jl` for example implementations.
"""
function intensitymap_numeric! end

"""
    intensitymap_analytic!(img::IntensityMap, m::AbstractModel)
    intensitymap_analytic!(img::StokesIntensityMap, m::AbstractModel)

Updates the `img` using the model `m`  by broadcasting over the analytic [`intensity_point`](@ref) method.
"""
function intensitymap_analytic! end







"""
    radialextent(model::AbstractModel)

Provides an estimate of the radial size/extent of the `model`. This is used internally to
estimate image size when plotting and using `modelimage`
"""
function radialextent end

"""
    visibilitymap(model::AbstractModel, p)

Computes the complex visibilities at the locations p.
"""
function visibilitymap end


"""
    visibilitymap!(vis::AbstractArray, model::AbstractModel, p)

Computes the complex visibilities `vis` in place at the locations p
"""
function visibilitymap! end


"""
    _visibilitymap(model::AbstractModel, p)

Internal method used for trait dispatch and unpacking of args arguments in `visibilities`

!!! warn
    Not part of the public API so it may change at any moment.
"""
function _visibilitymap end

"""
    _visibilitymap!(model::AbstractModel, p)

Internal method used for trait dispatch and unpacking of args arguments in `visibilities!`

!!! warn
    Not part of the public API so it may change at any moment.
"""
function _visibilitymap! end

"""
    visibilties_numeric(model, p)

Computes the visibilties of a `model` using a numerical fourier transform. Note that
none of these are implemented in `ComradeBase`. For implementations please see `Comrade`.
"""
function visibilitymap_numeric end

"""
    visibilties_analytic(model, p)

Computes the visibilties of a `model` using using the analytic visibility expression given by
`visibility_point`.
"""
function visibilitymap_analytic end

"""
    visibilties_numeric!(vis, model)

Computes the visibilties of a `model` in-place using a numerical fourier transform. Note that
none of these are implemented in `ComradeBase`. For implementations please see `Comrade`.
"""
function visibilitymap_numeric! end

"""
    visibilties_analytic!(vis, model)

Computes the visibilties of a `model` in-place, using using the analytic visibility expression given by
`visibility_point`.
"""
function visibilitymap_analytic! end
