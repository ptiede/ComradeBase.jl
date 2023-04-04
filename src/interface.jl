
"""
    AbstractModel

The Comrade abstract model type. To instantiate your own model type you should
subtybe from this model. Additionally you need to implement the following
methods to satify the interface:

**Mandatory Methods**
- [`isprimitive`](@ref): defines whether a model is standalone or is defined in terms of other models.
   is the model is primitive then this should return `IsPrimitive()` otherwise it returns
   `NotPrimitive()`
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
- [`visibilities_analytic`](@ref): Vectorized version of `visibility_point` for models where `visanalytic` returns `IsAnalytic()`
- [`visibilities_numeric`](@ref): Vectorized version of `visibility_point` for models where `visanalytic` returns `NotAnalytic()` typically these are numerical FT's
- [`intensitymap_analytic`](@ref): Computes the entire image for models where `imanalytic` returns `IsAnalytic()`
- [`intensitymap_numeric`](@ref): Computes the entire image for models where `imanalytic` returns `NotAnalytic()`
- [`intensitymap_analytic!`](@ref): Inplace version of `intensitymap`
- [`intensitymap_numeric!`](@ref): Inplace version of `intensitymap`


"""
abstract type AbstractModel end

"""
        $(TYPEDEF)

Type the classifies a model as being intrinsically polarized. This means that any call
to visibility must return a `StokesParams` to denote the full stokes polarization of the model.
"""
abstract type AbstractPolarizedModel <: AbstractModel end


export stokes, IsPolarized, NotPolarized
"""
    stokes(m::AbstractPolarizedModel, p::Symbol)

Extract the specific stokes component `p` from the polarized model `m`
"""
stokes(m::AbstractPolarizedModel, v::Symbol) = getproperty(m, v)

struct IsPolarized end
struct NotPolarized end


"""
    ispolarized(::Type)

Trait function that defines whether a model is polarized or not.
"""
ispolarized(::Type{<:AbstractModel}) = NotPolarized()
ispolarized(::Type{<:AbstractPolarizedModel}) = IsPolarized()



"""
    $(TYPEDEF)

This trait specifies whether the model is a *primitive*

# Notes
This will likely turn into a trait in the future so people
can inject their models into Comrade more easily.
"""
abstract type PrimitiveTrait end

"""
    $(TYPEDEF)
Trait for primitive model
"""
struct IsPrimitive end
"""
    $(TYPEDEF)
Trait for not-primitive model
"""
struct NotPrimitive end

"""
    isprimitive(::Type)

Dispatch function that specifies whether a type is a primitive Comrade model.
This function is used for dispatch purposes when composing models.

# Notes
If a user is specifying their own model primitive model outside of Comrade they need
to specify if it is primitive

```julia
struct MyPrimitiveModel end
ComradeBase.isprimitive(::Type{MyModel}) = ComradeBase.IsPrimitive()
```

"""
function isprimitive end






@inline isprimitive(::Type{<:AbstractModel}) = NotPrimitive()

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
    intensitymap(model::AbstractModel, p::AbstractDims)

Computes the intensity map of model. For the inplace version see [`intensitymap!`](@ref)
"""
function intensitymap end

"""
    intensitymap_numeric(m::AbstractModel, p::AbstractDims)

Computes the `IntensityMap` of a model `m` at the image positions `p` using a numerical method.
This has to be specified uniquely for every model `m` if `imanalytic(typeof(m)) === NotAnalytic()`.
See `Comrade.jl` for example implementations.
"""
function intensitymap_numeric end

"""
    intensitymap_analytic(m::AbstractModel, p::AbstractDims)

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
    visibilities(model::AbstractModel, args...)

Computes the complex visibilities at the locations given by `args...`
"""
function visibilities end


"""
    visibilities!(vis::AbstractArray, model::AbstractModel, args...)

Computes the complex visibilities `vis` in place at the locations given by `args...`
"""
function visibilities! end


"""
    _visibilities(model::AbstractModel, args...)

Internal method used for trait dispatch and unpacking of args arguments in `visibilities`

!!! warn
    Not part of the public API so it may change at any moment.
"""
function _visibilities end

"""
    _visibilities!(model::AbstractModel, args...)

Internal method used for trait dispatch and unpacking of args arguments in `visibilities!`

!!! warn
    Not part of the public API so it may change at any moment.
"""
function _visibilities! end

"""
    visibilties_numeric(model, u, v, time, freq)

Computes the visibilties of a `model` using a numerical fourier transform. Note that
none of these are implemented in `ComradeBase`. For implementations please see `Comrade`.
"""
function visibilities_numeric end

"""
    visibilties_analytic(model, u, v, time, freq)

Computes the visibilties of a `model` using using the analytic visibility expression given by
`visibility_point`.
"""
function visibilities_analytic end

"""
    visibilties_numeric!(vis, model, u, v, time, freq)

Computes the visibilties of a `model` in-place using a numerical fourier transform. Note that
none of these are implemented in `ComradeBase`. For implementations please see `Comrade`.
"""
function visibilities_numeric! end

"""
    visibilties_analytic!(vis, model, u, v, time, freq)

Computes the visibilties of a `model` in-place, using using the analytic visibility expression given by
`visibility_point`.
"""
function visibilities_analytic! end


#     ($SIGNATURES)
# A threaded version of the intensitymap function.

# # Notes
# If using autodiff this won't play well with zygote given the mutation. This will be fixed in
# future versions.
# """
# function tintensitymap end

# """
#     ($SIGNATURES)
# A threaded version of the intensitymap! function.

# # Notes
# If using autodiff this won't play well with zygote given the mutation. This will be fixed in
# future versions.
# """
# function tintensitymap! end
