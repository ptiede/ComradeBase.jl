
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

**Optional Methods:**

- [`intensity_point`](@ref): Defines how to compute model intensities pointwise. Note this is
  must be defined if `imanalytic(::Type{YourModel})==IsAnalytic()`.
- [`visibility_point`](@ref): Defines how to compute model visibilties pointwise. Note this is
    must be defined if `visanalytic(::Type{YourModel})==IsAnalytic()`.
- [`_visibilities`](@ref): Vectorized version of `visibility_point` if you can gain additional
  speed
- [`intensitymap`](@ref): Computes the whole image of the model
- [`intensitymap!`](@ref): Inplace version of `intensitymap`


"""
abstract type AbstractModel end
abstract type AbstractPolarizedModel <: AbstractModel end




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
Base.@constpop :aggressive Base.:*(::IsAnalytic, ::IsAnalytic) = IsAnalytic()
Base.@constpop :aggressive Base.:*(::IsAnalytic, ::NotAnalytic) = NotAnalytic()
Base.@constpop :aggressive Base.:*(::NotAnalytic, ::IsAnalytic) = NotAnalytic()
Base.@constpop :aggressive Base.:*(::NotAnalytic, ::NotAnalytic) = NotAnalytic()


"""
    visibility_point(model::AbstractModel, u, v, args...)

Function that computes the pointwise visibility. This must be implemented
in the model interface if `visanalytic(::Type{MyModel}) == IsAnalytic()`
"""
function visibility_point end

"""
    intensity_point(model::AbstractModel, x, y, args...)

Function that computes the pointwise intensity if the model has the trait
in the image domain `IsAnalytic()`. Otherwise it will use construct the image in visibility
space and invert it.
"""
function intensity_point end


"""
    intensitymap!(buffer::AbstractMatrix, model::AbstractModel, args...)

Computes the intensity map of `model` by modifying the `buffer`
"""
function intensitymap! end


"""
    intensitymap(model::AbstractModel, args...)

Computes the intensity map of model. For the inplace version see [`intensitymap!`](@ref)
"""
function intensitymap end


"""
    radialextent(model::AbstractModel)

Provides an estimate of the radial size/extent of the `model`. This is used internally to
estimate image size when plotting and using `modelimage`
"""
function radialextent end

# """
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
