var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ComradeBase","category":"page"},{"location":"#ComradeBase","page":"Home","title":"ComradeBase","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ComradeBase.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ComradeBase]","category":"page"},{"location":"#ComradeBase.AbstractModel","page":"Home","title":"ComradeBase.AbstractModel","text":"AbstractModel\n\nThe Comrade abstract model type. To instantiate your own model type you should subtybe from this model. Additionally you need to implement the following methods to satify the interface:\n\nMandatory Methods\n\nisprimitive: defines whether a model is standalone or is defined in terms of other models.  is the model is primitive then this should return IsPrimitive() otherwise it returns  NotPrimitive()\nvisanalytic: defines whether the model visibilities can be computed analytically. If yes  then this should return IsAnalytic() and the user must to define visibility_point.  If not analytic then visanalytic should return NotAnalytic().\nimanalytic: defines whether the model intensities can be computed pointwise. If yes\n\nthen this should return IsAnalytic() and the user must to define intensity_point. If not analytic then imanalytic should return NotAnalytic().\n\nradialextent: Provides a estimate of the radial extent of the model in the image domain.  This is used for estimating the size of the image, and for plotting.\nflux: Returns the total flux of the model.\n\nOptional Methods:\n\nintensity_point: Defines how to compute model intensities pointwise. Note this is must be defined if imanalytic(::Type{YourModel})==IsAnalytic().\nvisibility_point: Defines how to compute model visibilties pointwise. Note this is   must be defined if visanalytic(::Type{YourModel})==IsAnalytic().\n_visibilities: Vectorized version of visibility_point if you can gain additional speed\nintensitymap: Computes the whole image of the model\nintensitymap!: Inplace version of intensitymap\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.BSplinePulse","page":"Home","title":"ComradeBase.BSplinePulse","text":"$(TYPEDEF)\n\nUses the basis spline (BSpline) kernel of order N. These are the kernel that come from recursively convolving the tophat kernel\n\n    B_0(x) = begincases 1  x  1  0  otherwise endcases\n\nN times.\n\nNotes\n\nBSpline kernels have a number of nice properties:\n\nSimple frequency response sinc(u2)^N\npreserve total intensity\n\nFor N>1 these kernels aren't actually interpolation kernels however, this doesn't matter for us.\n\nCurrently only the 0,1,3 order kernels are implemented.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.CoherencyMatrix","page":"Home","title":"ComradeBase.CoherencyMatrix","text":"struct CoherencyMatrix{T} <: StaticArrays.FieldMatrix{2, 2, T}\n\nStatic matrix that holds construct the coherency matrix of a polarized complex visibility\n\nTo convert between a StokesVector and CoherencyMatrix use the convert function\n\nconvert(::StokesVector, CoherencyMatrix(1.0, 0.1, 0.1, 0.4))\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.DeltaPulse","page":"Home","title":"ComradeBase.DeltaPulse","text":"struct DeltaPulse{T} <: ComradeBase.Pulse\n\nA dirac comb pulse function. This means the image is just the dicrete Fourier transform\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.DensityAnalytic","page":"Home","title":"ComradeBase.DensityAnalytic","text":"DensityAnalytic\n\nInternal type for specifying the nature of the model functions. Whether they can be easily evaluated pointwise analytic. This is an internal type that may change.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.IntensityMap","page":"Home","title":"ComradeBase.IntensityMap","text":"struct IntensityMap{T, S<:(AbstractMatrix), F, K<:ComradeBase.Pulse} <: ComradeBase.AbstractIntensityMap{T, S<:(AbstractMatrix)}\n\nImage array type. This is an Matrix with a number of internal fields to describe the field of view, pixel size, and the pulse function that makes the image a continuous quantity.\n\nTo use it you just specify the array and the field of view/pulse julia img = IntensityMap(zeros(512512) 1000 1000 DeltaPulse)`\n\nFields\n\nim\nImage matrix\n\nfovx\nfield of view in x direction\n\nfovy\nfield of view in y direction\n\npsizex\npixel size in the x direction\n\npsizey\npixel size in the y direction\n\npulse\npulse function that turns the image grid into a continuous object\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.IsAnalytic","page":"Home","title":"ComradeBase.IsAnalytic","text":"struct IsAnalytic <: ComradeBase.DensityAnalytic\n\nDefines a trait that a states that a model is analytic. This is usually used with an abstract model where we use it to specify whether a model has a analytic fourier transform and/or image.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.IsPrimitive","page":"Home","title":"ComradeBase.IsPrimitive","text":"struct IsPrimitive\n\nTrait for primitive model\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.NotAnalytic","page":"Home","title":"ComradeBase.NotAnalytic","text":"struct NotAnalytic <: ComradeBase.DensityAnalytic\n\nDefines a trait that a states that a model is analytic. This is usually used with an abstract model where we use it to specify whether a model has does not have a easy analytic fourier transform and/or intensity function.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.NotPrimitive","page":"Home","title":"ComradeBase.NotPrimitive","text":"struct NotPrimitive\n\nTrait for not-primitive model\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.PrimitiveTrait","page":"Home","title":"ComradeBase.PrimitiveTrait","text":"abstract type PrimitiveTrait\n\nThis trait specifies whether the model is a primitive\n\nNotes\n\nThis will likely turn into a trait in the future so people can inject their models into Comrade more easily.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.Pulse","page":"Home","title":"ComradeBase.Pulse","text":"Pulse Pixel response function for a radio image model. This makes a discrete sampling continuous by picking a certain smoothing kernel for the image.\n\nNotes\n\nTo see the implemented Pulses please use the subtypes function i.e. subtypes(Pulse)\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.SingleStokes","page":"Home","title":"ComradeBase.SingleStokes","text":"struct SingleStokes{M, S} <: ComradeBase.AbstractModel\n\nHelper function that converts a model from something that compute polarized images to just a single stokes parameter. This is useful if you just want to fit a single stokes parameter.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.SqExpPulse","page":"Home","title":"ComradeBase.SqExpPulse","text":"struct SqExpPulse{T} <: ComradeBase.Pulse\n\nNormalized square exponential kernel, i.e. a Gaussian. Note the smoothness is modfied with ϵ which is the inverse variance in units of 1/pixels².\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.StokesVector","page":"Home","title":"ComradeBase.StokesVector","text":"struct StokesVector{T} <: StaticArrays.FieldVector{4, T}\n\nStatic vector that holds the stokes parameters of a polarized complex visibility\n\nTo convert between a StokesVector and CoherencyMatrix use the convert function\n\nconvert(::CoherencyMatrix, StokesVector(1.0, 0.1, 0.1, 0.4))\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.evpa-Tuple{StokesVector}","page":"Home","title":"ComradeBase.evpa","text":"evpa(m)\n\n\nCompute the evpa of a stokes vector or cohereny matrix.\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.flux-Union{Tuple{ComradeBase.AbstractIntensityMap{T, S}}, Tuple{S}, Tuple{T}, Tuple{F}} where {F, T<:StokesVector{F}, S}","page":"Home","title":"ComradeBase.flux","text":"flux(im)\n\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.flux-Union{Tuple{ComradeBase.AbstractIntensityMap{T, S}}, Tuple{S}, Tuple{T}} where {T, S}","page":"Home","title":"ComradeBase.flux","text":"flux(im::AbstractIntensityMap)\n\nComputes the flux of a intensity map\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.imanalytic-Tuple{Type{<:ComradeBase.AbstractModel}}","page":"Home","title":"ComradeBase.imanalytic","text":"imanalytic(::Type{<:AbstractModel})\n\nDetermines whether the model is pointwise analytic in the image domain, i.e. we can evaluate its intensity at an arbritrary point.\n\nIf IsAnalytic() then it will try to call intensity_point to calculate the intensity.\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.intensity_point","page":"Home","title":"ComradeBase.intensity_point","text":"intensity_point(model::AbstractModel, x, y, args...)\n\nFunction that computes the pointwise intensity if the model has the trait in the image domain IsAnalytic(). Otherwise it will use construct the image in visibility space and invert it.\n\n\n\n\n\n","category":"function"},{"location":"#ComradeBase.intensitymap","page":"Home","title":"ComradeBase.intensitymap","text":"intensitymap(model::AbstractModel, args...)\n\nComputes the intensity map of model. For the inplace version see intensitymap!\n\n\n\n\n\n","category":"function"},{"location":"#ComradeBase.intensitymap!","page":"Home","title":"ComradeBase.intensitymap!","text":"intensitymap!(buffer::AbstractMatrix, model::AbstractModel, args...)\n\nComputes the intensity map of model by modifying the buffer\n\n\n\n\n\n","category":"function"},{"location":"#ComradeBase.intensitymap!-Union{Tuple{M}, Tuple{ComradeBase.AbstractIntensityMap, M}, Tuple{ComradeBase.AbstractIntensityMap, M, Any}} where M","page":"Home","title":"ComradeBase.intensitymap!","text":"intensitymap!(img::AbstractIntensityMap, model, fovx, fovy, nx, ny; executor, pulse)\n\nComputes the intensity map or image of the model. This updates the IntensityMap object img.\n\nOptionally the user can specify the executor that uses FLoops.jl to specify how the loop is done. By default we use the SequentialEx which uses a single-core to construct the image.\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.intensitymap-Union{Tuple{M}, Tuple{M, Number, Number, Int64, Int64}} where M<:ComradeBase.AbstractModel","page":"Home","title":"ComradeBase.intensitymap","text":"intensitymap(model::AbstractModel, fovx, fovy, nx, ny; executor=SequentialEx(), pulse=DeltaPulse())\n\nComputes the intensity map or image of the model. This returns an IntensityMap object that have a field of view of fovx, fovy in the x and y direction  respectively with nx pixels in the x-direction and ny pixels in the y-direction.\n\nOptionally the user can specify the pulse function that converts the image from a discrete to continuous quantity, and the executor that uses FLoops.jl to specify how the loop is done. By default we use the SequentialEx which uses a single-core to construct the image.\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.isprimitive","page":"Home","title":"ComradeBase.isprimitive","text":"isprimitive(::Type)\n\nDispatch function that specifies whether a type is a primitive Comrade model. This function is used for dispatch purposes when composing models.\n\nNotes\n\nIf a user is specifying their own model primitive model outside of Comrade they need to specify if it is primitive\n\nstruct MyPrimitiveModel end\nComradeBase.isprimitive(::Type{MyModel}) = ComradeBase.IsPrimitive()\n\n\n\n\n\n","category":"function"},{"location":"#ComradeBase.m̆-Tuple{StokesVector}","page":"Home","title":"ComradeBase.m̆","text":"m̆(m)\n\n\nCompute the fractional linear polarization of a stokes vector or coherency matrix\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.radialextent","page":"Home","title":"ComradeBase.radialextent","text":"radialextent(model::AbstractModel)\n\nProvides an estimate of the radial size/extent of the model. This is used internally to estimate image size when plotting and using modelimage\n\n\n\n\n\n","category":"function"},{"location":"#ComradeBase.visanalytic-Tuple{Type{<:ComradeBase.AbstractModel}}","page":"Home","title":"ComradeBase.visanalytic","text":"visanalytic(::Type{<:AbstractModel})\n\nDetermines whether the model is pointwise analytic in Fourier domain, i.e. we can evaluate its fourier transform at an arbritrary point.\n\nIf IsAnalytic() then it will try to call visibility_point to calculate the complex visibilities. Otherwise it fallback to using the FFT that works for all models that can compute an image.\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.visibility_point","page":"Home","title":"ComradeBase.visibility_point","text":"visibility_point(model::AbstractModel, u, v, args...)\n\nFunction that computes the pointwise visibility. This must be implemented in the model interface if visanalytic(::Type{MyModel}) == IsAnalytic()\n\n\n\n\n\n","category":"function"}]
}
