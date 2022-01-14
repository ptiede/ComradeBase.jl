var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ComradeBase","category":"page"},{"location":"#ComradeBase","page":"Home","title":"ComradeBase","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ComradeBase.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ComradeBase]","category":"page"},{"location":"#ComradeBase.BSplinePulse","page":"Home","title":"ComradeBase.BSplinePulse","text":"$(TYPEDEF)\n\nUses the basis spline (BSpline) kernel of order N. These are the kernel that come from recursively convolving the tophat kernel\n\n    B_0(x) = begincases 1  x  1  0  otherwise endcases\n\nN times.\n\nNotes\n\nBSpline kernels have a number of nice properties:\n\nSimple frequency response sinc(u2)^N\npreserve total intensity\n\nFor N>1 these kernels aren't actually interpolation kernels however, this doesn't matter for us.\n\nCurrently only the 0,1,3 order kernels are implemented.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.DeltaPulse","page":"Home","title":"ComradeBase.DeltaPulse","text":"struct DeltaPulse{T} <: ComradeBase.Pulse\n\nA dirac comb pulse function. This means the image is just the dicrete Fourier transform\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.DensityAnalytic","page":"Home","title":"ComradeBase.DensityAnalytic","text":"DensityAnalytic\n\nInternal type for specifying the nature of the model functions. Whether they can be easily evaluated pointwise analytic. This is an internal type that may change.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.IsAnalytic","page":"Home","title":"ComradeBase.IsAnalytic","text":"struct IsAnalytic <: ComradeBase.DensityAnalytic\n\nDefines a trait that a states that a model is analytic. This is usually used with an abstract model where we use it to specify whether a model has a analytic fourier transform and/or image.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.NotAnalytic","page":"Home","title":"ComradeBase.NotAnalytic","text":"struct NotAnalytic <: ComradeBase.DensityAnalytic\n\nDefines a trait that a states that a model is analytic. This is usually used with an abstract model where we use it to specify whether a model has does not have a easy analytic fourier transform and/or intensity function.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.PrimitiveTrait","page":"Home","title":"ComradeBase.PrimitiveTrait","text":"abstract type PrimitiveTrait\n\nThis trait specifies whether the model is a primitive\n\nNotes\n\nThis will likely turn into a trait in the future so people can inject their models into Comrade more easily.\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.Pulse","page":"Home","title":"ComradeBase.Pulse","text":"Pulse Pixel response function for a radio image model. This makes a discrete sampling continuous by picking a certain smoothing kernel for the image.\n\nNotes\n\nTo see the implemented Pulses please use the subtypes function i.e. subtypes(Pulse)\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.SqExpPulse","page":"Home","title":"ComradeBase.SqExpPulse","text":"struct SqExpPulse{T} <: ComradeBase.Pulse\n\nNormalized square exponential kernel, i.e. a Gaussian. Note the smoothness is modfied with ϵ which is the inverse variance in units of 1/pixels².\n\n\n\n\n\n","category":"type"},{"location":"#ComradeBase.flux-Union{Tuple{ComradeBase.AbstractIntensityMap{T, S}}, Tuple{S}, Tuple{T}} where {T, S}","page":"Home","title":"ComradeBase.flux","text":"flux(im)\n\n\nComputes the flux of a intensity map\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.imanalytic-Tuple{Type{<:ComradeBase.AbstractModel}}","page":"Home","title":"ComradeBase.imanalytic","text":"imanalytic(::Type{<:AbstractModel})\n\nDetermines whether the model is pointwise analytic in the image domain, i.e. we can evaluate its intensity at an arbritrary point.\n\nIf IsAnalytic() then it will try to call intensity_point to calculate the intensity.\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.intensity_point","page":"Home","title":"ComradeBase.intensity_point","text":"Function that computes the pointwise intensity if the model has the trait in the image domain IsAnalytic(). Otherwise it will use construct the image in visibility space and invert it.\n\n\n\n\n\n","category":"function"},{"location":"#ComradeBase.intensitymap","page":"Home","title":"ComradeBase.intensitymap","text":"Computes the intensity map of model. This version requires additional information to construct the grid.\n\nExample\n\nm = Gaussian()\n# field of view\nfovx, fovy = 5.0\nfovy = 5.0\n# number of pixels\nnx, ny = 128\n\nimg = intensitymap(m, fovx, fovy, nx, ny; pulse=DeltaPulse())\n\n\n\n\n\n","category":"function"},{"location":"#ComradeBase.intensitymap!","page":"Home","title":"ComradeBase.intensitymap!","text":"Computes the intensity map of model by modifying the input IntensityMap object\n\n\n\n\n\n","category":"function"},{"location":"#ComradeBase.isprimitive","page":"Home","title":"ComradeBase.isprimitive","text":"isprimitive(::Type)\n\nDispatch function that specifies whether a type is a primitive Comrade model. This function is used for dispatch purposes when composing models.\n\nNotes\n\nIf a user is specifying their own model primitive model outside of Comrade they need to specify if it is primitive\n\nstruct MyPrimitiveModel end\nComrade.isprimitive(::Type{MyModel}) = Comrade.IsPrimitive()\n\n\n\n\n\n","category":"function"},{"location":"#ComradeBase.visanalytic-Tuple{Type{<:ComradeBase.AbstractModel}}","page":"Home","title":"ComradeBase.visanalytic","text":"visanalytic(::Type{<:AbstractModel})\n\nDetermines whether the model is pointwise analytic in Fourier domain, i.e. we can evaluate its fourier transform at an arbritrary point.\n\nIf IsAnalytic() then it will try to call visibility_point to calculate the complex visibilities. Otherwise it fallback to using the FFT that works for all models that can compute an image.\n\n\n\n\n\n","category":"method"},{"location":"#ComradeBase.visibility_point","page":"Home","title":"ComradeBase.visibility_point","text":"Function that computes the pointwise visibility if the model has the trait in the fourier domain IsAnalytic(). Otherwise it will use the FFTW fallback.\n\n\n\n\n\n","category":"function"}]
}
