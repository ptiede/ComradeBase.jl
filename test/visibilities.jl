
struct GaussTest{T} <: StokedBase.AbstractModel
end
GaussTest() = GaussTest{Float64}()

StokedBase.visanalytic(::Type{<:GaussTest}) = StokedBase.IsAnalytic()
StokedBase.imanalytic(::Type{<:GaussTest}) = StokedBase.IsAnalytic()
StokedBase.ispolarized(::Type{<:GaussTest}) = StokedBase.NotPolarized()

function StokedBase.intensity_point(m::GaussTest, p)
    (; X, Y) = p
    return exp(-(X^2 + Y^2) * inv(2)) / (2π)
end

function StokedBase.visibility_point(m::GaussTest, p)
    (; U, V) = p
    return complex(exp(-2π^2 * (U^2 + V^2)))
end

StokedBase.flux(::GaussTest{T}) where {T} = one(T)
StokedBase.radialextent(::GaussTest{T}) where {T} = 5 * one(T)

struct GaussTestNA{T} <: StokedBase.AbstractModel
end
GaussTestNA() = GaussTestNA{Float64}()

StokedBase.visanalytic(::Type{<:GaussTestNA}) = StokedBase.NotAnalytic()
StokedBase.imanalytic(::Type{<:GaussTestNA}) = StokedBase.NotAnalytic()
StokedBase.ispolarized(::Type{<:GaussTestNA}) = StokedBase.NotPolarized()

function StokedBase.intensity_point(m::GaussTestNA, p)
    (; X, Y) = p
    return exp(-(X^2 + Y^2) * inv(2)) / (2π)
end

function StokedBase.visibility_point(m::GaussTestNA, p)
    (; U, V) = p
    return complex(exp(-2π^2 * (U^2 + V^2)))
end

# Fake it to for testing
function StokedBase.intensitymap_numeric(m::GaussTestNA,
                                         p::StokedBase.AbstractSingleDomain)
    return StokedBase.intensitymap_analytic(m, p)
end

function StokedBase.intensitymap_numeric!(img, m::GaussTestNA)
    return StokedBase.intensitymap_analytic!(img, m)
end

function StokedBase.visibilitymap_numeric(m::GaussTestNA,
                                          p::StokedBase.AbstractSingleDomain)
    return StokedBase.visibilitymap_analytic(m, p)
end

function StokedBase.visibilitymap_numeric!(vis, m::GaussTestNA)
    return StokedBase.visibilitymap_analytic!(vis, m)
end

StokedBase.flux(::GaussTestNA{T}) where {T} = one(T)
StokedBase.radialextent(::GaussTestNA{T}) where {T} = 5 * one(T)

@testset "visibilities" begin
    u = 0.1 * randn(60)
    v = 0.1 * randn(60)
    ti = collect(Float64, 1:60)
    fr = fill(230e9, 60)
    m = GaussTest()
    p = (; U=u, V=v, Ti=ti, Fr=fr)
    g = UnstructuredDomain(p)
    @test visibilitymap(m, g) ≈ StokedBase.visibilitymap_analytic(m, g)
    @test amplitudemap(m, g) ≈ abs.(StokedBase.visibilitymap_analytic(m, g))
    closure_phasemap(m, g, g, g)
    logclosure_amplitudemap(m, g, g, g, g)
    @test angle.(bispectrummap(m, g, g, g)) ≈ closure_phasemap(m, g, g, g)

    vmappol = StokedBase.allocate_vismap(StokedBase.IsPolarized(), m, g)
    @test vmappol isa StokedBase.UnstructuredMap
    @test eltype(vmappol) <: StokesParams

    gim = imagepixels(10.0, 10.0, 64, 64)
    imgpol = StokedBase.allocate_imgmap(StokedBase.IsPolarized(), m, gim)
    @test imgpol isa StokedBase.IntensityMap
    @test eltype(imgpol) <: StokesParams
end

@testset "visibilities not analytic" begin
    u = 0.1 * randn(60)
    v = 0.1 * randn(60)
    ti = collect(Float64, 1:60)
    fr = fill(230e9, 60)
    m = GaussTestNA()
    p = (; U=u, V=v, Ti=ti, Fr=fr)
    g = UnstructuredDomain(p)
    @test visibilitymap(m, g) ≈ StokedBase.visibilitymap_analytic(m, g)
    @test amplitudemap(m, g) ≈ abs.(StokedBase.visibilitymap_analytic(m, g))
    closure_phasemap(m, g, g, g)
    logclosure_amplitudemap(m, g, g, g, g)
    @test angle.(bispectrummap(m, g, g, g)) ≈ closure_phasemap(m, g, g, g)
end
