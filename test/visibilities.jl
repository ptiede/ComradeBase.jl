struct GaussTest{T} <: ComradeBase.AbstractModel
end
GaussTest() = GaussTest{Float64}()

ComradeBase.visanalytic(::Type{<:GaussTest}) = ComradeBase.IsAnalytic()
ComradeBase.imanalytic(::Type{<:GaussTest}) = ComradeBase.IsAnalytic()
ComradeBase.ispolarized(::Type{<:GaussTest}) = ComradeBase.NotPolarized()

function ComradeBase.intensity_point(m::GaussTest, p)
    (; X, Y) = p
    return exp(-(X^2 + Y^2) * inv(2)) / (2π)
end

function ComradeBase.visibility_point(m::GaussTest, p)
    (; U, V) = p
    return complex(exp(-2π^2 * (U^2 + V^2)))
end

ComradeBase.flux(::GaussTest{T}) where {T} = one(T)
ComradeBase.radialextent(::GaussTest{T}) where {T} = 5 * one(T)

struct GaussTestNA{T} <: ComradeBase.AbstractModel
end
GaussTestNA() = GaussTestNA{Float64}()

ComradeBase.visanalytic(::Type{<:GaussTestNA}) = ComradeBase.NotAnalytic()
ComradeBase.imanalytic(::Type{<:GaussTestNA}) = ComradeBase.NotAnalytic()
ComradeBase.ispolarized(::Type{<:GaussTestNA}) = ComradeBase.NotPolarized()

function ComradeBase.intensity_point(m::GaussTestNA, p)
    (; X, Y) = p
    return exp(-(X^2 + Y^2) * inv(2)) / (2π)
end

function ComradeBase.visibility_point(m::GaussTestNA, p)
    (; U, V) = p
    return complex(exp(-2π^2 * (U^2 + V^2)))
end

# Fake it to for testing
function ComradeBase.intensitymap_numeric(
        m::GaussTestNA,
        p::ComradeBase.AbstractSingleDomain
    )
    return ComradeBase.intensitymap_analytic(m, p)
end

function ComradeBase.intensitymap_numeric!(img, m::GaussTestNA)
    return ComradeBase.intensitymap_analytic!(img, m)
end

function ComradeBase.visibilitymap_numeric(
        m::GaussTestNA,
        p::ComradeBase.AbstractSingleDomain
    )
    return ComradeBase.visibilitymap_analytic(m, p)
end

function ComradeBase.visibilitymap_numeric!(vis, m::GaussTestNA)
    return ComradeBase.visibilitymap_analytic!(vis, m)
end

ComradeBase.flux(::GaussTestNA{T}) where {T} = one(T)
ComradeBase.radialextent(::GaussTestNA{T}) where {T} = 5 * one(T)

@testset "visibilities" begin
    u = 0.1 * randn(60)
    v = 0.1 * randn(60)
    ti = collect(Float64, 1:60)
    fr = fill(230.0e9, 60)
    m = GaussTest()
    p = (; U = u, V = v, Ti = ti, Fr = fr)
    g = UnstructuredDomain(p)
    @test visibilitymap(m, g) ≈ ComradeBase.visibilitymap_analytic(m, g)
    @test amplitudemap(m, g) ≈ abs.(ComradeBase.visibilitymap_analytic(m, g))
    closure_phasemap(m, g, g, g)
    logclosure_amplitudemap(m, g, g, g, g)
    @test angle.(bispectrummap(m, g, g, g)) ≈ closure_phasemap(m, g, g, g)

    vmappol = ComradeBase.allocate_vismap(ComradeBase.IsPolarized(), m, g)
    @test vmappol isa ComradeBase.UnstructuredMap
    @test eltype(vmappol) <: StokesParams

    gim = imagepixels(10.0, 10.0, 64, 64)
    imgpol = ComradeBase.allocate_imgmap(ComradeBase.IsPolarized(), m, gim)
    @test imgpol isa ComradeBase.IntensityMap
    @test eltype(imgpol) <: StokesParams
end

@testset "visibilities not analytic" begin
    u = 0.1 * randn(60)
    v = 0.1 * randn(60)
    ti = collect(Float64, 1:60)
    fr = fill(230.0e9, 60)
    m = GaussTestNA()
    p = (; U = u, V = v, Ti = ti, Fr = fr)
    g = UnstructuredDomain(p)
    @test visibilitymap(m, g) ≈ ComradeBase.visibilitymap_analytic(m, g)
    @test amplitudemap(m, g) ≈ abs.(ComradeBase.visibilitymap_analytic(m, g))
    closure_phasemap(m, g, g, g)
    logclosure_amplitudemap(m, g, g, g, g)
    @test angle.(bispectrummap(m, g, g, g)) ≈ closure_phasemap(m, g, g, g)
end
