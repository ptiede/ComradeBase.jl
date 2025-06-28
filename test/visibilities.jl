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

@testset "Modifiers" begin
    u = randn(100) * 0.5
    v = randn(100) * 0.5
    t = sort(rand(100) * 0.5)
    f = fill(230.0e9, 100)
    guv = UnstructuredDomain((U = u, V = v, Ti = t, Fr = f))
    gim = imagepixels(10.0, 10.0, 64, 64)
    ma = GaussTest()

    @testset "Shifted" begin
        mas = shifted(ma, 0.1, 0.1)
        gims = imagepixels(10.0, 10.0, 64, 64, -0.1, -0.1)
        @test ComradeBase.intensity_point(ma, (X = 0.5, Y = 0.5)) ≈
            ComradeBase.intensity_point(mas, (X = 0.6, Y = 0.6))

        @test ComradeBase.visibility_point(ma, (U = 0.1, V = 0.1)) ≈
            ComradeBase.visibility_point(mas, (U = 0.1, V = 0.1)) * exp(-2π * 1im * (0.1 * 0.1 + 0.1 * 0.1))

        @test baseimage(intensitymap(ma, gim)) ≈ baseimage(intensitymap(mas, gims))
    end

    @testset "Renormed" begin
        m1 = 3.0 * ma
        m2 = ma * 3.0
        m2inv = ma / (1 / 3)
        p = (U = 4.0, V = 0.0)
        @test visibility(m1, p) == visibility(m2, p)
        @test ComradeBase.intensity_point(m1, (X = 0.5, Y = 0.5)) ≈
            ComradeBase.intensity_point(m2, (X = 0.5, Y = 0.5))
        @test ComradeBase.intensity_point(m1, (X = 0.5, Y = 0.5)) ≈
            ComradeBase.intensity_point(m2inv, (X = 0.5, Y = 0.5))

        @test intensitymap(m1, gim) ≈ intensitymap(m2, gim)
        @test visibilitymap(m1, guv) ≈ visibilitymap(m2, guv)
    end

    @testset "Stretched" begin
        mas = stretched(ma, 5.0, 4.0)
        gims = imagepixels(10.0 * 5.0, 10.0 * 4.0, 64, 64)
        @test ComradeBase.intensity_point(mas, (X = 0.5, Y = 0.5)) ≈
            ComradeBase.intensity_point(ma, (X = 0.5 / 5, Y = 0.5 / 4)) / 20

        @test ComradeBase.visibility_point(mas, (U = 0.1, V = 0.1)) ≈
            ComradeBase.visibility_point(ma, (U = 0.1 * 5, V = 0.1 * 4))

        @test baseimage(intensitymap(ma, gim)) ≈ baseimage(intensitymap(mas, gims))
        @test ComradeBase.radialextent(mas) ≈ ComradeBase.radialextent(ma) * 5.0
        @test flux(mas) ≈ flux(ma)

    end

    @testset "Rotated" begin
        ma = stretched(GaussTest(), 2.0, 1.0)
        mar = rotated(ma, π / 3)
        grs = imagepixels(10.0, 10.0, 64, 64; posang = π / 3)

        @test ComradeBase.intensity_point(mar, (X = 0.5, Y = 0.5)) ≈
            ComradeBase.intensity_point(
            ma,
            (
                X = 0.5 * cos(π / 3) - 0.5 * sin(π / 3),
                Y = 0.5 * sin(π / 3) + 0.5 * cos(π / 3),
            )
        )

        @test ComradeBase.visibility_point(mar, (U = 0.1, V = 0.1)) ≈
            ComradeBase.visibility_point(
            ma, (
                U = 0.1 * cos(π / 3) - 0.1 * sin(π / 3),
                V = 0.1 * sin(π / 3) + 0.1 * cos(π / 3),
            )
        )

        @test baseimage(intensitymap(ma, gim)) ≈ baseimage(intensitymap(mar, grs))

    end

    @testset "Compose" begin
        m = GaussTest()
        m1 = 5 * (m ∘ Rotate(π / 3) ∘ Shift(0.1, 0.1) ∘ Stretch(5.0, 4.0))
        m2 = 5 * modify(m, Stretch(5.0, 4.0), Shift(0.1, 0.1), Rotate(π / 3))

        @test m1 == m2
        @test_opt 5 * (m ∘ Rotate(π / 3) ∘ Shift(0.1, 0.1) ∘ Stretch(5.0, 4.0))
        @test_opt 5 * modify(m, Stretch(5.0, 4.0), Shift(0.1, 0.1), Rotate(π / 3))
    end
end
