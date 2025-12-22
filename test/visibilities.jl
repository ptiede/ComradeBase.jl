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

struct DualDomain{D1 <: ComradeBase.AbstractSingleDomain, D2 <: ComradeBase.AbstractSingleDomain} <: ComradeBase.AbstractDualDomain
    imgdomain::D1
    visdomain::D2
end

function ComradeBase.intensitymap(m::ComradeBase.AbstractModel, p::DualDomain)
    return ComradeBase.intensitymap(m, imgdomain(p))
end

function ComradeBase.visibilitymap(m::ComradeBase.AbstractModel, p::DualDomain)
    return ComradeBase.visibilitymap(m, visdomain(p))
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

    img = intensitymap(m, gim)
    vis = visibilitymap(m, g)

    dd = DualDomain(gim, g)
    dmap = ComradeBase.dualmap(m, dd)
    @test ComradeBase.imgmap(dmap) ≈ img
    @test ComradeBase.vismap(dmap) ≈ vis

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

@testset "Methods" begin
    gim = imagepixels(10.0, 10.0, 64, 64)
    m = GaussTest()
    img = intensitymap(m, gim)
    @test all(x -> isapprox(x[1], x[2]), zip(centroid(img), centroid(m, gim)))
    @test flux(img) ≈ flux(m, gim)
    @test second_moment(img) ≈ second_moment(m, gim)

    display(img)
    show(img)

end

struct ExpParams{P} <: ComradeBase.DomainParams{P}
    scale::P
end

ComradeBase.build_param(param::ExpParams, p) = exp(param.scale) * p.Fr


@testset "Multi domain" begin
    p = ExpParams(0.1)
    @test ComradeBase.paramtype(typeof(p)) == Float64
    @test ComradeBase.paramtype(typeof(1.0)) == Float64

    @test ComradeBase.build_param(p, (; Fr = 2.0)) ≈ exp(0.1) * 2.0
    @test ComradeBase.build_param(p, (; Fr = 2.0)) ≈ getparam(ExpParams(p), :scale, (; Fr = 2.0))
    pp = ExpParams(p)
    @unpack_params scale = pp((; Fr = 2.0))
    @test scale ≈ exp(0.1) * 2.0
end
