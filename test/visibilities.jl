

struct GaussTest{T} <: ComradeBase.AbstractModel end
GaussTest() = GaussTest{Float64}()

ComradeBase.visanalytic(::Type{<:GaussTest}) = ComradeBase.IsAnalytic()
ComradeBase.imanalytic(::Type{<:GaussTest}) = ComradeBase.IsAnalytic()
ComradeBase.ispolarized(::Type{<:GaussTest}) = ComradeBase.NotPolarized()

function ComradeBase.intensity_point(::GaussTest, p)
    (;X, Y) = p
    return exp(-(X^2+Y^2)/2)/2π
end

function ComradeBase.visibility_point(::GaussTest, p)
    u = p.U
    v = p.V
    return complex(exp(-2π^2*(u^2 + v^2)))
end

ComradeBase.flux(::GaussTest{T}) where {T} = one(T)
ComradeBase.radialextent(::GaussTest{T}) where {T} = 5*one(T)


@testset "visibilities" begin
    U = 0.1*randn(60)
    V = 0.1*randn(60)
    Ti = collect(Float64, 1:60)
    Fr = fill(230e9, 60)
    m = GaussTest()
    g = UnStructuredGrid((;U, V, Ti, Fr))
    @test visibilitymap(m, g) ≈ ComradeBase.visibilitymap_analytic(m, g)
    @test amplitudemap(m, g) ≈ abs.(ComradeBase.visibilitymap_analytic(m, g))
    closure_phasemap(m, g, g, g)
    logclosure_amplitudemap(m, g, g, g, g)
    @test angle.(bispectrummap(m, g, g, g)) ≈ closure_phasemap(m, g, g, g)
end
