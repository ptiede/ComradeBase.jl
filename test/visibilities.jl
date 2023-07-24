

struct GaussTest{T} <: ComradeBase.AbstractModel end
GaussTest() = GaussTest{Float64}()

ComradeBase.visanalytic(::Type{<:GaussTest}) = ComradeBase.IsAnalytic()
ComradeBase.imanalytic(::Type{<:GaussTest}) = ComradeBase.IsAnalytic()
ComradeBase.ispolarized(::Type{<:GaussTest}) = ComradeBase.NotPolarized()

function ComradeBase.intensity_point(::GaussTest, p)
    (;X, Y) = p
    return exp(-(X^2+Y^2)/2)/2π
end

function ComradeBase.visibility_point(::GaussTest, u, v, time, freq) where {T}
    return complex(exp(-2π^2*(u^2 + v^2)))
end

ComradeBase.flux(::GaussTest{T}) where {T} = one(T)
ComradeBase.radialextent(::GaussTest{T}) where {T} = 5*one(T)


@testset "visibilities" begin
    U = 0.1*randn(60)
    V = 0.1*randn(60)
    T = collect(Float64, 1:60)
    F = fill(230e9, 60)
    m = GaussTest()
    @test visibilities(m, (;U,V,T,F)) ≈ ComradeBase.visibilities_analytic(m, U, V, T, F)
    @test amplitudes(m, (;U, V, T, F)) ≈ abs.(ComradeBase.visibilities_analytic(m, U, V, T, F))
    p = (;U, V, T, F)
    closure_phases(m, p, p, p)
    logclosure_amplitudes(m, p, p, p, p)
    @test angle.(bispectra(m, p, p, p)) ≈ closure_phases(m, p, p, p)

end
