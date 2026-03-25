using Reactant

@testset "Reactant" begin
    x = rand(54, 32)
    r = Reactant.to_rarray(x)

    @test @jit(ComradeBase.rgetindex(r, 10)) ≈ x[10]

    @test ComradeBase.rgetindex(x, 1:10) ≈ x[1:10]
    @test ComradeBase.rgetindex(x, :, 1) ≈ x[:, 1]
    @test ComradeBase.rgetindex(x, 1) ≈ x[1]

    @test @jit(ComradeBase.rgetindex(r, 1:10)) ≈ x[1:10]
    @test @jit(ComradeBase.rgetindex(r, :, 1)) ≈ x[:, 1]
    @test @jit(ComradeBase.rgetindex(r, 1:2, 1:2)) ≈ x[1:2, 1:2]

    @jit(ComradeBase.rsetindex!(r, 3.14, 20))
    @test @allowscalar ComradeBase.rgetindex(r, 20) ≈ 3.14
    @jit(ComradeBase.setindex!(r, ones(10, 10), 1:10, 1:10))
    @test ComradeBase.rgetindex(r, 1:10, 1:10) ≈ ones(10, 10)

    g = imagepixels(10.0, 10.0, 8, 8)
    go = @jit(identity(g))
    @test executor(go) isa ComradeBase.ReactantEx

    m1 = BlobTest(4.0)
    m2 = @jit BlobTest(ConcreteRNumber(m1.size))

    guv = UnstructuredDomain((; U = randn(64), V = randn(64)))
    guvr = Reactant.to_rarray(guv)

    @test baseimage(@jit(intensitymap(m2, go))) ≈ baseimage(intensitymap(m1, g))
    @test baseimage(@jit(visibilitymap(m2, guvr))) ≈ baseimage(visibilitymap(m1, guv))

end
