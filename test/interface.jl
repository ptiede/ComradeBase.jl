using ComradeBase: IsAnalytic, NotAnalytic, IsPolarized, NotPolarized
@testset "interface" begin
    @test IsAnalytic()*NotAnalytic() == NotAnalytic()
    @test IsAnalytic()*IsAnalytic() == IsAnalytic()
    @test NotAnalytic()*NotAnalytic() == NotAnalytic()
    @test NotAnalytic()*IsAnalytic() == NotAnalytic()

    @test ComradeBase.ispolarized(ComradeBase.AbstractPolarizedModel) == IsPolarized()
    @test ComradeBase.ispolarized(ComradeBase.AbstractModel) == NotPolarized()

    @test IsPolarized()*NotPolarized() == IsPolarized()
    @test IsPolarized()*IsPolarized() == IsPolarized()
    @test NotPolarized()*NotPolarized() == NotPolarized()
    @test NotPolarized()*IsPolarized() == IsPolarized()

end
