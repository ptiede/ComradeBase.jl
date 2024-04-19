using ComradeBase: IsAnalytic, NotAnalytic
@testset "interface" begin
    @test IsAnalytic()*NotAnalytic() == NotAnalytic()
    @test IsAnalytic()*IsAnalytic() == IsAnalytic()
    @test NotAnalytic()*NotAnalytic() == NotAnalytic()
    @test NotAnalytic()*IsAnalytic() == NotAnalytic()

    @test ComradeBase.ispolarized(ComradeBase.AbstractPolarizedModel) == IsPolarized()
    @test ComradeBase.ispolarized(ComradeBase.AbstractModel) == NotPolarized()
end
