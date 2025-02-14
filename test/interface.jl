using StokedBase: IsAnalytic, NotAnalytic, IsPolarized, NotPolarized
@testset "interface" begin
    @test IsAnalytic() * NotAnalytic() == NotAnalytic()
    @test IsAnalytic() * IsAnalytic() == IsAnalytic()
    @test NotAnalytic() * NotAnalytic() == NotAnalytic()
    @test NotAnalytic() * IsAnalytic() == NotAnalytic()

    @test StokedBase.ispolarized(StokedBase.AbstractPolarizedModel) == IsPolarized()
    @test StokedBase.ispolarized(StokedBase.AbstractModel) == NotPolarized()

    @test IsPolarized() * NotPolarized() == IsPolarized()
    @test IsPolarized() * IsPolarized() == IsPolarized()
    @test NotPolarized() * NotPolarized() == NotPolarized()
    @test NotPolarized() * IsPolarized() == IsPolarized()
end
