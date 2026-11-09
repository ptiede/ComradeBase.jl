using ComradeBase: IsAnalytic, NotAnalytic, IsPolarized, NotPolarized
@testset "interface" begin
    @test IsAnalytic() * NotAnalytic() == NotAnalytic()
    @test IsAnalytic() * IsAnalytic() == IsAnalytic()
    @test NotAnalytic() * NotAnalytic() == NotAnalytic()
    @test NotAnalytic() * IsAnalytic() == NotAnalytic()

    @test ComradeBase.ispolarized(ComradeBase.AbstractPolarizedModel) == IsPolarized()
    @test ComradeBase.ispolarized(ComradeBase.AbstractModel) == NotPolarized()

    @test IsPolarized() * NotPolarized() == IsPolarized()
    @test IsPolarized() * IsPolarized() == IsPolarized()
    @test NotPolarized() * NotPolarized() == NotPolarized()
    @test NotPolarized() * IsPolarized() == IsPolarized()
end

@testset "public interface" begin
    model_interface = (
        :AbstractModel, :AbstractPolarizedModel,
        :intensity_point, :visibility_point,
        :visanalytic, :imanalytic, :IsAnalytic, :NotAnalytic,
        :ispolarized, :radialextent,
        :intensitymap_analytic, :intensitymap_analytic!,
        :intensitymap_numeric, :intensitymap_numeric!,
        :visibilitymap_analytic, :visibilitymap_analytic!,
        :visibilitymap_numeric, :visibilitymap_numeric!,
        :intensitymap_analytic_executor!,
        :AbstractDomain, :AbstractSingleDomain, :AbstractDualDomain, :AbstractRectiGrid,
        :StructuredMap,
        :create_map, :create_imgmap, :create_vismap,
        :allocate_map, :allocate_imgmap, :allocate_vismap,
        :basedim, :NoHeader, :MinimalHeader,
        :DomainParams, :paramtype,
        :rgetindex, :rsetindex!, :pointbroadcasted,
    )
    for n in model_interface
        @test Base.Docs.hasdoc(ComradeBase, n)
        @static if VERSION >= v"1.11"
            @test Base.ispublic(ComradeBase, n)
        end
    end
end

@testset "public functions take no trait arguments" begin
    m = ComradeBase.AbstractModel
    d = ComradeBase.AbstractSingleDomain
    @test !hasmethod(intensitymap, Tuple{IsAnalytic, m, ComradeBase.AbstractDomain})
    @test !hasmethod(intensitymap, Tuple{NotAnalytic, m, ComradeBase.AbstractDomain})
    @test !hasmethod(intensitymap!, Tuple{IsAnalytic, Any, m})
    @test !hasmethod(intensitymap!, Tuple{NotAnalytic, Any, m})
    @test !hasmethod(ComradeBase.allocate_imgmap, Tuple{IsPolarized, m, d})
    @test !hasmethod(ComradeBase.allocate_vismap, Tuple{NotPolarized, m, d})
    @test !isdefined(ComradeBase, :UnstructuredDomain)
end
