function testeximg(img, m, ex)
    g = axisdims(img)
    gnew = DD.rebuild(g; executor = ex)
    img2 = intensitymap(m, gnew)
    @test img ≈ img2
    intensitymap!(img2, m)
    return @test img ≈ img2
end

function testexvis(img, m, ex)
    g = axisdims(img)
    gnew = DD.rebuild(g; executor = ex)
    img2 = visibilitymap(m, gnew)
    @test img ≈ img2
    visibilitymap!(img2, m)
    return @test img ≈ img2
end

const loopexecutors = (
    ThreadsEx(), ThreadsEx(:static), DynamicScheduler(), StaticScheduler(), SerialScheduler(),
    CPU(), ThreadsEx(:Enzyme), ThreadsEx(:Polyester),
)

@testset "executors" begin
    u = 0.1 * randn(60)
    v = 0.1 * randn(60)
    ti = collect(Float64, 1:60)
    fr = fill(230.0e9, 60)
    m = GaussTest()

    @test ThreadsEx() === ThreadsEx(:dynamic)

    @testset "RectiGrid" begin
        pim = (; X = range(-10.0, 10.0; length = 64), Y = range(-10.0, 10.0; length = 64))
        gim = RectiGrid(pim)
        img = intensitymap(m, gim)
        img0 = copy(img)
        intensitymap!(img, m)
        @test img ≈ img0

        foreach(ex -> testeximg(img, m, ex), loopexecutors)

        puv = (U = range(-2.0, 2.0; length = 128), V = range(-2.0, 2.0; length = 64))
        vis = visibilitymap(m, RectiGrid(puv))
        vis0 = copy(vis)
        visibilitymap!(vis, m)
        @test vis ≈ vis0

        @test size(vis) == size(RectiGrid(puv))
        foreach(ex -> testexvis(vis, m, ex), loopexecutors)
    end

    @testset "StructuredDomain (Pt,)" begin
        pim = (; X = randn(64), Y = randn(64))
        puv = (; U = u, V = v, Ti = ti, Fr = fr)
        gim = UnstructuredDomain(pim)
        img = intensitymap(m, gim)
        img0 = copy(img)
        intensitymap!(img, m)
        @test img ≈ img0

        foreach(ex -> testeximg(img, m, ex), loopexecutors)

        vis = visibilitymap(m, UnstructuredDomain(puv))
        vis0 = copy(vis)
        visibilitymap!(vis, m)
        @test vis ≈ vis0
        @test size(vis) == size(UnstructuredDomain(puv))
        foreach(ex -> testexvis(vis, m, ex), loopexecutors)
    end
end

@testset "executors NotAnalytic" begin
    u = 0.1 * randn(60)
    v = 0.1 * randn(60)
    ti = collect(Float64, 1:60)
    fr = fill(230.0e9, 60)
    m = GaussTestNA()

    @test ThreadsEx() === ThreadsEx(:dynamic)

    @testset "RectiGrid" begin
        pim = (; X = range(-10.0, 10.0; length = 64), Y = range(-10.0, 10.0; length = 64))
        gim = RectiGrid(pim)
        img = intensitymap(m, gim)

        @test img ≈ intensitymap(m, RectiGrid(pim; executor = ThreadsEx()))
        @test img ≈ intensitymap(m, RectiGrid(pim; executor = ThreadsEx(:static)))
        @test img ≈ intensitymap(m, RectiGrid(pim; executor = DynamicScheduler()))
        @test img ≈ intensitymap(m, RectiGrid(pim; executor = StaticScheduler()))
        @test img ≈ intensitymap(m, RectiGrid(pim; executor = SerialScheduler()))

        puv = (U = range(-2.0, 2.0; length = 128), V = range(-2.0, 2.0; length = 64))
        vis = visibilitymap(m, RectiGrid(puv))
        @test size(vis) == size(RectiGrid(puv))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor = ThreadsEx()))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor = ThreadsEx(:static)))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor = DynamicScheduler()))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor = StaticScheduler()))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor = SerialScheduler()))
    end

    @testset "StructuredDomain (Pt,)" begin
        pim = (; X = randn(64), Y = randn(64))
        puv = (; U = u, V = v, Ti = ti, Fr = fr)
        gim = UnstructuredDomain(pim)
        img = intensitymap(m, gim)

        @test img ≈ intensitymap(m, UnstructuredDomain(pim; executor = ThreadsEx()))
        @test img ≈ intensitymap(m, UnstructuredDomain(pim; executor = ThreadsEx(:static)))
        @test img ≈ intensitymap(m, UnstructuredDomain(pim; executor = DynamicScheduler()))
        @test img ≈ intensitymap(m, UnstructuredDomain(pim; executor = StaticScheduler()))
        @test img ≈ intensitymap(m, UnstructuredDomain(pim; executor = SerialScheduler()))

        vis = visibilitymap(m, UnstructuredDomain(puv))
        @test size(vis) == size(UnstructuredDomain(puv))
        @test vis ≈ visibilitymap(m, UnstructuredDomain(puv; executor = ThreadsEx()))
        @test vis ≈ visibilitymap(m, UnstructuredDomain(puv; executor = ThreadsEx(:static)))
        @test vis ≈ visibilitymap(m, UnstructuredDomain(puv; executor = DynamicScheduler()))
        @test vis ≈ visibilitymap(m, UnstructuredDomain(puv; executor = StaticScheduler()))
        @test vis ≈ visibilitymap(m, UnstructuredDomain(puv; executor = SerialScheduler()))
    end
end

@testset "EnzymeExecutors" begin
    u = 0.1 * randn(60)
    v = 0.1 * randn(60)
    ti = collect(Float64, 1:60)
    fr = fill(230.0e9, 60)
    m = GaussTest()

    pim = (; X = range(-10.0, 10.0; length = 64), Y = range(-10.0, 10.0; length = 64))
    gim = RectiGrid(pim)
    guv = UnstructuredDomain((; U = u, V = v, Ti = ti, Fr = fr))
    guvm = RectiGrid((; U = u, V = v))

    img = intensitymap(m, gim)
    vis = visibilitymap(m, guv)
    vism = visibilitymap(m, guvm)
    testeximg(img, m, ThreadsEx(:Enzyme))
    testexvis(vis, m, ThreadsEx(:Enzyme))
    testexvis(vism, m, ThreadsEx(:Enzyme))
end

@testset "executors on (X, Y, Fr) and rotated grids" begin
    m = GaussTest()
    frs = (Fr([230.0e9, 345.0e9, 690.0e9]),)
    for (extra, posang) in (((), 0.3), (frs, 0.0), (frs, 0.3))
        g = gridproduct(spatialgrid(10.0, 12.0, 8, 6; posang), extra...)
        img = intensitymap(m, g)
        @test size(img) == size(g)
        @test baseimage(img) ≈ map(p -> ComradeBase.intensity_point(m, p), domainpoints(g)) .* prod(pixelsizes(g))
        guv = RectiGrid((U(range(-0.2, 0.2; length = 8)), V(range(-0.2, 0.2; length = 6)), extra...); posang)
        vis = visibilitymap(m, guv)
        for ex in loopexecutors
            testeximg(img, m, ex)
            testexvis(vis, m, ex)
        end
    end
end

@testset "@threaded" begin
    function threadsum(ex, n)
        out = zeros(Int, n)
        ComradeBase.@threaded ex for i in 1:n
            out[i] = i
        end
        return sum(out)
    end
    function threadsum(n)
        out = zeros(Int, n)
        ComradeBase.@threaded for i in 1:n
            out[i] = i
        end
        return sum(out)
    end
    @test threadsum(20) == 210
    for ex in (Serial(), ThreadsEx(), ThreadsEx(:static), ThreadsEx(:dynamic))
        @test threadsum(ex, 20) == 210
    end
    @test_throws "@threaded does not handle the executor ThreadsEx{:Polyester}()" threadsum(ThreadsEx(:Polyester), 20)
    @test_throws "@threaded does not handle the executor CPU" threadsum(CPU(), 20)
    @test_throws ArgumentError threadsum(DynamicScheduler(), 20)

    function spectrum!(ex, ns, a, k)
        ComradeBase.@threaded ex for i in eachindex(k)
            for j in eachindex(k)
                ns[j, i] = inv(1 + (k[j]^2 + k[i]^2)^a)
            end
        end
        return nothing
    end
    function spectrumloss(a, ex, k)
        ns = zeros(typeof(a), length(k), length(k))
        spectrum!(ex, ns, a, k)
        return sum(abs2, ns)
    end
    k = collect(range(0.1, 2.0; length = 12))
    fd = (spectrumloss(1.3 + 1.0e-6, Serial(), k) - spectrumloss(1.3 - 1.0e-6, Serial(), k)) / 2.0e-6
    # Without runtime activity: the Serial loop must not put `k` and `ns` in one closure.
    rev = Enzyme.autodiff(Enzyme.Reverse, spectrumloss, Enzyme.Active, Enzyme.Active(1.3), Enzyme.Const(Serial()), Enzyme.Const(k))[1][1]
    fwd = Enzyme.autodiff(Enzyme.Forward, spectrumloss, Enzyme.Duplicated(1.3, 1.0), Enzyme.Const(Serial()), Enzyme.Const(k))[1]
    @test rev ≈ fd rtol = 1.0e-5
    @test fwd ≈ fd rtol = 1.0e-5
end

@testset "unknown executor" begin
    m = GaussTest()
    @test_throws "the executor ThreadsEx{:nope}() cannot run a loop" intensitymap(m, spatialgrid(10.0, 10.0, 4, 4; executor = ThreadsEx(:nope)))
    @test_throws "the executor ThreadsEx{:nope}() cannot run a loop" visibilitymap(m, UnstructuredDomain((; U = randn(4), V = randn(4)); executor = ThreadsEx(:nope)))
end

@testset "image executor allocation" begin
    m = GaussTest()
    for g in (spatialgrid(10.0, 10.0, 8, 6), spatialgrid(10.0, 10.0, 8, 6; posang = 0.3) ⊗ Fr([230.0e9, 345.0e9]))
        img = intensitymap(m, g)
        intensitymap!(img, m)
        @test (@allocated intensitymap!(img, m)) == 0
        JET.@test_opt target_modules = (ComradeBase,) intensitymap!(img, m)
    end
end

@testset "broadcast executor allocation" begin
    d = StructuredDomain((Pt(9), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(9), v = 3.0e4 .* randn(9), executor = CPU())
    ComradeBase.shapedcoords(d)
    shaped = @allocated ComradeBase.shapedcoords(d)
    for (m, ncomp) in ((GaussTest(), 1), (PolTest(1.5), 4))
        vis = visibilitymap(m, d)
        visibilitymap!(vis, m)
        @test (@allocated visibilitymap!(vis, m)) <= ncomp * shaped
        img = intensitymap(m, spatialgrid(10.0, 10.0, 8, 6; executor = CPU()))
        intensitymap!(img, m)
        @test (@allocated intensitymap!(img, m)) == 0
    end
end
