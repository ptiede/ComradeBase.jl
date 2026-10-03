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

        testeximg(img, m, ThreadsEx())
        testeximg(img, m, ThreadsEx(:static))
        testeximg(img, m, DynamicScheduler())
        testeximg(img, m, StaticScheduler())
        testeximg(img, m, SerialScheduler())
        testeximg(img, m, CPU())
        testeximg(img, m, ThreadsEx(:Enzyme))
        testeximg(img, m, ThreadsEx(:Polyester))

        puv = (U = range(-2.0, 2.0; length = 128), V = range(-2.0, 2.0; length = 64))
        vis = visibilitymap(m, RectiGrid(puv))
        vis0 = copy(vis)
        visibilitymap!(vis, m)
        @test vis ≈ vis0

        @test size(vis) == size(RectiGrid(puv))
        testexvis(vis, m, ThreadsEx())
        testexvis(vis, m, ThreadsEx(:static))
        testexvis(vis, m, DynamicScheduler())
        testexvis(vis, m, StaticScheduler())
        testexvis(vis, m, SerialScheduler())
        testexvis(vis, m, CPU())
        testexvis(vis, m, ThreadsEx(:Enzyme))
        testexvis(vis, m, ThreadsEx(:Polyester))
    end

    @testset "StructuredDomain (Pt,)" begin
        pim = (; X = randn(64), Y = randn(64))
        puv = (; U = u, V = v, Ti = ti, Fr = fr)
        gim = UnstructuredDomain(pim)
        img = intensitymap(m, gim)
        img0 = copy(img)
        intensitymap!(img, m)
        @test img ≈ img0

        testeximg(img, m, ThreadsEx())
        testeximg(img, m, ThreadsEx(:static))
        testeximg(img, m, DynamicScheduler())
        testeximg(img, m, StaticScheduler())
        testeximg(img, m, SerialScheduler())
        testeximg(img, m, CPU())
        testeximg(img, m, ThreadsEx(:Enzyme))
        testeximg(img, m, ThreadsEx(:Polyester))

        vis = visibilitymap(m, UnstructuredDomain(puv))
        vis0 = copy(vis)
        visibilitymap!(vis, m)
        @test vis ≈ vis0
        @test size(vis) == size(UnstructuredDomain(puv))
        testexvis(vis, m, ThreadsEx())
        testexvis(vis, m, ThreadsEx(:static))
        testexvis(vis, m, DynamicScheduler())
        testexvis(vis, m, StaticScheduler())
        testexvis(vis, m, SerialScheduler())
        testexvis(vis, m, CPU())
        testexvis(vis, m, ThreadsEx(:Enzyme))
        testexvis(vis, m, ThreadsEx(:Polyester))
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
    mdims = (Fr([230.0e9, 345.0e9, 690.0e9]),)
    for (mdims, posang) in (((), 0.3), (mdims, 0.0), (mdims, 0.3))
        g = imagepixels(10.0, 12.0, 8, 6; mdims, posang)
        img = intensitymap(m, g)
        @test size(img) == size(g)
        @test baseimage(img) ≈ map(p -> ComradeBase.intensity_point(m, p), domainpoints(g)) .* prod(pixelsizes(g))
        guv = RectiGrid((U(range(-0.2, 0.2; length = 8)), V(range(-0.2, 0.2; length = 6)), mdims...); posang)
        vis = visibilitymap(m, guv)
        for ex in (
                ThreadsEx(), ThreadsEx(:static), DynamicScheduler(), StaticScheduler(),
                SerialScheduler(), CPU(), ThreadsEx(:Enzyme), ThreadsEx(:Polyester),
            )
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
end
