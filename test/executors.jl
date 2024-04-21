using OhMyThreads

function testeximg(img, m, ex)
    g = axisdims(img)
    d = DD.dims(g)
    gnew = DD.rebuild(typeof(g), d, ex)
    img2 = intensitymap(m, gnew)
    @test img ≈ img2
    intensitymap!(img2, m)
    @test img ≈ img2
end

function testexvis(img, m, ex)
    g = axisdims(img)
    d = DD.dims(g)
    gnew = DD.rebuild(typeof(g), d, ex)
    img2 = visibilitymap(m, gnew)
    @test img ≈ img2
    visibilitymap!(img2, m)
    @test img ≈ img2
end


@testset "executors" begin
    u = 0.1*randn(60)
    v = 0.1*randn(60)
    ti = collect(Float64, 1:60)
    fr = fill(230e9, 60)
    m = GaussTest()

    @test ThreadsEx() === ThreadsEx(:dynamic)

    @testset "RectiGrid" begin
        pim = (;X=range(-10.0, 10.0, length=64), Y=range(-10.0, 10.0, length=64))
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

        puv = (U=range(-2.0, 2.0, length=128), V=range(-2.0, 2.0, length=64))
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
    end

    @testset "UnstructuredGrid" begin
        pim = (;X=randn(64), Y=randn(64))
        puv = (;U=u, V=v, Ti=ti, Fr=fr)
        gim = UnstructuredGrid(pim)
        img = intensitymap(m, gim)
        img0 = copy(img)
        intensitymap!(img, m)
        @test img ≈ img0


        testeximg(img, m, ThreadsEx())
        testeximg(img, m, ThreadsEx(:static))
        testeximg(img, m, DynamicScheduler())
        testeximg(img, m, StaticScheduler())
        testeximg(img, m, SerialScheduler())

        vis = visibilitymap(m, UnstructuredGrid(puv))
        vis0 = copy(vis)
        visibilitymap!(vis, m)
        @test vis ≈ vis0
        @test size(vis) == size(UnstructuredGrid(puv))
        testexvis(vis, m, ThreadsEx())
        testexvis(vis, m, ThreadsEx(:static))
        testexvis(vis, m, DynamicScheduler())
        testexvis(vis, m, StaticScheduler())
        testexvis(vis, m, SerialScheduler())
    end

end


@testset "executors NotAnalytic" begin
    u = 0.1*randn(60)
    v = 0.1*randn(60)
    ti = collect(Float64, 1:60)
    fr = fill(230e9, 60)
    m = GaussTestNA()

    @test ThreadsEx() === ThreadsEx(:dynamic)

    @testset "RectiGrid" begin
        pim = (;X=range(-10.0, 10.0, length=64), Y=range(-10.0, 10.0, length=64))
        gim = RectiGrid(pim)
        img = intensitymap(m, gim)

        @test img ≈ intensitymap(m, RectiGrid(pim; executor=ThreadsEx()))
        @test img ≈ intensitymap(m, RectiGrid(pim; executor=ThreadsEx(:static)))
        @test img ≈ intensitymap(m, RectiGrid(pim; executor=DynamicScheduler()))
        @test img ≈ intensitymap(m, RectiGrid(pim; executor=StaticScheduler()))
        @test img ≈ intensitymap(m, RectiGrid(pim; executor=SerialScheduler()))

        puv = (U=range(-2.0, 2.0, length=128), V=range(-2.0, 2.0, length=64))
        vis = visibilitymap(m, RectiGrid(puv))
        @test size(vis) == size(RectiGrid(puv))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor=ThreadsEx()))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor=ThreadsEx(:static)))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor=DynamicScheduler()))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor=StaticScheduler()))
        @test vis ≈ visibilitymap(m, RectiGrid(puv; executor=SerialScheduler()))
    end

    @testset "UnstructuredGrid" begin
        pim = (;X=randn(64), Y=randn(64))
        puv = (;U=u, V=v, Ti=ti, Fr=fr)
        gim = UnstructuredGrid(pim)
        img = intensitymap(m, gim)

        @test img ≈ intensitymap(m, UnstructuredGrid(pim; executor=ThreadsEx()))
        @test img ≈ intensitymap(m, UnstructuredGrid(pim; executor=ThreadsEx(:static)))
        @test img ≈ intensitymap(m, UnstructuredGrid(pim; executor=DynamicScheduler()))
        @test img ≈ intensitymap(m, UnstructuredGrid(pim; executor=StaticScheduler()))
        @test img ≈ intensitymap(m, UnstructuredGrid(pim; executor=SerialScheduler()))

        vis = visibilitymap(m, UnstructuredGrid(puv))
        @test size(vis) == size(UnstructuredGrid(puv))
        @test vis ≈ visibilitymap(m, UnstructuredGrid(puv; executor=ThreadsEx()))
        @test vis ≈ visibilitymap(m, UnstructuredGrid(puv; executor=ThreadsEx(:static)))
        @test vis ≈ visibilitymap(m, UnstructuredGrid(puv; executor=DynamicScheduler()))
        @test vis ≈ visibilitymap(m, UnstructuredGrid(puv; executor=StaticScheduler()))
        @test vis ≈ visibilitymap(m, UnstructuredGrid(puv; executor=SerialScheduler()))
    end

end
