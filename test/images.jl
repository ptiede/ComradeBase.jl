function test_grid_interface(grid::ComradeBase.AbstractSingleDomain{D, E}) where {D,E}
    @test typeof(executor(grid)) == E
    arr = zeros(size(grid))
    @inferred ComradeBase.create_map(arr, grid)
    map = ComradeBase.create_map(arr, grid)
    @test typeof(map) == typeof(ComradeBase.allocate_map(Array{eltype(arr)}, grid))
    @inferred domainpoints(grid)
    @test typeof(DD.dims(grid)) == D

    @test header(grid) isa ComradeBase.AbstractHeader
    @test keys(grid) == propertynames(grid)
    @test ndims(grid) == ndims(domainpoints(grid))

    @test keys(grid) == keys(named_dims(grid))
    @test firstindex(grid) == 1
    @test lastindex(grid) == length(grid)
    iterate(grid)
    # @test Base.front(grid) == DD.dims(grid)[1:end-1]
    grid[1]
    axes(grid)
    show(grid)
    show(IOBuffer(), MIME"text/plain"(), grid)
    summary(grid)
end

@testset "AbstractSingleDomain" begin
    ex = Serial()
    prect = (;X=range(-10.0, 10.0, length=128),
                        Y=range(-10.0, 10.0, length=128),
                        Fr = [230.0, 345.0],
                        Ti = sort(rand(24)))
    pustr = (;X=range(-10.0, 10.0, length=128),
                    Y=range(-10.0, 10.0, length=128),
                    Fr = fill(230e9, 128),
                    Ti = sort(rand(128)))
    grect = RectiGrid(prect)
    pc = phasecenter(grect)
    @test pc.X ≈ 0.0
    @test pc.Y ≈ 0.0
    gustr = UnstructuredDomain(pustr)

    test_grid_interface(grect)
    test_grid_interface(gustr)

    head = ComradeBase.MinimalHeader("M87", 90.0, 45, 21312, 230e9)
    g = RectiGrid(prect; header=head)
    @test header(g) == head

end

@testset "IntensityMap" begin
    x = X(range(-10.0, 10.0, length=64))
    y = Y(range(-10.0, 10.0, length=64))
    t = Ti([0.0, 0.5, 0.8])
    f = Fr([86e9, 230e9, 345e9])

    gsp = RectiGrid((x,y))
    g1 = RectiGrid((x, y, f, t))
    g2 = RectiGrid((x, y, t, f))

    imp = rand(64, 64, 3, 3)


    img1 = IntensityMap(imp[:,:,1,1], gsp)
    img2 = IntensityMap(imp, g1)
    img3 = IntensityMap(imp, g2)
    phasecenter(img2)
    centroid(img2)
    second_moment(img2)
    second_moment(img2; center=false)

    @test header(img1) == header(gsp)
    @test executor(img1) == executor(gsp)

    @test_throws ArgumentError img1.Fr


    @testset "Slicing" begin
        @test img1[X=1:1, Y=1:10] isa IntensityMap
        @test img1[X=5:10, Y=1:end-10] isa IntensityMap
        @test img2[X=1, Y=1] isa IntensityMap




        @test img1[X=1, Y=1] ≈ imp[1,1,1,1]
        @test img1[X=1, Y=1:10] ≈ imp[1,1:10,1,1]
        @test img1[X=5:10, Y=1:end-10] ≈ imp[5:10,1:end-10,1,1]
        @test img1[Y=1, X=1] ≈ imp[1,1,1,1]
        @test img2[X=1, Y=1] ≈ imp[1,1,:,:]


        subimg1 = img1[X=5:10, Y=1:20]
        nk = named_dims(subimg1)
        nnk = axisdims(subimg1)
        @test nnk.X == ComradeBase.basedim(nk.X)
        @test nnk.Y == ComradeBase.basedim(nk.Y)
        @test ComradeBase.basedim(nk.X) == ComradeBase.basedim(x[5:10])
        @test ComradeBase.basedim(nk.Y) == ComradeBase.basedim(y[1:20])
    end

    @testset "keys" begin
        @test pixelsizes(img1) == pixelsizes(img2) == pixelsizes(img3)
    end

    @testset "broadcast and map" begin
        @test img1.^2 isa typeof(img1)
        @test cos.(img1) isa typeof(img1)
        @test img1 .+ img1 isa typeof(img1)
        @test cos.(img2[Fr=1,Ti=1]) isa IntensityMap
    end

    @testset "polarized" begin
        imgI = rand(64, 64, 3, 3)
        imgQ = rand(64, 64, 3, 3)
        imgU = rand(64, 64, 3, 3)
        imgV = rand(64, 64, 3, 3)

        imgP = StructArray{StokesParams}(I=imgI, Q=imgQ, U=imgU, V=imgV)
        img1 = IntensityMap(imgP[:,:,1,1], RectiGrid((;X=x,Y=y)))
        img2 = IntensityMap(imgP, RectiGrid((x, y, t, f)))

        @test flux(img1) ≈ flux(img2)[1,1,1,1]
        @test centroid(img1) == centroid(stokes(img1, :I))
        @test second_moment(img1) == second_moment(stokes(img1, :I))
    end
end


@testset "io.jl" begin
    imc = ComradeBase.load(joinpath(@__DIR__, "example_image.fits"), IntensityMap)
    ime = ehtim.image.load_image(joinpath(@__DIR__, "example_image.fits"))
    @test pyconvert(Tuple, ime.imarr("I").shape) == size(imc)
    @test flux(imc) ≈ pyconvert(Float64, ime.total_flux())
    fov = fieldofview(imc)
    @test fov.X ≈ pyconvert(Float64, ime.fovx())
    @test fov.Y ≈ pyconvert(Float64, ime.fovx())
    ComradeBase.save("test.fits", imc)
    rm("test.fits")
end



function FiniteDifferences.to_vec(k::IntensityMap)
    v, b = to_vec(DD.data(k))
    back(x) = DD.rebuild(k, b(x))
    return v, back
end

function FiniteDifferences.to_vec(k::UnstructuredMap)
    v, b = to_vec(baseimage(k))
    d = axisdims(k)
    back(x) = UnstructuredMap(b(x), d)
    return v, back
end


@testset "ProjectTo" begin

    data = rand(32, 32)
    g = imagepixels(10.0, 10.0, 32, 32)
    img = IntensityMap(data, g)

    test_rrule(centroid, img)


    pr = ProjectTo(img)
    @test pr(data) == img
    @test pr(NoTangent()) == NoTangent()

    imgs = img[X=1, Y=:]
    prs = ProjectTo(imgs)
    @test prs(data[1,:]) == imgs
    @test prs(NoTangent()) == NoTangent()
end

@testset "rrule IntensityMap" begin
    data = rand(32, 32)
    g = imagepixels(10.0, 10.0, 32, 32)
    test_rrule(IntensityMap, data, g⊢NoTangent())
end

@testset "rrule UnstructuredMap" begin
    data = rand(64)
    g = UnstructuredDomain((X=randn(64), Y=randn(64)))
    test_rrule(UnstructuredMap, data, g⊢NoTangent())
end


@testset "rrule baseimage" begin
    data = rand(32, 24)
    g = imagepixels(5.0, 10.0, 32, 24)
    img = IntensityMap(data, g)

    test_rrule(ComradeBase.baseimage, img)
end

@testset "UnstructuredMap" begin
    pustr = (;X=range(-10.0, 10.0, length=128),
                    Y=range(-10.0, 10.0, length=128),
                    Fr = fill(230e9, 128),
                    Ti = sort(rand(128)))

    g = UnstructuredDomain(pustr)
    img = UnstructuredMap(rand(128), g)
    @test typeof(img.^2) == typeof(img)
    @test img[[1,4,6]] isa UnstructuredMap
    @test view(img, [1,4,6]) isa UnstructuredMap

    @test img[[6]][1] == img[6]
    @test @view(img[[6]])[1] == img[6]

    @test header(img) == header(g)
    @test executor(img) == executor(g)
end
