@testset "images" begin
    X = range(-10.0, 10.0, length=64)
    Y = range(-10.0, 10.0, length=64)
    T = [0.0, 0.5, 0.8]
    F = [86e9, 230e9, 345e9]

    imp = rand(64, 64, 3, 3)

    img1 = IntensityMap(imp[:,:,1,1], (;X,Y))
    img2 = IntensityMap(imp, (;X, Y, T, F))
    img3 = IntensityMap(imp, (;X, Y, F, T))

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
        @test ComradeBase.basedim(nk.X) == X[5:10]
        @test ComradeBase.basedim(nk.Y) == Y[1:20]
    end

    @testset "keys" begin
        @test imagepixels(img1) == imagepixels(img2) == imagepixels(img3)
        @test pixelsizes(img1) == pixelsizes(img2) == pixelsizes(img3)
    end

    @testset "broadcast and map" begin
        @test img1.^2 isa typeof(img)
        @test cos.(img1) isa typeof(img)
        @test img1 .+ img1 isa typeof(img)
        @test cos.(img2[F=1,T=1]) isa IntensityMap
    end

    @testset "polarized" begin
        imgI = rand(64, 64, 3, 3)
        imgQ = rand(64, 64, 3, 3)
        imgU = rand(64, 64, 3, 3)
        imgV = rand(64, 64, 3, 3)

        imgP = StructArray{StokesParams}(I=imgI, Q=imgQ, U=imgU, V=imgV)
        img1 = IntensityMap(imgP[:,:,1,1], (;X,Y))
        img2 = IntensityMap(imgP, (;X, Y, T, F))
        simg1 = StokesIntensityMap(img1)
        simg2 = StokesIntensityMap(img2)

        @test flux(img1) ≈ flux(simg1)
        @test centroid(img1) == centroid(stokes(img1, :I))
        @test second_moment(img1) == second_moment(stokes(img1, :I))
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




end
