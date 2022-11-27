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
        @test img1[X=1, Y=1] ≈ imp[1,1,1,1]
        @test img1[X=1, Y=1:10] ≈ imp[1,1:10,1,1]
        @test img1[X=5:10, Y=1:end-10] ≈ imp[5:10,1:end-10,1,1]
        @test img1[Y=1, X=1] ≈ imp[1,1,1,1]
        @test img2[X=1, Y=1] ≈ imp[1,1,:,:]

        subimg1 = img1[X=5:10, Y=1:20]
        nk = named_axiskeys(subimg1)
        @test nk.X == X[5:10]
        @test nk.Y == Y[1:20]
    end

    @testset "keys" begin
        @test imagepixels(img1) == imagepixels(img2) == imagepixels(img3)
        @test pixelsizes(img1) == pixelsizes(img2) == pixelsizes(img3)
    end

    @testset "broadcast and map" begin
        @test typeof(img1) == typeof(img1.^2)
        @test typeof(img1) == typeof(cos.(img1))
        @test typeof(img1) == typeof(img1 .+ img1)
        @test typeof(img1) == typeof(cos.(img2[F=1,T=1]))
    end




end
