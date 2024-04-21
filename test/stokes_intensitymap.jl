@testset "StokesIntensityMap" begin
    x = X(range(-10.0, 10.0, length=64))
    y = Y(range(-10.0, 10.0, length=64))
    t = Ti([0.0, 0.5, 0.8])
    f = Fr([86e9, 230e9, 345e9])

    imgI = rand(64, 64, 3, 3)
    imgQ = rand(64, 64, 3, 3)
    imgU = rand(64, 64, 3, 3)
    imgV = rand(64, 64, 3, 3)

    imgP = StructArray{StokesParams}(I=imgI, Q=imgQ, U=imgU, V=imgV)
    img1 = IntensityMap(imgP[:,:,1,1], RectiGrid((;X=x,Y=y)))
    simg1 = StokesIntensityMap(img1)

    @test size(simg1) == size(img1)
    @test eltype(simg1) == StokesParams{Float64}
    @test ndims(simg1) == ndims(img1)
    @test ndims(typeof(simg1)) == ndims(img1)
    @test getindex(simg1, 1) ≈ getindex(img1, 1)
    @test getindex(simg1, 2, 2) ≈ getindex(img1, 2, 2)
    @test pixelsizes(simg1) == pixelsizes(img1)
    @test imagepixels(simg1) == imagepixels(img1)
    @test fieldofview(simg1) == fieldofview(img1)
    @test domaingrid(simg1) == domaingrid(img1)

    simg1[2, 2] = img1[2,2]
    StokesIntensityMap(imgI, imgQ, imgU, imgV, axisdims(img1))
    stokes(simg1, :I) ≈ stokes(img1, :I)
    summary(simg1)
    show(simg1)

    IntensityMap(simg1)
end
