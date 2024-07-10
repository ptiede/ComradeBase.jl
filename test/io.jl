μas2rad(x) = x * π / 180 / 3600 / 1e6

@testset "io.jl" begin
    imc = ComradeBase.load(joinpath(@__DIR__, "example_image.fits"), IntensityMap)
    ComradeBase.load(joinpath(@__DIR__, "example_image.fits"), IntensityMap{StokesParams})
    ime = ehtim.image.load_image(joinpath(@__DIR__, "example_image.fits"))
    @test pyconvert(Tuple, ime.imarr("I").shape) == size(imc)
    @test flux(imc) ≈ pyconvert(Float64, ime.total_flux())
    fov = fieldofview(imc)
    @test fov.X ≈ pyconvert(Float64, ime.fovx())
    @test fov.Y ≈ pyconvert(Float64, ime.fovx())
    ComradeBase.save("test.fits", imc)
    rm("test.fits")

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
    ComradeBase.save("ptest.fits", img1)
    ComradeBase.load("ptest.fits", IntensityMap)
    img2 = ComradeBase.load("ptest.fits", IntensityMap{StokesParams})
    rm("ptest.fits")
    @test parent(img1) ≈ parent(img2)
    @test parent(img1.X) ≈ parent(img2.X)
    @test parent(img1.Y) ≈ parent(img2.Y)

    # Now test shift
    Xnew = imc.X .+ μas2rad(50.0)
    Ynew = imc.Y .- μas2rad(75.0)

    imgc2 = IntensityMap(baseimage(imc), RectiGrid((;X=Xnew, Y=Ynew); header=header(imc)))
    ComradeBase.save("etest.fits", imgc2)
    imgc2l = ComradeBase.load("etest.fits", IntensityMap)
    @test parent(imgc2) ≈ parent(imgc2l)
    @test parent(imgc2.X) ≈ parent(imgc2l.X)
    @test parent(imgc2.Y) ≈ parent(imgc2l.Y)
    rm("etest.fits")
end
