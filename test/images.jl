function test_grid_interface(grid::ComradeBase.AbstractSingleDomain{D, E}) where {D, E}
    @test typeof(executor(grid)) == E
    arr = zeros(size(grid))
    @inferred ComradeBase.create_map(arr, grid)
    map = ComradeBase.create_map(arr, grid)
    @test typeof(map) == typeof(ComradeBase.allocate_map(Array{eltype(arr)}, grid))
    @inferred domainpoints(grid)
    @test typeof(DD.dims(grid)) == D

    @test header(grid) isa ComradeBase.AMeta
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
    return summary(grid)
end

@testset "AbstractSingleDomain" begin
    ex = Serial()
    prect = (;
        X = range(-10.0, 10.0; length = 128),
        Y = range(-10.0, 10.0; length = 128),
        Fr = [230.0, 345.0],
        Ti = sort(rand(24)),
    )
    pustr = (;
        X = range(-10.0, 10.0; length = 128),
        Y = range(-10.0, 10.0; length = 128),
        Fr = fill(230.0e9, 128),
        Ti = sort(rand(128)),
    )
    grect = RectiGrid(prect)
    pc = phasecenter(grect)
    @test pc.X ≈ 0.0
    @test pc.Y ≈ 0.0
    gustr = UnstructuredDomain(pustr)

    test_grid_interface(grect)
    @test keys(grect) == propertynames(grect)
    test_grid_interface(gustr)
    @test gustr isa ComradeBase.StructuredDomain
    @test keys(gustr) == (:Pt,)
    @test propertynames(gustr) == (:X, :Y, :Fr, :Ti)

    @test fieldofview(grect) == (X = 20.0 + step(prect.X), Y = 20.0 + step(prect.Y))

    head = ComradeBase.MinimalHeader("M87", 90.0, 45, 21312, 230.0e9)
    g = RectiGrid(prect; header = head)
    @test header(g) === head
end

@testset "multidomain imagepixels" begin
    fr = Fr([230.0e9, 345.0e9])
    ti = Ti([1.0, 2.0, 3.0])
    g_fr = imagepixels(10.0, 20.0, 4, 5, mdims = (fr,))

    @test length(g_fr.X) == 4
    @test length(g_fr.Y) == 5
    @test length(g_fr.Fr) == 2
    @test collect(g_fr.Fr) == [230.0e9, 345.0e9]

    g_fr_ti = imagepixels(10.0, 20.0, 4, 5, mdims = (fr, ti))
    g_ti_fr = imagepixels(10.0, 20.0, 4, 5, mdims = (ti, fr))

    @test length(g_fr_ti.Fr) == 2
    @test length(g_fr_ti.Ti) == 3
    @test length(g_ti_fr.Ti) == 3
    @test length(g_ti_fr.Fr) == 2

    @test dims(g_fr_ti)[3] != dims(g_ti_fr)[3]
    @test dims(g_fr_ti)[4] != dims(g_ti_fr)[4]

    fr = Fr([230.0e9, 345.0e9])

    @test_throws AssertionError imagepixels(10.0, 20.0, 0, 5, mdims = (fr,))
    @test_throws AssertionError imagepixels(10.0, 20.0, 4, 0, mdims = (fr,))
end

@testset "IntensityMap" begin
    x = X(range(-10.0, 10.0; length = 64))
    y = Y(range(-10.0, 10.0; length = 64))
    t = Ti([0.0, 0.5, 0.8])
    f = Fr([86.0e9, 230.0e9, 345.0e9])

    gsp = RectiGrid((x, y))
    g1 = RectiGrid((x, y, f, t))
    g2 = RectiGrid((x, y, t, f))

    imp = rand(64, 64, 3, 3)

    img1 = IntensityMap(imp[:, :, 1, 1], gsp)
    img2 = IntensityMap(imp, g1)
    img3 = IntensityMap(imp, g2)
    phasecenter(img2)
    centroid(img2)
    second_moment(img2)
    second_moment(img2; center = false)

    @test header(img1) == header(gsp)
    @test executor(img1) == executor(gsp)

    @test_throws ArgumentError img1.Fr

    @testset "Slicing" begin
        @test img1[X = 1:1, Y = 1:10] isa IntensityMap
        @test img1[X = 5:10, Y = 1:(end - 10)] isa IntensityMap
        @test img2[X = 1, Y = 1] isa IntensityMap

        @test img1[X = 1, Y = 1] ≈ imp[1, 1, 1, 1]
        @test img1[X = 1, Y = 1:10] ≈ imp[1, 1:10, 1, 1]
        @test img1[X = 5:10, Y = 1:(end - 10)] ≈ imp[5:10, 1:(end - 10), 1, 1]
        @test img1[Y = 1, X = 1] ≈ imp[1, 1, 1, 1]
        @test img2[X = 1, Y = 1] ≈ imp[1, 1, :, :]

        subimg1 = img1[X = 5:10, Y = 1:20]
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
        @test img1 .^ 2 isa typeof(img1)
        @test cos.(img1) isa typeof(img1)
        @test img1 .+ img1 isa typeof(img1)
        @test cos.(img2[Fr = 1, Ti = 1]) isa IntensityMap
    end

    @testset "polarized" begin
        imgI = rand(64, 64, 3, 3)
        imgQ = rand(64, 64, 3, 3)
        imgU = rand(64, 64, 3, 3)
        imgV = rand(64, 64, 3, 3)

        imgP = StructArray{StokesParams}(; I = imgI, Q = imgQ, U = imgU, V = imgV)
        img1 = IntensityMap(imgP[:, :, 1, 1], RectiGrid((; X = x, Y = y)))
        img2 = IntensityMap(imgP, RectiGrid((x, y, t, f)))

        @test img1 * 2 ≈ img2[:, :, 1, 1, :] * 2
        @test img1 .* imgI[:, :, 1, 1] ≈ img2[:, :, 1, 1, :] .* imgI[:, :, 1, 1]

        @test flux(img1) ≈ StokesParams(map(c -> c[1, 1, 1, 1], Tuple(flux(img2)))...)
        @test centroid(img1) == centroid(stokes(img1, :I))
        @test second_moment(img1) == second_moment(stokes(img1, :I))

        img1 = IntensityMap(imgP[:, :, 1, 1], RectiGrid((; X = x, Y = y)))
        img2 = IntensityMap(Array(imgP[:, :, 1, 1]), RectiGrid((; X = x, Y = y)))

        @test stokes(img1, :I) ≈ stokes(img2, :I)
        @test stokes(img1, :Q) ≈ stokes(img2, :Q)
        @test stokes(img1, :U) ≈ stokes(img2, :U)
        @test stokes(img1, :V) ≈ stokes(img2, :V)

        @test stokes(first(imgP), :I) ≈ first(stokes(imgP, :I))
        @test stokes(last(imgP), :I) ≈ last(stokes(imgP, :I))


    end
end

function FiniteDifferences.to_vec(k::IntensityMap)
    v, b = to_vec(DD.data(k))
    back(x) = DD.rebuild(k, b(x))
    return v, back
end

# @testset "ProjectTo" begin

#     data = rand(32, 32)
#     g = imagepixels(10.0, 10.0, 32, 32)
#     img = IntensityMap(data, g)

#     # test_rrule(centroid, img)

#     # pr = ProjectTo(img)
#     # @test pr(data) == img
#     # @test pr(NoTangent()) == NoTangent()

#     # imgs = img[X=1, Y=:]
#     # prs = ProjectTo(imgs)
#     # @test prs(data[1,:]) == imgs
#     # @test prs(NoTangent()) == NoTangent()
# end

# @testset "rrule IntensityMap" begin
#     data = rand(32, 32)
#     g = imagepixels(10.0, 10.0, 32, 32)
#     # test_rrule(IntensityMap, data, g⊢NoTangent())
# end

# @testset "rrule baseimage" begin
#     data = rand(32, 24)
#     g = imagepixels(5.0, 10.0, 32, 24)
#     img = IntensityMap(data, g)

#     test_rrule(ComradeBase.baseimage, img)
# end

@testset "IntensityMap over a (Pt,) domain" begin
    pustr = (;
        X = range(-10.0, 10.0; length = 128),
        Y = range(-10.0, 10.0; length = 128),
        Fr = fill(230.0e9, 128),
        Ti = sort(rand(128)),
    )

    g = UnstructuredDomain(pustr)
    img = IntensityMap(rand(128), g)
    @test typeof(img .^ 2) == typeof(img)
    @test img[[1, 4, 6]] isa IntensityMap
    @test view(img, [1, 4, 6]) isa IntensityMap

    @test img[[6]][1] == img[6]
    @test @view(img[[6]])[1] == img[6]

    @test header(img) == header(g)
    @test executor(img) == executor(g)

    @test propertynames(img) == propertynames(g)
    @test img.X == g.X

    @testset "BroadcastStyle" begin
        using Base.Broadcast: BroadcastStyle, DefaultArrayStyle
        @test BroadcastStyle(typeof(img)) isa DD.DimensionalStyle{DefaultArrayStyle{1}}
        sa128 = StructArray{StokesParams{Float64}}(
            (I = rand(128), Q = rand(128), U = rand(128), V = rand(128))
        )
        @test Base.Broadcast.combine_styles(img, sa128) isa DD.DimensionalStyle
        @test Base.Broadcast.combine_styles(sa128, img) isa DD.DimensionalStyle
    end

    @testset "broadcast correctness" begin
        @test parent(img .^ 2) == parent(img) .^ 2
        img2 = IntensityMap(rand(128), g)
        res = img .+ img2
        @test res isa IntensityMap
        @test parent(res) == parent(img) .+ parent(img2)
        res = img .* 3.0
        @test res isa IntensityMap
        @test parent(res) == parent(img) .* 3.0
        res = 3.0 .* img
        @test res isa IntensityMap
        @test parent(res) == 3.0 .* parent(img)
        res = img .* 2.0 .+ img2
        @test res isa IntensityMap
        @test parent(res) ≈ parent(img) .* 2.0 .+ parent(img2)
        @test axisdims(res) === g
        arr = rand(128)
        res = img .* arr
        @test res isa IntensityMap
        @test parent(res) ≈ parent(img) .* arr
        @test axisdims(res) === g
        res = arr .* img
        @test res isa IntensityMap
        @test parent(res) ≈ arr .* parent(img)
        @test axisdims(res) === g
        t = ntuple(_ -> 2.0, 128)
        res = img .* t
        @test res isa IntensityMap
        @test parent(res) ≈ parent(img) .* collect(t)
        @test axisdims(res) === g
        res = t .* img
        @test res isa IntensityMap
        @test parent(res) ≈ collect(t) .* parent(img)
        @test axisdims(res) === g
    end

    @testset "broadcast in-place" begin
        dest = IntensityMap(zeros(128), g)
        dest .= img .^ 2
        @test parent(dest) == parent(img) .^ 2
        dest .= img .+ img
        @test parent(dest) == parent(img) .+ parent(img)
    end

    @testset "broadcast with a StructArray converted to Stokes storage" begin
        sdata = StructArray{StokesParams{Float64}}(
            (
                I = rand(128), Q = rand(128), U = rand(128), V = rand(128),
            )
        )
        simg = IntensityMap(sdata, g)
        @test simg isa StokesMap{Float64, 2}
        @test baseimage(simg) isa Matrix{Float64}
        @test baseimage(simg) == hcat(sdata.I, sdata.Q, sdata.U, sdata.V)
        res = simg .+ simg
        @test res isa StokesMap
        @test baseimage(res) ≈ 2 .* baseimage(simg)
    end

    @testset "broadcast bare StructArray against a scalar map" begin
        sa = StructArray{StokesParams{Float64}}(
            (I = rand(128), Q = rand(128), U = rand(128), V = rand(128))
        )
        res = img .* sa
        @test res isa IntensityMap{<:StokesParams}
        @test baseimage(res) isa StructArray
        @test baseimage(res).I ≈ parent(img) .* sa.I
        @test axisdims(res) === g
        res = sa .* img
        @test res isa IntensityMap{<:StokesParams}
        @test baseimage(res) isa StructArray
        @test baseimage(res).Q ≈ sa.Q .* parent(img)
        @test axisdims(res) === g
    end
end

@testset "IntensityMap keywords and domain checks" begin
    g = imagepixels(10.0, 10.0, 4, 3)
    rd = (Ti(1.0),)
    img = IntensityMap(rand(4, 3), g; refdims = rd, name = :flux)
    @test DD.refdims(img) == rd
    @test DD.name(img) == :flux
    d = UnstructuredDomain((; U = randn(5), V = randn(5)))
    vis = IntensityMap(rand(5), d; refdims = rd, name = :vis)
    @test DD.refdims(vis) == rd
    @test DD.name(vis) == :vis
    @test IntensityMap(img, g) === img
    @test_throws "the domain of the IntensityMap is not the RectiGrid given" IntensityMap(img, imagepixels(10.0, 10.0, 4, 3; posang = 0.1))
    @test_throws "the domain of the IntensityMap is not the StructuredDomain given" IntensityMap(vis, UnstructuredDomain((; U = randn(5), V = randn(5))))
end
