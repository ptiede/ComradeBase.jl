function test_grid_interface(grid::ComradeBase.AbstractSingleDomain{D, E}) where {D, E}
    @test typeof(executor(grid)) == E
    arr = zeros(size(grid))
    @inferred ComradeBase.create_map(arr, grid)
    map = ComradeBase.create_map(arr, grid)
    @test typeof(map) == typeof(ComradeBase.allocate_map(Array{eltype(arr)}, grid))
    @inferred domainpoints(grid)
    @test typeof(DD.dims(grid)) == D

    @test header(grid) isa ComradeBase.AMeta
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
    test_grid_interface(gustr)

    @test fieldofview(grect) == (X = 20.0 + step(prect.X), Y = 20.0 + step(prect.Y))

    head = ComradeBase.MinimalHeader("M87", 90.0, 45, 21312, 230.0e9)
    g = RectiGrid(prect; header = head)
    @test header(g) === head
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

        @test img1 * 2 ≈ img2[:, :, 1, 1] * 2
        @test img1 .* imgI[:, :, 1, 1] ≈ img2[:, :, 1, 1] .* imgI[:, :, 1, 1]

        @test flux(img1) ≈ flux(img2)[1, 1, 1, 1]
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

function FiniteDifferences.to_vec(k::UnstructuredMap)
    v, b = to_vec(baseimage(k))
    d = axisdims(k)
    back(x) = UnstructuredMap(b(x), d)
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

# @testset "rrule UnstructuredMap" begin
#     data = rand(64)
#     g = UnstructuredDomain((X=randn(64), Y=randn(64)))
#     test_rrule(UnstructuredMap, data, g⊢NoTangent())
# end

# @testset "rrule baseimage" begin
#     data = rand(32, 24)
#     g = imagepixels(5.0, 10.0, 32, 24)
#     img = IntensityMap(data, g)

#     test_rrule(ComradeBase.baseimage, img)
# end

@testset "UnstructuredMap" begin
    pustr = (;
        X = range(-10.0, 10.0; length = 128),
        Y = range(-10.0, 10.0; length = 128),
        Fr = fill(230.0e9, 128),
        Ti = sort(rand(128)),
    )

    g = UnstructuredDomain(pustr)
    img = UnstructuredMap(rand(128), g)
    @test typeof(img .^ 2) == typeof(img)
    @test img[[1, 4, 6]] isa UnstructuredMap
    @test view(img, [1, 4, 6]) isa UnstructuredMap

    @test img[[6]][1] == img[6]
    @test @view(img[[6]])[1] == img[6]

    @test header(img) == header(g)
    @test executor(img) == executor(g)

    @test propertynames(img) == propertynames(g)
    @test img.X == g.X

    @testset "BroadcastStyle" begin
        using Base.Broadcast: BroadcastStyle, DefaultArrayStyle, Unknown, Style
        UStyle = ComradeBase.UnstructuredStyle

        # Style wraps the inner array's style
        @test BroadcastStyle(typeof(img)) isa UStyle{DefaultArrayStyle{1}}

        # Style preserves inner type through Val promotion
        s = UStyle{DefaultArrayStyle{1}}()
        @test s isa UStyle{DefaultArrayStyle{1}}
        @test UStyle{DefaultArrayStyle{1}}(Val(2)) isa UStyle{DefaultArrayStyle{1}}

        # Combining two UnstructuredStyles resolves inner styles
        s2 = UStyle{DefaultArrayStyle{1}}()
        @test BroadcastStyle(s, s2) isa UStyle{DefaultArrayStyle{1}}

        # Combining with DefaultArrayStyle{0} (scalars) keeps UnstructuredStyle
        @test BroadcastStyle(s, DefaultArrayStyle{0}()) isa UStyle

        # Reversed Style + UnstructuredStyle (symmetric to the above)
        @test BroadcastStyle(DefaultArrayStyle{0}(), s) isa UStyle

        # Unknown propagation through BroadcastStyle combinators (both directions)
        @test BroadcastStyle(s, Unknown()) isa Unknown
        @test BroadcastStyle(Unknown(), s) isa Unknown

        # Key new path: UnstructuredStyle(::Unknown) constructor used by two-arg combinator
        @test ComradeBase.UnstructuredStyle(Unknown()) isa Unknown

        # Unparameterized Val{N} constructor
        @test UStyle(Val(1)) isa UStyle{DefaultArrayStyle{1}}
        @test UStyle(Val(2)) isa UStyle{DefaultArrayStyle{2}}

        # AbstractArrayStyle on the right: (UnstructuredStyle, DefaultArrayStyle{1})
        @test BroadcastStyle(s, DefaultArrayStyle{1}()) isa UStyle
        # AbstractArrayStyle on the left: (DefaultArrayStyle{1}, UnstructuredStyle)
        @test BroadcastStyle(DefaultArrayStyle{1}(), s) isa UStyle

        # Tuple style branches (Style{Tuple} is not AbstractArrayStyle, needs its own dispatch)
        @test BroadcastStyle(s, Style{Tuple}()) isa UStyle
        @test BroadcastStyle(Style{Tuple}(), s) isa UStyle

        # Bare StructArrayStyle: the (UnstructuredStyle, StructArrayStyle) method
        # must disambiguate from StructArrays' own (AbstractArrayStyle,
        # StructArrayStyle) rule, which would otherwise be ambiguous with our
        # (UnstructuredStyle, AbstractArrayStyle) method.
        sas = BroadcastStyle(
            typeof(
                StructArray{StokesParams{Float64}}(
                    (I = rand(2), Q = rand(2), U = rand(2), V = rand(2))
                )
            )
        )
        @test sas isa StructArrays.StructArrayStyle
        @test BroadcastStyle(s, sas) isa UStyle
        # The reverse direct call has no explicit method and resolves to Unknown,
        # so combine_styles (what broadcasting actually uses) must recover the
        # UnstructuredStyle in both operand orders.
        sa128 = StructArray{StokesParams{Float64}}(
            (I = rand(128), Q = rand(128), U = rand(128), V = rand(128))
        )
        @test Base.Broadcast.combine_styles(img, sa128) isa UStyle
        @test Base.Broadcast.combine_styles(sa128, img) isa UStyle
    end

    @testset "broadcast correctness" begin
        # Unary broadcast
        @test parent(img .^ 2) == parent(img) .^ 2
        # Binary broadcast with two UnstructuredMaps
        img2 = UnstructuredMap(rand(128), g)
        res = img .+ img2
        @test res isa UnstructuredMap
        @test parent(res) == parent(img) .+ parent(img2)
        # Scalar broadcast (UnstructuredMap on left)
        res = img .* 3.0
        @test res isa UnstructuredMap
        @test parent(res) == parent(img) .* 3.0
        # Scalar broadcast reversed (scalar on left — exercises AbstractArrayStyle{0} + UnstructuredStyle)
        res = 3.0 .* img
        @test res isa UnstructuredMap
        @test parent(res) == 3.0 .* parent(img)
        # Chained broadcast
        res = img .* 2.0 .+ img2
        @test res isa UnstructuredMap
        @test parent(res) ≈ parent(img) .* 2.0 .+ parent(img2)
        # Domain is preserved through broadcast
        @test axisdims(res) === g
        # AbstractArrayStyle{1} on right (UnstructuredStyle, AbstractArrayStyle branch)
        arr = rand(128)
        res = img .* arr
        @test res isa UnstructuredMap
        @test parent(res) ≈ parent(img) .* arr
        @test axisdims(res) === g
        # AbstractArrayStyle{1} on left (AbstractArrayStyle, UnstructuredStyle branch)
        res = arr .* img
        @test res isa UnstructuredMap
        @test parent(res) ≈ arr .* parent(img)
        @test axisdims(res) === g
        # Tuple on right (UnstructuredStyle, Style{Tuple} branch)
        t = ntuple(_ -> 2.0, 128)
        res = img .* t
        @test res isa UnstructuredMap
        @test parent(res) ≈ parent(img) .* collect(t)
        @test axisdims(res) === g
        # Tuple on left (Style{Tuple}, UnstructuredStyle branch)
        res = t .* img
        @test res isa UnstructuredMap
        @test parent(res) ≈ collect(t) .* parent(img)
        @test axisdims(res) === g
    end

    @testset "broadcast in-place" begin
        dest = UnstructuredMap(zeros(128), g)
        dest .= img .^ 2
        @test parent(dest) == parent(img) .^ 2
        # In-place with two sources
        dest .= img .+ img
        @test parent(dest) == parent(img) .+ parent(img)
    end

    @testset "broadcast with StructArray backing" begin
        sdata = StructArray{StokesParams{Float64}}(
            (
                I = rand(128), Q = rand(128), U = rand(128), V = rand(128),
            )
        )
        simg = UnstructuredMap(sdata, g)
        @test simg isa UnstructuredMap{StokesParams{Float64}, <:StructArray}
        res = simg .+ simg
        @test res isa UnstructuredMap
        @test parent(res) isa StructArray
        @test parent(res).I ≈ parent(simg).I .+ parent(simg).I
    end

    @testset "broadcast bare StructArray against scalar UnstructuredMap" begin
        # Regression: a plain-backed UnstructuredMap broadcast against a *bare*
        # StructArray pairs bare UnstructuredStyle with bare StructArrayStyle,
        # which was ambiguous with StructArrays' own BroadcastStyle rule.
        sa = StructArray{StokesParams{Float64}}(
            (I = rand(128), Q = rand(128), U = rand(128), V = rand(128))
        )
        res = img .* sa
        @test res isa UnstructuredMap{<:StokesParams}
        @test parent(res) isa StructArray
        @test parent(res).I ≈ parent(img) .* sa.I
        @test axisdims(res) === g
        # reversed operand order (StructArray on the left)
        res = sa .* img
        @test res isa UnstructuredMap{<:StokesParams}
        @test parent(res) isa StructArray
        @test parent(res).Q ≈ sa.Q .* parent(img)
        @test axisdims(res) === g
    end

    @testset "broadcast StructVector with UnstructuredMap columns" begin
        sv = StructArray{StokesParams{ComplexF64}}(
            (
                I = rand(ComplexF64, 128),
                Q = UnstructuredMap(rand(ComplexF64, 128), g),
                U = UnstructuredMap(rand(ComplexF64, 128), g),
                V = UnstructuredMap(rand(ComplexF64, 128), g),
            )
        )
        res = sv .- sv
        @test length(res) == 128
        @test all(x -> all(iszero, (x.I, x.Q, x.U, x.V)), res)
    end
end
