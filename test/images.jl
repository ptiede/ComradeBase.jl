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

@testset "grid products" begin
    fr = Fr([230.0e9, 345.0e9])
    ti = Ti([1.0, 2.0, 3.0])
    g = spatialgrid(10.0, 20.0, 4, 5; posang = 0.2, executor = ThreadsEx(), header = ComradeBase.MinimalHeader("s", 1.0, 2.0, 3.0, 4.0))
    @test DD.name(dims(g)) == (:X, :Y)
    @test size(g) == (4, 5)
    @test values(pixelsizes(spatialgrid(10.0, 20.0, 1, 4))) == (10.0, 5.0)
    @test values(fieldofview(spatialgrid(10.0, 20.0, 1, 4))) == (10.0, 20.0)

    g_fr_ti = g ⊗ fr ⊗ ti
    g_ti_fr = gridproduct(g, ti, fr)
    @test DD.name(dims(g_fr_ti)) == (:X, :Y, :Fr, :Ti)
    @test DD.name(dims(g_ti_fr)) == (:X, :Y, :Ti, :Fr)
    @test size(g_fr_ti) == (4, 5, 2, 3)
    @test collect(g_fr_ti.Fr) == [230.0e9, 345.0e9]
    @test posang(g_fr_ti) == 0.2
    @test executor(g_fr_ti) === executor(g)
    @test header(g_fr_ti) === header(g)
    @test dims(gridproduct(g)) == dims(g)

    ifr = frames(Fr, [226.0e9, 228.0e9, 230.0e9])
    gs = spatialgrid(10.0, 20.0, 4, 5)
    @inferred spatialgrid(10.0, 20.0, 4, 5)
    @inferred spatialgrid(10.0f0, 20.0f0, 4, 5)
    @inferred gridproduct(gs, ifr)
    @inferred gs ⊗ ifr

    @test_throws "the grid already has a Fr dim" g ⊗ fr ⊗ Fr([1.0])
    @test_throws "the appended dims repeat a dim: (:Ti, :Ti)" gridproduct(g, ti, Ti([4.0]))
    @test_throws "cannot append the spatial dim X" g ⊗ X(1:3)
    @test_throws "cannot append the spatial dim V" g ⊗ V(1:3)
    @test_throws "cannot append Pt to a grid" g ⊗ Pt(1:3)
    @test_throws "the right factor of a grid product must be a non-spatial dim" g ⊗ gs
    @test_throws "the number of pixels must be positive, got nx = 0, ny = 5" spatialgrid(10.0, 20.0, 0, 5)
    @test_throws "the number of pixels must be positive, got nx = 4, ny = 0" spatialgrid(10.0, 20.0, 4, 0)
end

@testset "frames and frameindex" begin
    ti = @inferred frames(Ti, [0.0, 1.5, 4.0], [1.0, 3.0, 5.0])
    @test ti isa Ti
    @test DD.sampling(ti) isa DD.Intervals
    @test collect(ti) == [0.5, 2.25, 4.5]
    @test DD.intervalbounds(ti) == [(0.0, 1.0), (1.5, 3.0), (4.0, 5.0)]
    fr = @inferred frames(Fr, [1.0, 2.0, 3.0])
    @test DD.intervalbounds(fr) == [(1.0, 2.0), (2.0, 3.0)]

    @test frameindex(ti, [0.0, 1.0, 1.5, 3.0, 4.5, 5.0]) == [1, 1, 2, 2, 3, 3]
    @test frameindex(fr, Float32[1, 2, 2.5, 3]) == [1, 2, 2, 2]
    @test frameindex(DD.lookup(fr), [2.0]) == [2]
    @test frameindex(Ti([0.0, 1.0, 2.0]), [2.0, 0.0, 1.0f0]) == [3, 1, 2]
    @test frameindex(Ti([-0.0, 1.0]), [0.0, -0.0]) == [1, 1]
    @test frameindex(Ti([:a, :b]), [:b]) == [2]
    @test frameindex(Ti(3.0:-1.0:0.0; sampling = DD.Intervals(DD.Start())), [0.0, 0.99, 3.5, 4.0]) == [4, 4, 1, 1]

    oc = OffsetArray([4.5, 0.2, 2.0], -1:1)
    oi = frameindex(ti, oc)
    @test axes(oi) == axes(oc)
    @test parent(oi) == [3, 1, 2]
    vc = view([9.0, 0.5, 4.2, 9.0], 2:3)
    @test frameindex(ti, vc) == [1, 3]
    @test frameindex(ti, [0.5 2.0; 4.5 1.0]) == [1 2; 3 1]

    @test_throws "2 of 3 coordinates match no plane of the lookup; the first is 1.2" frameindex(ti, [0.5, 1.2, 3.5])
    @test_throws "1 of 1 coordinates match no plane of the lookup; the first is 0.5" frameindex(Ti([0.0, 1.0]), [0.5])
    @test_throws "the lookup repeats the value 1.0" frameindex(Ti([1.0, 1.0]), [1.0])
    @test_throws "intervals must be sorted and must not overlap, got [0.0, 2.0] before [1.0, 3.0]" frames(Ti, [0.0, 1.0], [2.0, 3.0])
    @test_throws "intervals must be sorted and must not overlap, got [2.0, 3.0] before [0.0, 1.0]" frames(Ti, [2.0, 0.0], [3.0, 1.0])
    @test_throws "each interval needs start < stop, got [1.0, 1.0]" frames(Ti, [1.0], [1.0])
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

        imgP = StructArray{StokesParams{Float64}}(; I = imgI, Q = imgQ, U = imgU, V = imgV)
        img1 = IntensityMap(imgP[:, :, 1, 1], RectiGrid((; X = x, Y = y)))
        img2 = IntensityMap(imgP, RectiGrid((x, y, t, f)))

        @test baseimage(img2) === imgP
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

# @testset "ProjectTo" begin

#     data = rand(32, 32)
#     g = spatialgrid(10.0, 10.0, 32, 32)
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
#     g = spatialgrid(10.0, 10.0, 32, 32)
#     # test_rrule(IntensityMap, data, g⊢NoTangent())
# end

# @testset "rrule baseimage" begin
#     data = rand(32, 24)
#     g = spatialgrid(5.0, 10.0, 32, 24)
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

    @testset "broadcast a map over a StructArray" begin
        sdata = StructArray{StokesParams{Float64}}(
            (
                I = rand(128), Q = rand(128), U = rand(128), V = rand(128),
            )
        )
        simg = IntensityMap(sdata, g)
        @test simg isa StokesMap{Float64, 1}
        @test baseimage(simg) === sdata
        res = simg .+ simg
        @test res isa StokesMap
        @test baseimage(res) isa StructArray
        @test baseimage(res) ≈ 2 .* sdata
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
    g = spatialgrid(10.0, 10.0, 4, 3)
    rd = (Ti(1.0),)
    img = IntensityMap(rand(4, 3), g; refdims = rd, name = :flux)
    @test DD.refdims(img) == rd
    @test DD.name(img) == :flux
    d = UnstructuredDomain((; U = randn(5), V = randn(5)))
    vis = IntensityMap(rand(5), d; refdims = rd, name = :vis)
    @test DD.refdims(vis) == rd
    @test DD.name(vis) == :vis
    @test IntensityMap(img, g) === img
    @test_throws "the domain of the IntensityMap is not the RectiGrid given" IntensityMap(img, spatialgrid(10.0, 10.0, 4, 3; posang = 0.1))
    @test_throws "the domain of the IntensityMap is not the StructuredDomain given" IntensityMap(vis, UnstructuredDomain((; U = randn(5), V = randn(5))))
end

struct PointSum <: ComradeBase.AbstractModel end
ComradeBase.visanalytic(::Type{<:PointSum}) = ComradeBase.IsAnalytic()
ComradeBase.imanalytic(::Type{<:PointSum}) = ComradeBase.IsAnalytic()
ComradeBase.ispolarized(::Type{<:PointSum}) = ComradeBase.NotPolarized()
ComradeBase.intensity_point(::PointSum, p) = p.X + 2 * p.Y

lazygrid_getindex(p, I...) = p[I...]
function lazygrid_allocated(p::AbstractArray{<:Any, N}) where {N}
    I = ntuple(_ -> 2, Val(N))
    p[I...]
    return @allocated p[I...]
end

@testset "domainpoints of a RectiGrid" begin
    x32 = X(range(-1.0f0, 1.0f0; length = 4))
    y32 = Y(range(-2.0f0, 2.0f0; length = 3))
    x64 = X(range(-1.0, 1.0; length = 4))
    y64 = Y(range(-2.0, 2.0; length = 3))
    fr = Fr([230.0e9, 345.0e9])
    cases = (
        (RectiGrid((x32, y32)), @NamedTuple{X::Float32, Y::Float32}),
        (RectiGrid((x32, y32, fr)), @NamedTuple{X::Float32, Y::Float32, Fr::Float64}),
        (RectiGrid((x32, y32, fr); posang = 0.3f0), @NamedTuple{X::Float32, Y::Float32, Fr::Float64}),
        (RectiGrid((x32, y32, fr); posang = 0.3), @NamedTuple{X::Float64, Y::Float64, Fr::Float64}),
        (RectiGrid((x32, y64)), @NamedTuple{X::Float64, Y::Float64}),
        (RectiGrid((x64, y64, Ti([1, 2]))), @NamedTuple{X::Float64, Y::Float64, Ti::Int}),
        (RectiGrid((x64, y64, Ti([1, 2]), fr); posang = 0.3), @NamedTuple{X::Float64, Y::Float64, Ti::Int, Fr::Float64}),
    )
    for (g, T) in cases
        p = @inferred domainpoints(g)
        I = ntuple(_ -> 2, ndims(g))
        @test eltype(p) === T
        @test typeof(@inferred(lazygrid_getindex(p, I...))) === T
        @test all(q -> typeof(q) === T, p)
        @test eltype(collect(p)) === T
        @test map(q -> q.X, p) == (q -> q.X).(p)
        @test eltype(map(q -> q.X, p)) === eltype((q -> q.X).(p)) === fieldtype(T, :X)
        @test lazygrid_allocated(p) == 0
        rot = ComradeBase.rotmat(g)
        xy = rot * SVector(g.X[2], g.Y[2])
        @test p[I...].X ≈ xy[1]
        @test p[I...].Y ≈ xy[2]
        ndims(g) > 2 && @test p[I...][3] === ComradeBase.basedim(dims(g)[3])[2]
    end
    p = domainpoints(RectiGrid((x32, y32, fr)))
    @test_throws BoundsError p[5, 1, 1]
    @test_throws BoundsError p[1, 1, 3]

    g32 = RectiGrid((x32, y32, fr); posang = 0.3f0)
    ref = map(q -> ComradeBase.intensity_point(PointSum(), q), domainpoints(g32)) .* prod(pixelsizes(g32))
    @test eltype(ref) === Float32
    for ex in (Serial(), ThreadsEx(), CPU())
        img = intensitymap(PointSum(), DD.rebuild(g32; executor = ex))
        @test eltype(img) === Float32
        @test baseimage(img) ≈ ref
    end
end
