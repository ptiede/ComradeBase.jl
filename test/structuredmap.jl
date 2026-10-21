using ComradeBase: StructuredDomain, Node, coordspans, allocate_vismap, allocate_imgmap,
    IsPolarized, NotPolarized

@testset "IntensityMap over StructuredDomain" begin
    npt, nti, nfr = 7, 3, 2
    ti = [0.0, 1.0, 2.0]
    fr = [230.0e9, 345.0e9]

    @testset "(Node,)" begin
        d = StructuredDomain((U = randn(npt), V = randn(npt)); header = ComradeBase.NoHeader())
        data = randn(npt)
        img = IntensityMap(data, d)
        @test img isa IntensityMap{Float64, 1}
        @test baseimage(img) === data
        @test axisdims(img) === d
        @test dims(img) === dims(d)
        @test header(img) == ComradeBase.NoHeader()
        @test executor(img) == Serial()
        @test domainpoints(img) == domainpoints(d)
        @test img.U === d.U
        @test propertynames(img) == propertynames(d)
        @test IntensityMap(img, d) === img
        @test_throws "IntensityMap data has size (8,), but the StructuredDomain has size (7,)" IntensityMap(randn(npt + 1), d)
        @test_throws DimensionMismatch IntensityMap(randn(npt, 2), d)

        b = img .* 2 .+ 1
        @test b isa IntensityMap
        @test axisdims(b) === d
        @test baseimage(b) == 2 .* data .+ 1
        b2 = img .+ data
        @test b2 isa IntensityMap
        @test baseimage(b2) == 2 .* data
        img2 = copy(img)
        img2 .= 3.0
        @test all(==(3.0), baseimage(img2))
        @test baseimage(img) == data

        s = similar(img)
        @test s isa IntensityMap && axisdims(s) === d
        s2 = similar(img, ComplexF32)
        @test s2 isa IntensityMap{ComplexF32} && axisdims(s2) === d
        c = copy(img)
        @test c isa IntensityMap && c == img && baseimage(c) !== data

        @test img[3] == data[3]
        @test img[Node(3)] == data[3]
        sub = img[2:4]
        @test sub isa IntensityMap
        @test baseimage(sub) == data[2:4]
        @test axisdims(sub).U == d.U[2:4]
        @test parent(dims(sub, Node)) == 2:4
        v = view(img, [1, 5])
        @test v isa IntensityMap
        @test parent(v) isa SubArray
        @test baseimage(v) == data[[1, 5]]
        @test axisdims(v).V == d.V[[1, 5]]
        sel = img[Node = DD.At([2, 6])]
        @test baseimage(sel) == data[[2, 6]]
        @test axisdims(sel).U == d.U[[2, 6]]
    end

    @testset "(Node, Ti, Fr)" begin
        u = 1.0e6 .* randn(npt, nti)
        v = 1.0e6 .* randn(npt, nti)
        t = rand(npt, nti)
        d = StructuredDomain((Node(npt), Ti(ti), Fr(fr)); u, v, Ti = t => (:Node, :Ti))
        data = randn(npt, nti, nfr)
        img = IntensityMap(data, d)
        @test img isa IntensityMap{Float64, 3}
        @test keys(named_dims(img)) == (:Node, :Ti, :Fr)
        @test_throws "IntensityMap data has size (7, 3), but the StructuredDomain has size (7, 3, 2)" IntensityMap(randn(npt, nti), d)

        b = img .^ 2
        @test b isa IntensityMap && axisdims(b) === d
        @test baseimage(b) == data .^ 2
        bw = img .* reshape(fr, 1, 1, nfr)
        @test bw isa IntensityMap && axisdims(bw) === d
        @test baseimage(bw) ≈ data .* reshape(fr, 1, 1, nfr)

        x = img[Fr = DD.At(345.0e9)]
        @test x isa IntensityMap{Float64, 2}
        @test keys(axisdims(x)) == (:Node, :Ti)
        @test baseimage(x) == data[:, :, 2]
        @test coordspans(axisdims(x)) == (U = (:Node, :Ti), V = (:Node, :Ti), Ti = (:Node, :Ti))
        @test axisdims(x).U ≈ d.U[:, :, 2]
        @test axisdims(x).V ≈ d.V[:, :, 2]
        @test only(DD.refdims(x)) == Fr([345.0e9])

        y = img[Ti = 1:2]
        @test y isa IntensityMap{Float64, 3}
        @test size(y) == (npt, 2, nfr)
        @test baseimage(y) == data[:, 1:2, :]
        @test axisdims(y).Ti == t[:, 1:2]
        @test axisdims(y).U ≈ d.U[:, 1:2, :]

        z = img[Ti = DD.At(1.0)]
        @test size(z) == (npt, nfr)
        @test coordspans(axisdims(z)) == (u = (:Node,), v = (:Node,), Ti = (:Node,))
        @test axisdims(z).U ≈ d.U[:, 2, :]
        @test domainpoints(z)[3, 2].Ti == t[3, 2]

        p = img[Node = 2:3, Fr = 1:1]
        @test size(p) == (2, nti, 1)
        @test axisdims(p).U ≈ d.U[2:3, :, 1:1]

        vw = view(img, Ti = 3)
        @test vw isa IntensityMap
        @test parent(vw) isa SubArray
        vw .= 0
        @test all(iszero, data[:, 3, :])

        q = img[Node(2)]
        @test q isa DD.DimArray
        @test !(q isa IntensityMap)
        @test dims(q) == (dims(img, Ti), dims(img, Fr))
        @test parent(q) == data[2, :, :]

        @test_throws "cannot take dims (:Node, :Ti, :Fr) with size (7, 1, 2)" sum(img; dims = :Ti)
        @test sum(DD.DimArray(img); dims = :Ti) isa DD.DimArray
    end

    @testset "slicing u, v, w in meters" begin
        u = 1.0e6 .* randn(npt)
        fobs = fr' .+ 1.0e6 .* (1:npt)
        d = StructuredDomain((Node(npt), Fr(1:nfr)); u, v = u, Fr = fobs)
        x = IntensityMap(randn(npt, nfr), d)[Fr = 2]
        @test coordspans(axisdims(x)) == (u = (:Node,), v = (:Node,), Fr = (:Node,))
        @test axisdims(x).U ≈ d.U[:, 2]

        dw = StructuredDomain((Node(npt), Fr(fr)); u, v = u, w = u)
        xw = IntensityMap(randn(npt, nfr), dw)[Fr = 1]
        @test coordspans(axisdims(xw)) == (U = (:Node,), V = (:Node,), W = (:Node,))
        @test axisdims(xw).W ≈ dw.W[:, 1]
    end

    @testset "rectilinear-only methods" begin
        img = IntensityMap(rand(npt), StructuredDomain((U = randn(npt), V = randn(npt))))
        @test_throws MethodError pixelsizes(img)
        @test_throws MethodError fieldofview(img)
        @test_throws MethodError flux(img)
        @test_throws MethodError centroid(img)
        @test_throws MethodError second_moment(img)
        @test_throws MethodError phasecenter(img)
    end

    @testset "eltype" begin
        d32 = StructuredDomain((U = randn(Float32, npt), V = randn(Float32, npt)))
        @test @inferred(eltype(d32)) === Float32
        dmix = StructuredDomain((U = randn(Float32, npt), V = randn(npt)))
        @test @inferred(eltype(dmix)) === Float64
        dm = StructuredDomain((Node(npt), Fr(fr)); u = randn(Float32, npt), v = randn(Float32, npt))
        @test @inferred(eltype(dm)) === Float64
        dfc = StructuredDomain((; u = randn(Float32, npt), v = randn(Float32, npt), Fr = rand(fr, npt)))
        @test @inferred(eltype(dfc)) === Float64
        dn = StructuredDomain((Node(npt),); W = randn(npt))
        @test_throws "neither `U` nor `u`" eltype(dn)
        dxy = StructuredDomain((X = randn(Float32, npt), Y = randn(Float32, npt)))
        @test @inferred(eltype(dxy)) === Float32
        @test_throws "needs both `X` and `Y`, but `Y` is missing" eltype(StructuredDomain((X = randn(npt),)))
    end

    @testset "create_map and allocate" begin
        d = StructuredDomain((Node(npt), Ti(ti), Fr(fr)); U = randn(npt, nti, nfr), V = randn(npt, nti, nfr))
        data = randn(npt, nti, nfr)
        @test ComradeBase.create_map(data, d) == IntensityMap(data, d)
        @test ComradeBase.create_vismap(data, d) isa IntensityMap
        a = ComradeBase.allocate_map(Array{Float32}, d)
        @test a isa IntensityMap{Float32, 3}
        @test axisdims(a) === d
        for m in (BlobTest(2.0), GaussTest())
            vm = allocate_vismap(m, d)
            @test vm isa IntensityMap{ComplexF64, 3}
            @test size(vm) == size(d)
            @test axisdims(vm) === d
            im = allocate_imgmap(m, d)
            @test im isa IntensityMap{Float64, 3}
            vp = allocate_vismap(IsPolarized(), m, d)
            @test vp isa StokesMap{ComplexF64, 3}
            @test baseimage(vp) isa FieldDimArray{StokesParams{ComplexF64}, 3, Array{ComplexF64, 4}}
            @test size(vp) == size(d)
            @test size(parent(baseimage(vp))) == (size(d)..., 4)
            @test stokes(vp, :I) isa IntensityMap{ComplexF64, 3}
            ip = allocate_imgmap(IsPolarized(), m, d)
            @test ip isa StokesMap{Float64, 3}
        end
        dm = StructuredDomain((Node(npt), Fr(fr)); u = randn(Float32, npt), v = randn(Float32, npt))
        @test allocate_vismap(BlobTest(2.0), dm) isa IntensityMap{ComplexF64, 2}
        d32 = StructuredDomain((U = randn(Float32, npt), V = randn(Float32, npt)))
        @test allocate_vismap(GaussTest(), d32) isa IntensityMap{ComplexF32, 1}
    end
end

@testset "Blocked visibility domains" begin
    ti = frames(Ti, [0.0, 1.0, 2.0, 3.0], [0.9, 1.9, 2.9, 3.9])
    t = [0.1, 0.2, 0.3, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 3.5]
    n = length(t)
    fr = Fr([230.0e9, 345.0e9])
    u = 1.0e6 .* randn(n)
    vis = StructuredDomain((Node(n), fr); u, v = 1.0e6 .* randn(n), Ti = t, ant = collect(1:n))
    lookupof(d) = parent(DD.lookup(first(dims(d))))

    @testset "uneven blocks" begin
        b = ComradeBase._blockdomain(vis, (ti,), (2,))
        @test lookupof(b) == [1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 9, 10]
        @test b.valid == (lookupof(b) .!= 0)
        @test coordspans(b).valid == (:Node,)
        @test ComradeBase.coords(b).u == [u[1:4]; 0.0; 0.0; u[5:10]]
        @test all(iszero, b.U[5:6, :])
        @test b.Ti[5:6] == [0.45, 0.45]
        @test frameindex(ti, b.Ti) == [1, 1, 1, 2, 1, 1, 3, 3, 3, 3, 3, 4]
        @test b.ant == [1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10]
        @test dims(b)[2] == fr
        @test b.U[lookupof(b) .!= 0, :] == vis.U
    end

    @testset "an empty block and an existing valid mask" begin
        late = StructuredDomain((Node(n), fr); u, v = u, Ti = t .+ 2.0 .* (t .< 2), valid = (1:n) .!= 3)
        b = ComradeBase._blockdomain(late, (frames(Ti, 0:6.0),), (3,))
        @test lookupof(b) == [zeros(Int, 10); 1:10; zeros(Int, 10)]
        @test b.valid == [falses(10); (1:n) .!= 3; falses(10)]
        @test b.Ti[[1:10; 21:30]] == [fill(0.5, 10); fill(4.5, 10)]
        perfr = StructuredDomain((Node(n), fr); u, v = u, Ti = t, valid = [true, false] => (:Fr,))
        bf = ComradeBase._blockdomain(perfr, (ti,), (2,))
        @test coordspans(bf).valid == (:Node, :Fr)
        @test bf.valid == (lookupof(bf) .!= 0) .& [true false]
    end

    @testset "padding stays in its block's first plane" begin
        tend = Ti(DD.Lookups.Sampled([1.0, 2.0]; span = DD.Lookups.Regular(1.0), sampling = DD.Lookups.Intervals(DD.Lookups.End()), order = DD.Lookups.ForwardOrdered()))
        b = ComradeBase._blockdomain(StructuredDomain((; U = randn(3), V = randn(3), Ti = [0.2, 1.5, 1.7])), (tend,), (2,))
        @test lookupof(b) == [1, 0, 2, 3]
        @test b.Ti[2] == 0.5
        @test frameindex(tend, b.Ti) == [1, 1, 2, 2]
    end

    @testset "two split plane dims" begin
        f = [230.0e9, 345.0e9, 230.0e9, 230.0e9, 345.0e9, 345.0e9, 230.0e9, 345.0e9, 230.0e9, 345.0e9]
        dfr = StructuredDomain((Node(n),); U = randn(n), V = randn(n), Ti = t, Fr = f)
        @test_throws "sort the data by `Ti` and then `Fr`" ComradeBase._blockdomain(dfr, (ti, fr), (2, 2))
        p = sortperm(collect(zip(cld.(frameindex(ti, t), 2), f)))
        ds = StructuredDomain((Node(n),); U = randn(n), V = randn(n), Ti = t[p], Fr = f[p])
        b = ComradeBase._blockdomain(ds, (ti, fr), (2, 2))
        @test lookupof(b) == [1, 2, 3, 0, 4, 0, 0, 0, 5, 6, 0, 0, 7, 8, 9, 10]
        @test b.Fr[[4, 6, 11]] == [230.0e9, 345.0e9, 230.0e9]
        @test b.Ti[[4, 6, 11]] == [0.45, 0.45, 2.45]
    end

    @testset "errors" begin
        @test_throws "sort the data by `Ti`" ComradeBase._blockdomain(axisdims(IntensityMap(zeros(n, 2), vis)[Node(n:-1:1)]), (ti,), (2,))
        @test_throws "no `Ti` coordinate" ComradeBase._blockdomain(StructuredDomain((; U = u, V = u)), (ti,), (2,))
        dnf = StructuredDomain((Node(n), fr); U = randn(n, 2), V = randn(n, 2), Ti = repeat(t, 1, 2))
        @test_throws "coordinate must span only `Node`, got (:Node, :Fr)" ComradeBase._blockdomain(dnf, (ti,), (2,))
        @test_throws "4 planes, which do not split evenly over its 3 shards" ComradeBase._blockdomain(vis, (ti,), (3,))
        @test_throws "match no plane" ComradeBase._blockdomain(StructuredDomain((; U = u, V = u, Ti = t .+ 10)), (ti,), (2,))
        @test_throws "`Node` lookup contains 0" ComradeBase._blockdomain(StructuredDomain((Node(0:(n - 1)),); U = u, V = u, Ti = t), (ti,), (2,))
    end

    @testset "carrying maps over and back" begin
        b = ComradeBase._blockdomain(vis, (ti,), (2,))
        x = IntensityMap(randn(ComplexF64, n, 2), vis)
        y = IntensityMap(ComradeBase._blockvalues(x, b, 0.0im), b)
        @test baseimage(y)[5:6, :] == zeros(2, 2)
        back = y[Node(DD.Where(!=(0)))]
        @test baseimage(back) == baseimage(x)
        @test ComradeBase.coords(axisdims(back)).u == u
        @test axisdims(back).Ti == t
        @test all(axisdims(back).valid)

        c = IntensityMap(FieldDimArray{SMatrix{2, 2}}(rand(ComplexF64, n, 2, 2, 2)), vis)
        yc = IntensityMap(ComradeBase._blockvalues(c, b, zero(eltype(c))), b)
        @test yc isa CoherencyMap
        @test parent(baseimage(yc))[[1:4; 7:12], :, :, :] == parent(baseimage(c))
        @test iszero(parent(baseimage(yc))[5:6, :, :, :])
        @test parent(baseimage(yc[Node(DD.Where(!=(0)))])) == parent(baseimage(c))
        @test y[Node(DD.At(5))] == x[Node(5)]

        @test_throws "Node` lookup must equal" ComradeBase._blockvalues(x[Node(1:5)], b, 0.0im)
        @test_throws DimensionMismatch ComradeBase._blockvalues(IntensityMap(randn(n), StructuredDomain((; U = u, V = u))), b, 0.0)
    end
end
