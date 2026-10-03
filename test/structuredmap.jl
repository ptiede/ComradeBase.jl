using ComradeBase: StructuredDomain, Pt, coordspans, allocate_vismap, allocate_imgmap,
    IsPolarized, NotPolarized

@testset "IntensityMap over StructuredDomain" begin
    npt, nti, nfr = 7, 3, 2
    ti = [0.0, 1.0, 2.0]
    fr = [230.0e9, 345.0e9]

    @testset "(Pt,)" begin
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
        @test img[Pt(3)] == data[3]
        sub = img[2:4]
        @test sub isa IntensityMap
        @test baseimage(sub) == data[2:4]
        @test axisdims(sub).U == d.U[2:4]
        @test parent(dims(sub, Pt)) == 2:4
        v = view(img, [1, 5])
        @test v isa IntensityMap
        @test parent(v) isa SubArray
        @test baseimage(v) == data[[1, 5]]
        @test axisdims(v).V == d.V[[1, 5]]
        sel = img[Pt = DD.At([2, 6])]
        @test baseimage(sel) == data[[2, 6]]
        @test axisdims(sel).U == d.U[[2, 6]]
    end

    @testset "(Pt, Ti, Fr)" begin
        u = 1.0e6 .* randn(npt, nti)
        v = 1.0e6 .* randn(npt, nti)
        t = rand(npt, nti)
        d = StructuredDomain((Pt(npt), Ti(ti), Fr(fr)); u, v, Ti = t => (:Pt, :Ti))
        data = randn(npt, nti, nfr)
        img = IntensityMap(data, d)
        @test img isa IntensityMap{Float64, 3}
        @test keys(named_dims(img)) == (:Pt, :Ti, :Fr)
        @test_throws "IntensityMap data has size (7, 3), but the StructuredDomain has size (7, 3, 2)" IntensityMap(randn(npt, nti), d)

        b = img .^ 2
        @test b isa IntensityMap && axisdims(b) === d
        @test baseimage(b) == data .^ 2
        bw = img .* reshape(fr, 1, 1, nfr)
        @test bw isa IntensityMap && axisdims(bw) === d
        @test baseimage(bw) ≈ data .* reshape(fr, 1, 1, nfr)

        x = img[Fr = DD.At(345.0e9)]
        @test x isa IntensityMap{Float64, 2}
        @test keys(axisdims(x)) == (:Pt, :Ti)
        @test baseimage(x) == data[:, :, 2]
        @test coordspans(axisdims(x)) == (U = (:Pt, :Ti), V = (:Pt, :Ti), Ti = (:Pt, :Ti))
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
        @test coordspans(axisdims(z)) == (u = (:Pt,), v = (:Pt,), Ti = (:Pt,))
        @test axisdims(z).U ≈ d.U[:, 2, :]
        @test domainpoints(z)[3, 2].Ti == t[3, 2]

        p = img[Pt = 2:3, Fr = 1:1]
        @test size(p) == (2, nti, 1)
        @test axisdims(p).U ≈ d.U[2:3, :, 1:1]

        vw = view(img, Ti = 3)
        @test vw isa IntensityMap
        @test parent(vw) isa SubArray
        vw .= 0
        @test all(iszero, data[:, 3, :])

        q = img[Pt(2)]
        @test q isa DD.DimArray
        @test !(q isa IntensityMap)
        @test dims(q) == (dims(img, Ti), dims(img, Fr))
        @test parent(q) == data[2, :, :]

        @test_throws "cannot take dims (:Pt, :Ti, :Fr) with size (7, 1, 2)" sum(img; dims = :Ti)
        @test sum(DD.DimArray(img); dims = :Ti) isa DD.DimArray
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
        @test eltype(d32) === Float32
        dmix = StructuredDomain((U = randn(Float32, npt), V = randn(npt)))
        @test eltype(dmix) === Float64
        dm = StructuredDomain((Pt(npt), Fr(fr)); u = randn(Float32, npt), v = randn(Float32, npt))
        @test eltype(dm) === Float64
        dn = StructuredDomain((Pt(npt),); W = randn(npt))
        @test_throws "neither `U` nor `u`" eltype(dn)
        dxy = StructuredDomain((X = randn(Float32, npt), Y = randn(Float32, npt)))
        @test eltype(dxy) === Float32
        @test_throws "needs both `X` and `Y`, but `Y` is missing" eltype(StructuredDomain((X = randn(npt),)))
    end

    @testset "create_map and allocate" begin
        d = StructuredDomain((Pt(npt), Ti(ti), Fr(fr)); U = randn(npt, nti, nfr), V = randn(npt, nti, nfr))
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
            @test vp isa StokesMap{ComplexF64, 4}
            @test baseimage(vp) isa Array{ComplexF64, 4}
            @test size(vp) == (size(d)..., 4)
            @test stokes(vp, :I) isa IntensityMap{ComplexF64, 3}
            ip = allocate_imgmap(IsPolarized(), m, d)
            @test ip isa StokesMap{Float64, 4}
        end
        dm = StructuredDomain((Pt(npt), Fr(fr)); u = randn(Float32, npt), v = randn(Float32, npt))
        @test allocate_vismap(BlobTest(2.0), dm) isa IntensityMap{ComplexF64, 2}
        d32 = StructuredDomain((U = randn(Float32, npt), V = randn(Float32, npt)))
        @test allocate_vismap(GaussTest(), d32) isa IntensityMap{ComplexF32, 1}
    end
end
