using ComradeBase: StructuredDomain, Pt, shapedcoords, coordspans, speed_of_light

broadcast_points(sc) = broadcast((x...) -> NamedTuple{keys(sc)}(x), values(sc)...)
function points_approx(a, b)
    size(a) == size(b) || return false
    return all(zip(a, b)) do (x, y)
        keys(x) == keys(y) && all(map(isapprox, values(x), values(y)))
    end
end
function allocated_getindex(p, i)
    p[i]
    return @allocated p[i]
end

@testset "StructuredDomain" begin
    npt, nti, nfr = 7, 3, 2
    ti = [0.0, 1.0, 2.0]
    fr = [230.0e9, 345.0e9]

    @testset "(Pt,)" begin
        nt = (U = randn(npt), V = randn(npt), Ti = rand(npt), Fr = fill(230.0e9, npt))
        d = StructuredDomain(nt)
        @test size(d) == (npt,)
        @test keys(d) == (:Pt,)
        @test ndims(d) == 1
        @test length(d) == npt
        @test axes(d) == (Base.OneTo(npt),)
        @test keys(named_dims(d)) == (:Pt,)
        @test coordspans(d) == (U = (:Pt,), V = (:Pt,), Ti = (:Pt,), Fr = (:Pt,))
        @test d.U === nt.U
        @test propertynames(d) == (:U, :V, :Ti, :Fr)
        @test_throws "no property `W`" d.W
        @test executor(d) == Serial()
        @test header(d) == ComradeBase.NoHeader()

        p = domainpoints(d)
        @test size(p) == (npt,)
        @test eltype(p) == NamedTuple{(:U, :V, :Ti, :Fr), NTuple{4, Float64}}
        for i in eachindex(nt.U)
            @test p[i] == map(x -> x[i], nt)
        end
        @test allocated_getindex(p, 3) == 0
        sc = shapedcoords(d)
        @test broadcast_points(sc) == collect(p)

        d2 = StructuredDomain((Pt(npt),); U = nt.U, V = nt.V)
        @test domainpoints(d2) == domainpoints(StructuredDomain((; U = nt.U, V = nt.V)))
        @test DD.rebuild(d; executor = ThreadsEx()) isa StructuredDomain
        @test executor(DD.rebuild(d; executor = ThreadsEx())) == ThreadsEx()

        s = sprint(show, MIME"text/plain"(), d)
        @test occursin("StructuredDomain", s)
        @test occursin("spanning (:Pt,)", s)
        @test occursin("StructuredDomain with dims (:Pt,)", summary(d))
    end

    @testset "(Pt, Fr) with u, v in meters" begin
        u = 1.0e6 .* randn(npt)
        v = 1.0e6 .* randn(npt)
        d = StructuredDomain((Pt(npt), Fr(fr)); u, v)
        @test size(d) == (npt, nfr)
        @test keys(d) == (:Pt, :Fr)
        @test coordspans(d) == (u = (:Pt,), v = (:Pt,))
        @test propertynames(d) == (:U, :V, :Fr)
        @test d.U ≈ u .* fr' ./ speed_of_light
        @test d.Fr == fr
        p = domainpoints(d)
        for I in ((1, 1), (3, 2), (npt, 1))
            @test p[I...].U ≈ u[I[1]] * fr[I[2]] / speed_of_light
            @test p[I...].V ≈ v[I[1]] * fr[I[2]] / speed_of_light
            @test p[I...].Fr == fr[I[2]]
        end
        @test keys(p[1]) == (:U, :V, :Fr)
        sc = shapedcoords(d)
        @test size(sc.U) == (npt, nfr)
        @test size(sc.Fr) == (1, nfr)
        @test points_approx(broadcast_points(sc), collect(p))
        @test sprint(show, MIME"text/plain"(), d) isa String
    end

    @testset "(Pt, Ti, Fr)" begin
        u = 1.0e6 .* randn(npt, nti)
        v = 1.0e6 .* randn(npt, nti)
        tobs = ti' .+ 0.01 .* rand(npt, nti)
        a1 = rand(1:5, npt)
        d = StructuredDomain((Pt(npt), Ti(ti), Fr(fr)); u, v, Ti = tobs, antenna1 = a1)
        @test size(d) == (npt, nti, nfr)
        @test keys(d) == (:Pt, :Ti, :Fr)
        @test keys(named_dims(d)) == (:Pt, :Ti, :Fr)
        @test coordspans(d) ==
            (u = (:Pt, :Ti), v = (:Pt, :Ti), Ti = (:Pt, :Ti), antenna1 = (:Pt,))
        @test d.Ti === tobs
        @test d.antenna1 === a1
        @test size(d.U) == (npt, nti, nfr)
        p = domainpoints(d)
        @test size(p) == size(d)
        @test (@inferred p[2, 3, 2]) isa NamedTuple
        @test keys(p[1]) == (:U, :V, :Ti, :antenna1, :Fr)
        for I in ((1, 1, 1), (2, 3, 2), (npt, 2, 1))
            i, t, f = I
            @test p[I...].U ≈ u[i, t] * fr[f] / speed_of_light
            @test p[I...].V ≈ v[i, t] * fr[f] / speed_of_light
            @test p[I...].Ti == tobs[i, t]
            @test p[I...].Fr == fr[f]
            @test p[I...].antenna1 == a1[i]
        end
        sc = shapedcoords(d)
        @test size(sc.antenna1) == (npt, 1, 1)
        @test size(sc.Ti) == (npt, nti, 1)
        @test points_approx(broadcast_points(sc), collect(p))

        dl = StructuredDomain((Pt(npt), Ti(ti), Fr(fr)); U = randn(npt, nti, nfr), V = randn(npt, nti, nfr))
        @test domainpoints(dl)[2, 3, 1].Ti == ti[3]

        de = StructuredDomain(
            (Pt(nti), Ti(ti), Fr(fr));
            U = randn(nti, nti) => (:Pt, :Ti), V = randn(nti, nti) => (:Pt, :Ti),
            valid = trues(nti) => :Pt
        )
        @test coordspans(de) == (U = (:Pt, :Ti), V = (:Pt, :Ti), valid = (:Pt,))
        @test domainpoints(de)[1, 2, 2].valid
        s = sprint(show, MIME"text/plain"(), d)
        @test occursin("spanning (:Pt, :Ti)", s)
    end

    @testset "validation" begin
        u = randn(npt)
        @test_throws "first dim of a StructuredDomain must be `Pt`" StructuredDomain((Fr(fr), Pt(npt)); U = randn(nfr, npt))
        @test_throws "must be DimensionalData dimensions" StructuredDomain((Pt(npt), fr); U = u)
        @test_throws "needs dims starting with `Pt`" StructuredDomain(())
        @test_throws "needs a vector lookup" StructuredDomain((Pt(npt), Fr(2.0)); U = u)
        @test_throws "dim names must be distinct" StructuredDomain((Pt(npt), Fr(fr), Fr(fr)); U = u)
        @test_throws DimensionMismatch StructuredDomain((Pt(npt),); U = randn(npt + 1))
        @test_throws "matches no ordered subset" StructuredDomain((Pt(npt),); U = randn(npt + 1))
        @test_throws "is ambiguous" StructuredDomain((Pt(2), Fr(fr)); U = randn(2))
        @test_throws "`U = U => (:Pt,)`" StructuredDomain((Pt(2), Fr(fr)); U = randn(2))
        @test_throws "not a dim of this domain" StructuredDomain((Pt(npt),); U = u => (:Ti,))
        @test_throws "in the domain's dim order" StructuredDomain((Pt(npt), Ti(ti)); U = randn(nti, npt) => (:Ti, :Pt))
        @test_throws "in the domain's dim order" StructuredDomain((Pt(npt), Ti(ti)); U = randn(npt, npt) => (:Pt, :Pt))
        @test_throws DimensionMismatch StructuredDomain((Pt(npt), Ti(ti)); U = randn(npt, 2) => (:Pt, :Ti))
        @test_throws "the dims (:Pt, :Ti) it spans have axes" StructuredDomain((Pt(npt), Ti(ti)); U = randn(npt, 2) => (:Pt, :Ti))
        @test_throws "must be a dim name or a tuple of dim names" StructuredDomain((Pt(npt),); U = u => 1)
        @test_throws "must be an array or `array => (dim names...)`" StructuredDomain((Pt(npt),); U = 1.0)
        @test_throws "give either `U` in wavelengths or `u` in meters" StructuredDomain((Pt(npt), Fr(fr)); U = u, u = u)
        @test_throws "give either `V` in wavelengths or `v` in meters" StructuredDomain((Pt(npt), Fr(fr)); V = u, v = u)
        @test_throws "u and v in meters need a frequency axis" StructuredDomain((Pt(npt),); u, v = u)
        @test_throws "u and v in meters need a frequency axis" StructuredDomain((; u, v = u))
        @test_throws "`valid` must have a Bool element type" StructuredDomain((Pt(npt),); U = u, valid = ones(npt))
        @test_throws "needs at least one coordinate" StructuredDomain(NamedTuple())
        d = StructuredDomain((Pt(npt), Ti(ti)); U = randn(npt, nti))
        @test_throws DimensionMismatch DD.rebuild(d; dims = (Pt(npt + 1), Ti(ti)))
    end

    @testset "generic indexing: view coords" begin
        ubig = 1.0e6 .* randn(npt + 2, nti + 1)
        vbig = 1.0e6 .* randn(npt + 2, nti + 1)
        uv = view(ubig, 2:(npt + 1), 1:nti)
        vv = view(vbig, 2:(npt + 1), 1:nti)
        dims = (Pt(npt), Ti(ti), Fr(fr))
        dview = StructuredDomain(dims; u = uv, v = vv)
        dcopy = StructuredDomain(dims; u = copy(uv), v = copy(vv))
        @test collect(domainpoints(dview)) == collect(domainpoints(dcopy))
        @test dview.U == dcopy.U
        scv = shapedcoords(dview)
        @test broadcast_points(scv) == collect(domainpoints(dcopy))

        p1 = view(randn(2npt), 1:2:(2npt))
        dp = StructuredDomain((; U = p1, V = p1))
        @test collect(domainpoints(dp)) == collect(domainpoints(StructuredDomain((; U = collect(p1), V = collect(p1)))))
    end
end
