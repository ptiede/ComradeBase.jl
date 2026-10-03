function plane_coords(N, planes)
    U_vals = range(-10.0e9, 10.0e9; length = N)
    U_final = vec(U_vals' .* ones(N))
    V_final = vec(ones(N)' .* U_vals)
    sz = (length(U_final), map(length, planes)...)
    U = similar(U_final, sz)
    V = similar(V_final, sz)
    for I in CartesianIndices(Base.tail(sz))
        U[:, I] .= U_final .* sum(Tuple(I))
        V[:, I] .= V_final .+ prod(Tuple(I))
    end
    return U, V
end

# Selecting a plane by its Ti and Fr values gives the (Pt,) domain of that plane.
function planes_match(N, planes)
    U, V = plane_coords(N, planes)
    npt = size(U, 1)
    d = ComradeBase.StructuredDomain((ComradeBase.Pt(npt), planes...); U, V)
    img = IntensityMap(zeros(size(d)), d)
    C = true
    for I in CartesianIndices(map(length, planes))
        sel = map((p, i) -> DD.rebuild(p, DD.At(parent(p)[i])), planes, Tuple(I))
        for order in (sel, reverse(sel))
            sub = img[order...]
            g = UnstructuredDomain((U = U[:, I], V = V[:, I]))
            C = C && (domainpoints(axisdims(sub)) == domainpoints(g))
        end
    end
    return C
end

@testset "Test plane selection for visdomain" begin
    ti = Ti(sort(10 * rand(10)))
    fr = Fr(sort(1.0e11 * rand(4)))
    @test planes_match(8, (ti, fr))
    @test planes_match(8, (ti,))
    @test planes_match(8, (fr,))
end
