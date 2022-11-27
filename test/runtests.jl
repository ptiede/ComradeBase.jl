using ComradeBase, StaticArrays, JET
using Test

@testset "ComradeBase.jl" begin
    include(joinpath(@__DIR__, "polarizedtypes.jl"))
    include(joinpath(@__DIR__, "images.jl"))
end
