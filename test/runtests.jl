using ComradeBase, StaticArrays, JET
using Pyehtim
using Test

@testset "ComradeBase.jl" begin
    include(joinpath(@__DIR__, "polarizedtypes.jl"))
    include(joinpath(@__DIR__, "images.jl"))
    include(joinpath(@__DIR__, "io.jl"))
end
