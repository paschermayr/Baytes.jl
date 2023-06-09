############################################################################################
# Import External Packages
using Test
using Random: Random, AbstractRNG, seed!
using SimpleUnPack: SimpleUnPack, @unpack, @pack!
using Distributions

############################################################################################
# Import Baytes Packages
using
    ModelWrappers,
    Baytes
#using .Baytes
############################################################################################
# Include Files
include("testhelper/TestHelper.jl")

############################################################################################
# Run Tests
@testset "All tests" begin
    include("test-construction.jl")
end
