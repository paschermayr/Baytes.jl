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
using BaytesOptim
using ForwardDiff    

#include("D:/OneDrive/1_Life/1_Git/0_Dev/Julia/modules/Baytes.jl/src/Baytes.jl")
#using .Baytes

############################################################################################
# Include Files
include("testhelper/TestHelper.jl")

############################################################################################
# Run Tests
@testset "All tests" begin
    include("test-construction.jl")
end
