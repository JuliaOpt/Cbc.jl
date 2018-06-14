using Base.Test, MathOptInterface, MathOptInterface.Test, MathOptInterface.Utilities

const MOI  = MathOptInterface
const MOIT = MathOptInterface.Test
const MOIU = MathOptInterface.Utilities

# @testset "Cont Linear tests" begin
#     linconfig = MOIT.TestConfig(modify_lhs = false, duals = false)
#     @testset "Default Solver"  begin
#         solver = CbcOptimizer(LogLevel = 0)
#         MOIT.contlineartest(solver, linconfig, ["linear10","linear12","linear8a","linear8b","linear8c"])
#     end
# end

@MOIU.model ModelForCachingOptimizer (ZeroOne, Integer) (EqualTo, GreaterThan, LessThan, Interval) (Zeros, Nonnegatives, Nonpositives, SecondOrderCone, RotatedSecondOrderCone, GeometricMeanCone, ExponentialCone, DualExponentialCone, PositiveSemidefiniteConeTriangle, RootDetConeTriangle, LogDetConeTriangle) () (SingleVariable,) (ScalarAffineFunction,ScalarQuadraticFunction) (VectorOfVariables,) (VectorAffineFunction,)

@testset "CachingOptimizer Manual mode" begin
    m = MOIU.CachingOptimizer(ModelForCachingOptimizer{Float64}(), MOIU.Manual)

    v = MOI.addvariable!(m)

    @test MOI.canaddconstraint(m, MOI.SingleVariable, MOI.GreaterThan{Float64})
    vc = MOI.addconstraint!(m, MOI.SingleVariable(v), MOI.EqualTo(5.0))

    saf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([3.0], [v]), 0.0)
    @test MOI.canset(m, MOI.ObjectiveFunction{typeof(saf)}())
    MOI.set!(m, MOI.ObjectiveFunction{typeof(saf)}(), saf)
    @test MOI.get(m, MOIU.AttributeFromModelCache(MOI.ObjectiveFunction{typeof(saf)}())) ≈ saf
    @test MOI.get(m, MOI.ObjectiveFunction{typeof(saf)}()) ≈ saf

    solver = CbcOptimizer()

    MOI.copy!(solver, m)
    MOI.optimize!(solver)

end