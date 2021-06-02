module TestMOIWrapper

using Test
using MathOptInterface
import Cbc

const MOI = MathOptInterface

const OPTIMIZER = Cbc.Optimizer()
MOI.set(OPTIMIZER, MOI.Silent(), true)

function test_SolverName()
    @test MOI.get(OPTIMIZER, MOI.SolverName()) == "COIN Branch-and-Cut (Cbc)"
end

function test_supports_default_copy_to()
    @test !MOI.Utilities.supports_allocate_load(OPTIMIZER, false)
    @test !MOI.Utilities.supports_allocate_load(OPTIMIZER, true)
    @test !MOI.Utilities.supports_default_copy_to(OPTIMIZER, false)
    @test !MOI.Utilities.supports_default_copy_to(OPTIMIZER, true)
end

const CACHED = MOI.Utilities.CachingOptimizer(
    MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
    OPTIMIZER,
)
const BRIDGED = MOI.Bridges.full_bridge_optimizer(CACHED, Float64)

const CONFIG = MOI.Test.Config(duals = false, infeas_certificates = false)

function test_basic_constraint_tests()
    return MOI.Test.basic_constraint_tests(CACHED, CONFIG)
end

function test_Unit()
    return MOI.Test.unittest(
        BRIDGED,
        CONFIG,
        [
            # TODO(odow): implement attributes:
            "number_threads",

            # INFEASIBLE_OR_UNBOUNDED instead of DUAL_INFEASIBLE
            "solve_unbounded_model",

            # No quadratics
            "delete_soc_variables",
            "solve_qcp_edge_cases",
            "solve_qp_edge_cases",
        ],
    )
end

function test_solvername()
    @test MOI.get(CACHED, MOI.SolverName()) == "COIN Branch-and-Cut (Cbc)"
end

function test_default_objective_test()
    return MOI.Test.default_objective_test(CACHED)
end

function test_default_status_test()
    return MOI.Test.default_status_test(CACHED)
end

function test_nametest()
    return MOI.Test.nametest(CACHED)
end

function test_validtest()
    return MOI.Test.validtest(CACHED)
end

function test_emptytest()
    return MOI.Test.emptytest(BRIDGED)
end

function test_orderedindicestest()
    return MOI.Test.orderedindicestest(CACHED)
end

function test_ContinuousLinear()
    return MOI.Test.contlineartest(BRIDGED, CONFIG)
end

function test_IntegerLinear()
    return MOI.Test.intlineartest(
        BRIDGED,
        CONFIG,
        [
            # Cbc does not support indicator constraints.
            "indicator1",
            "indicator2",
            "indicator3",
            "indicator4",
            # SOS issues
            "int2",
        ],
    )
end

function test_Testparams()
    # Note: we generate a non-trivial problem to ensure that Cbc struggles to
    # find a solution at the root node.
    knapsack_model = MOI.Utilities.Model{Float64}()
    N = 100
    x = MOI.add_variables(knapsack_model, N)
    MOI.add_constraint.(knapsack_model, MOI.SingleVariable.(x), MOI.ZeroOne())
    MOI.add_constraint(
        knapsack_model,
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([1 + sin(i) for i in 1:N], x),
            0.0,
        ),
        MOI.LessThan(10.0),
    )
    MOI.set(
        knapsack_model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm.([cos(i) for i in 1:N], x),
            0.0,
        ),
    )
    model = Cbc.Optimizer(
        maxSol = 1,
        presolve = "off",
        cuts = "off",
        heur = "off",
        logLevel = 0,
    )
    MOI.copy_to(model, knapsack_model)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.SOLUTION_LIMIT
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    MOI.empty!(model)
    MOI.copy_to(model, knapsack_model)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.SOLUTION_LIMIT
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
end

function test_TestPrimalStatus()
    model = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    MOI.add_constraint(model, x, MOI.LessThan(0.0))
    cbc = Cbc.Optimizer()
    MOI.copy_to(cbc, model)
    MOI.optimize!(cbc)
    return MOI.get(cbc, MOI.PrimalStatus()) == MOI.NO_SOLUTION
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if !startswith("$name", "test_")
            continue
        end
        @testset "$(name)" begin
            getfield(@__MODULE__, name)()
        end
    end
    return
end

end

TestMOIWrapper.runtests()
