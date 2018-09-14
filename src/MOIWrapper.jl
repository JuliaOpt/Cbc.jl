export CbcOptimizer


using Compat.SparseArrays
using MathOptInterface
using Cbc.CbcCInterface

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const CbcCI = CbcCInterface

mutable struct CbcOptimizer <: MOI.AbstractOptimizer
    inner::CbcModel
    CbcOptimizer() = new(CbcModel()) # Initializes with an empty model
end

struct CbcModelFormat
    num_rows::Int
    num_cols::Int
    row_idx::Vector{Int}
    col_idx::Vector{Int}
    values::Vector{Float64}
    col_lb::Vector{Float64}
    col_ub::Vector{Float64}
    obj::Vector{Float64}
    row_lb::Vector{Float64}
    row_ub::Vector{Float64}
    function CbcModelFormat(num_rows::Int, num_cols::Int)
        obj = fill(0.0, num_cols)
        row_idx = Int[]
        col_idx = Int[]
        values = Float64[]
        col_lb = fill(-Inf, num_cols)
        col_ub = fill(Inf, num_cols)
        row_lb = fill(-Inf, num_rows)
        row_ub = fill(Inf, num_rows)
        constraint_matrix = Tuple{Int,Int,Float64}[]
        new(num_rows, num_cols, row_idx, col_idx, values, col_lb, col_ub, obj, row_lb, row_ub)
    end
end


function load_constraint(ci::MOI.ConstraintIndex, cbc_model_format::CbcModelFormat, mapping::MOIU.IndexMap,
    f::MOI.SingleVariable, s::MOI.EqualTo)
    cbc_model_format.col_lb[mapping.varmap[f.variable].value] = s.value
    cbc_model_format.col_ub[mapping.varmap[f.variable].value] = s.value
end

function load_constraint(ci::MOI.ConstraintIndex, cbc_model_format::CbcModelFormat, mapping::MOIU.IndexMap,
    f::MOI.SingleVariable, s::MOI.LessThan)
    cbc_model_format.col_ub[mapping.varmap[f.variable].value] = s.upper
end

function load_constraint(ci::MOI.ConstraintIndex, cbc_model_format::CbcModelFormat, mapping::MOIU.IndexMap,
    f::MOI.SingleVariable, s::MOI.GreaterThan)
    cbc_model_format.col_lb[mapping.varmap[f.variable].value] = s.lower
end

function load_constraint(ci::MOI.ConstraintIndex, cbc_model_format::CbcModelFormat, mapping::MOIU.IndexMap,
    f::MOI.SingleVariable, s::MOI.Interval)
    cbc_model_format.col_lb[mapping.varmap[f.variable].value] = s.lower
    cbc_model_format.col_ub[mapping.varmap[f.variable].value] = s.upper
end

function push_terms(row_idx::Vector{Int}, col_idx::Vector{Int}, values::Vector{Float64},
    ci::MOI.ConstraintIndex{F,S}, terms::Vector{MOI.ScalarAffineTerm{Float64}},
    mapping::MOIU.IndexMap) where {F,S}
    for term in terms
        push!(row_idx, mapping.conmap[ci].value)
        push!(col_idx, mapping.varmap[term.variable_index].value)
        push!(values, term.coefficient)
    end
end

function load_constraint(ci::MOI.ConstraintIndex, cbc_model_format::CbcModelFormat,
    mapping::MOIU.IndexMap, f::MOI.ScalarAffineFunction, s::MOI.EqualTo)
    push_terms(cbc_model_format.row_idx, cbc_model_format.col_idx,
               cbc_model_format.values, ci, f.terms, mapping)
    cbc_model_format.row_lb[mapping.conmap[ci].value] = s.value - f.constant
    cbc_model_format.row_ub[mapping.conmap[ci].value] = s.value - f.constant
end

function load_constraint(ci::MOI.ConstraintIndex, cbc_model_format::CbcModelFormat,
    mapping::MOIU.IndexMap, f::MOI.ScalarAffineFunction, s::MOI.GreaterThan)
    push_terms(cbc_model_format.row_idx, cbc_model_format.col_idx,
               cbc_model_format.values, ci, f.terms, mapping)
    cbc_model_format.row_lb[mapping.conmap[ci].value] = s.lower - f.constant
end

function load_constraint(ci::MOI.ConstraintIndex, cbc_model_format::CbcModelFormat, mapping::MOIU.IndexMap,
    f::MOI.ScalarAffineFunction, s::MOI.LessThan)
    push_terms(cbc_model_format.row_idx, cbc_model_format.col_idx,
               cbc_model_format.values, ci, f.terms, mapping)
    cbc_model_format.row_ub[mapping.conmap[ci].value] = s.upper - f.constant
end

function load_constraint(ci::MOI.ConstraintIndex, cbc_model_format::CbcModelFormat, mapping::MOIU.IndexMap,
    f::MOI.ScalarAffineFunction, s::MOI.Interval)
    push_terms(cbc_model_format.row_idx, cbc_model_format.col_idx,
               cbc_model_format.values, ci, f.terms, mapping)
    cbc_model_format.row_ub[mapping.conmap[ci].value] = s.upper - f.constant
    cbc_model_format.row_lb[mapping.conmap[ci].value] = s.lower - f.constant
end


function load_obj(cbc_model_format::CbcModelFormat, mapping::MOIU.IndexMap,
    f::MOI.ScalarAffineFunction)
    # We need to increment values of objective function with += to handle cases like $x_1 + x_2 + x_1$
    # This is safe becasue objective function is initialized with zeros in the constructor
    for term in f.terms
        cbc_model_format.obj[mapping.varmap[term.variable_index].value] += term.coefficient
    end
end

function copy_constraints!(cbc_model_format::CbcModelFormat, user_optimizer::MOI.ModelLike,
    mapping::MOIU.IndexMap)
    for (F,S) in MOI.get(user_optimizer, MOI.ListOfConstraints())
        if !(S <: Union{MOI.ZeroOne, MOI.Integer})
            for ci in MOI.get(user_optimizer, MOI.ListOfConstraintIndices{F,S}())
                f = MOI.get(user_optimizer, MOI.ConstraintFunction(), ci)
                s = MOI.get(user_optimizer,  MOI.ConstraintSet(), ci)
                load_constraint(ci, cbc_model_format, mapping, f, s)
            end
        end
    end
end

function update_bounds_for_binary_vars!(col_lb::Vector{Float64}, col_ub::Vector{Float64}, zero_one_indices::Vector{Int})
    for idx in zero_one_indices
        if col_lb[idx] < 0.0
            col_lb[idx] = 0.0
        end
        if col_ub[idx] > 1.0
            col_ub[idx] = 1.0
        end
    end
end

function update_zeroone_indices(user_optimizer::MOI.ModelLike, mapping::MOIU.IndexMap,
    ci::Vector{MOI.ConstraintIndex{F,S}}, zero_one_indices::Vector{Int}) where {F, S}
    for i in 1:length(ci)
        f = MOI.get(user_optimizer, MOI.ConstraintFunction(), ci[i])
        push!(zero_one_indices, mapping.varmap[f.variable].value)
    end
end

function update_integer_indices(user_optimizer::MOI.ModelLike, mapping::MOIU.IndexMap,
    ci::Vector{MOI.ConstraintIndex{F,S}}, integer_indices::Vector{Int}) where {F, S}
    for i in 1:length(ci)
        f = MOI.get(user_optimizer, MOI.ConstraintFunction(), ci[i])
        push!(integer_indices, mapping.varmap[f.variable].value)
    end
end

"""
    function copy_to(cbc_optimizer, user_optimizer; copy_names=false)

Receive a cbc_optimizer which contains the pointer to the cbc C object and instantiate the object cbc_model_format::CbcModelFormat based on user_optimizer::AbstractModel (also provided by the user).
Function loadProblem of CbcCInterface requires all information stored in cbc_model_format.
"""
function MOI.copy_to(cbc_optimizer::CbcOptimizer,
    user_optimizer::MOI.ModelLike; copy_names=false)

    mapping = MOIU.IndexMap()

    num_cols = MOI.get(user_optimizer, MOI.NumberOfVariables())
    var_index = MOI.get(user_optimizer, MOI.ListOfVariableIndices())
    for i in 1:num_cols
        mapping.varmap[var_index[i]] = MOI.VariableIndex(i)
    end

    zero_one_indices = Int[]
    integer_indices = Int[]
    list_of_constraints = MOI.get(user_optimizer, MOI.ListOfConstraints())
    num_rows = 0
    for (F,S) in list_of_constraints
        if !(MOI.supports_constraint(cbc_optimizer, F, S))
            throw(MOI.UnsupportedConstraint{F,S}("Cbc MOI Interface does not support constraints of type " * (F,S) * "."))
        end

        ci = MOI.get(user_optimizer, MOI.ListOfConstraintIndices{F,S}())

        if F == MOI.SingleVariable
            if S == MOI.ZeroOne
                update_zeroone_indices(user_optimizer, mapping, ci, zero_one_indices)
            elseif S == MOI.Integer
                update_integer_indices(user_optimizer, mapping, ci, integer_indices)
            end
        else
            ## Update conmap for (F,S) for F != MOI.SingleVariable
            ## Single variables are treated by bounds in Cbc, so no
            ## need to add a row
            for i in 1:length(ci)
                mapping.conmap[ci[i]] = MOI.ConstraintIndex{F,S}(num_rows + i)
            end
            num_rows += MOI.get(user_optimizer, MOI.NumberOfConstraints{F,S}())
        end
    end

    cbc_model_format = CbcModelFormat(num_rows, num_cols)

    copy_constraints!(cbc_model_format, user_optimizer, mapping)
    update_bounds_for_binary_vars!(cbc_model_format.col_lb, cbc_model_format.col_ub, zero_one_indices)


    ## Copy objective function
    objF = MOI.get(user_optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    load_obj(cbc_model_format, mapping, objF)
    sense = MOI.get(user_optimizer, MOI.ObjectiveSense())
    MOI.set(cbc_optimizer, MOI.ObjectiveSense(), sense)

    ## Load the problem to Cbc
    CbcCI.loadProblem(cbc_optimizer.inner, sparse(cbc_model_format.row_idx, cbc_model_format.col_idx,
                                                  cbc_model_format.values, cbc_model_format.num_rows,
                                                  cbc_model_format.num_cols),
                      cbc_model_format.col_lb, cbc_model_format.col_ub, cbc_model_format.obj,
                      cbc_model_format.row_lb, cbc_model_format.row_ub)
    
    empty!(cbc_model_format.row_idx)
    empty!(cbc_model_format.col_idx)
    empty!(cbc_model_format.values)    

    ## Set integer variables
    for idx in vcat(integer_indices, zero_one_indices)
        CbcCI.setInteger(cbc_optimizer.inner, idx-1)
    end

    return MOIU.IndexMap(mapping.varmap, mapping.conmap)
end


function MOI.optimize!(cbc_optimizer::CbcOptimizer)
    # Call solve function
    CbcCI.solve(cbc_optimizer.inner)
end



## canadd, canset, canget functions

function MOI.add_variable(cbc_optimizer::CbcOptimizer)
    throw(MOI.AddVariableNotAllowed())
end

## supports constraints


MOI.supports_constraint(::CbcOptimizer, ::Type{<:Union{MOI.ScalarAffineFunction{Float64}, MOI.SingleVariable}},
::Type{<:Union{MOI.EqualTo{Float64}, MOI.Interval{Float64}, MOI.LessThan{Float64},
MOI.GreaterThan{Float64}, MOI.ZeroOne, MOI.Integer}}) = true

MOI.supports(cbc_optimizer::CbcOptimizer, object::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true

MOI.supports(cbc_optimizer::CbcOptimizer, object::MOI.ObjectiveSense) = true

## Set functions

function MOI.write_to_file(cbc_optimizer::CbcOptimizer, filename::String)
    if !endswith("filename", "mps")
        error("CbcOptimizer only supports writing .mps files")
    else
        writeMps(cbc_optimizer.inner, filename)
    end
end


# empty!
function MOI.empty!(cbc_optimizer::CbcOptimizer)
    cbc_optimizer.inner = CbcModel()
end


function MOI.set(cbc_optimizer::CbcOptimizer, object::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    if sense == MOI.MaxSense
        CbcCI.setObjSense(cbc_optimizer.inner, -1)
    else ## Other senses are set as minimization (cbc default)
        CbcCI.setObjSense(cbc_optimizer.inner, 1)
    end
end


## Get functions

function MOI.is_empty(cbc_optimizer::CbcOptimizer)
    return (CbcCI.getNumCols(cbc_optimizer.inner) == 0 && CbcCI.getNumRows(cbc_optimizer.inner) == 0)
end

MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.NumberOfVariables) = getNumCols(cbc_optimizer.inner)

MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.ObjectiveBound) = CbcCI.getBestPossibleObjValue(cbc_optimizer.inner)

MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.NodeCount) = CbcCI.getNodeCount(cbc_optimizer.inner)


function MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.ObjectiveValue)
    return CbcCI.getObjValue(cbc_optimizer.inner)
end

function MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.VariablePrimal, ref::MOI.VariableIndex)
    variablePrimals = CbcCI.getColSolution(cbc_optimizer.inner)
    return variablePrimals[ref.value]
end

function MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.VariablePrimal, ref::Vector{MOI.VariableIndex})
    variablePrimals = CbcCI.getColSolution(cbc_optimizer.inner)
    return [variablePrimals[vi.value] for vi in ref]
end


function MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.ResultCount)
    if (isProvenInfeasible(cbc_optimizer.inner) || isContinuousUnbounded(cbc_optimizer.inner)
        || isAbandoned(cbc_optimizer.inner) || CbcCI.getObjValue(cbc_optimizer.inner) >= 1e300)
        return 0
    end
    return 1
end


function MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.ObjectiveSense)
    CbcCI.getObjSense(cbc_optimizer.inner) == 1 && return MOI.MinSense
    CbcCI.getObjSense(cbc_optimizer.inner) == -1 && return MOI.MaxSense
end



function MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.TerminationStatus)

    if isProvenInfeasible(cbc_optimizer.inner)
        return MOI.InfeasibleNoResult
    elseif isContinuousUnbounded(cbc_optimizer.inner)
        return MOI.InfeasibleOrUnbounded
    elseif isNodeLimitReached(cbc_optimizer.inner)
        return MOI.NodeLimit
    elseif isSecondsLimitReached(cbc_optimizer.inner)
        return MOI.TimeLimit
    elseif isSolutionLimitReached(cbc_optimizer.inner)
        return MOI.SolutionLimit
    elseif (isProvenOptimal(cbc_optimizer.inner) || isInitialSolveProvenOptimal(cbc_optimizer.inner)
        || MOI.get(cbc_optimizer, MOI.ResultCount()) == 1)
        return MOI.Success
    elseif isAbandoned(cbc_optimizer.inner)
        return MOI.Interrupted
    else
        error("Internal error: Unrecognized solution status")
    end

end

function MOI.get(cbc_optimizer::CbcOptimizer, object::MOI.PrimalStatus)
    if isProvenOptimal(cbc_optimizer.inner) || isInitialSolveProvenOptimal(cbc_optimizer.inner)
        return MOI.FeasiblePoint
    elseif isProvenInfeasible(cbc_optimizer.inner)
        return MOI.InfeasiblePoint
    end
end
