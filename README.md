# COIN-OR Branch and Cut Interface (Cbc.jl)

[![](https://www.coin-or.org/wordpress/wp-content/uploads/2014/08/COINOR.png)](https://www.coin-or.org)

`Cbc.jl` is an interface to the **[COIN-OR Branch and Cut](https://projects.coin-or.org/Cbc)**
solver. It provides a complete interface to the low-level C API, as well as an
implementation of the solver-independent `MathProgBase` and `MathOptInterface`
API's.   

*Note: This wrapper is maintained by the JuliaOpt community and is not a COIN-OR
project.*


[![Build Status](https://travis-ci.org/JuliaOpt/Cbc.jl.svg?branch=master)](https://travis-ci.org/JuliaOpt/Cbc.jl)

[![Cbc](http://pkg.julialang.org/badges/Cbc_0.6.svg)](http://pkg.julialang.org/?pkg=Cbc&ver=0.6)

## Installation

The package is registered in `METADATA.jl` and so can be installed with
`Pkg.add`.

```
julia> Pkg.add("Cbc")
```

The supported platforms are Linux, OS X, and Windows. Binaries are provided for
Windows and OS X, and will be installed by default. On Linux, Cbc will be
automatically compiled from source. Ensure that a C++ compiler is installed
first; on Debian-based systems, install the ``build-essential`` package.

### Using with **[MathProgBase]**

Cbc provides a solver object that can be passed to ``mixintprog`` in MathProgBase (and used to create instances of the solver-independent ``AbstractMathProgModel`` type):

```julia
using Cbc
using MathProgBase
mixintprog(..., CbcSolver(Option1=value1,Option2=value2,...))
```

see the MathProgBase documentation for further information.

Options are solver-dependent, and unfortunately not well documented.
The following options are likely to be the most useful:

* ``seconds`` -- Solution timeout limit. (Must be a ``Float64``)
* ``logLevel`` -- Set to 1 to enable solution output.
* ``maxSolutions`` -- Terminate after this many feasible solutions have been found.
* ``maxNodes`` -- Terminate after this many branch-and-bound nodes have been evaluated.
* ``allowableGap`` -- Terminate after optimality gap is less than this value (on an absolute scale).
* ``ratioGap`` -- Terminate after optimality gap is smaller than this relative fraction.
* ``threads`` -- Set the number of threads to use for parallel branch & bound

The complete list of parameters can be found by running the ``cbc`` executable and typing ``?`` at the prompt.

In addition, we provide the julia-specific option ``check_warmstart`` which, if set to ``false``, will tell the wrapper to pass along the warmstart solution regardless of if it satisfies the constraints of the problem. The default value is ``true``.

### Using the C interface

The low-level C interface is available in the ``CbcCInterface`` submodule:
```julia
using Cbc.CbcCInterface
```

Using this interface is not recommended.

[Cbc]: https://projects.coin-or.org/Cbc
[MathProgBase]: https://github.com/JuliaOpt/MathProgBase.jl
