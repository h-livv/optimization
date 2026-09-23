# Optimization

### From-scratch implementations of optimization algorithms, derived directly from their mathematical formulations and implemented using **NumPy**.

The goal is to understand the connection between

**mathematics → algorithm → implementation → behavior**

rather than treating optimization methods as black-box procedures.

> **Status:** In progress alongside my study of optimization. I’ll add
> methods and examples as I work through them.

## Implemented

### Simplex Method

A from-scratch implementation of the simplex method for linear programming.

<img src="assets/simplex.gif" alt="Simplex method visualization" width="500">

The implementation exposes the simplex trajectory through the feasible region,
including basis changes, pivot directions, reduced costs, and objective values.

It explores:

* standard-form linear programs
* basic feasible solutions
* basis selection and exchange
* pivot operations
* the ratio test
* movement between vertices
* the linear-algebraic structure of simplex

## Notes

The simplex method is the first implementation in what I expect to be a
growing collection. Each method starts from its mathematical formulation and
is worked through in code.
