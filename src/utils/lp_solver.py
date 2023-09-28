from turtle import up
import numpy as np
from pulp import *
import pulp as lp
import re

def lp_solver_pulp(cost_matrix):
    # note that there is no binary variable here, so we are solving a normal constrained lp, not an ilp

    n_mechanisms = cost_matrix.shape[0]
    p = cost_matrix.shape[1]
    num_slots = int(np.sqrt(p))
    costs_flattened = np.reshape(cost_matrix, n_mechanisms * p)
    # costs_indices = list(product(range(n_mechanisms), range(p)))
    
    # using cts. variables so that the problem gets solved in polynomial time
    coeffs = lp.LpVariable.dicts("coeffs", ((i, j) for i in range(n_mechanisms) for j in range(p)), lowBound=0.0, upBound=1.0)
    costs = dict(zip(coeffs.keys(), list(costs_flattened)))

    prob = LpProblem("3_dim_matching", LpMinimize)
    prob += lpSum([costs[i,j]*coeffs[i,j] for i in range(n_mechanisms) for j in range(p)])
    # prob += lpSum([coeffs[i,j] for i in range(m) for j in range(p)]) = m, "Perfect matching of mechanisms"
    for i in range(n_mechanisms):
        prob += lpSum([coeffs[i,j] for j in range(p)]) == 1, f"{i}-Perfect matching of mechanisms"
    for k in range(num_slots):
        prob += lpSum([coeffs[i,j] for i in range(n_mechanisms) for j in range(k*num_slots, (k+1)*num_slots)]) <= 1, f"{k}-No duplicate slots at t"
    for k in range(num_slots):
        prob += lpSum([coeffs[i,j] for i in range(n_mechanisms) for j in range(k, p, num_slots)]) <= 1, f"{k}-No duplicate slots at t+1"  

    # prob += lpSum([coeffs[i,j] for i in range(m) for j in range(p)]) = m, "No duplicate slots at t"
    # prob += lpSum([coeffs[i,j] for i in range(m) for j in range(p)]) = m, "No duplicate slots at t+1"
    # LpStatus[prob.status]

    # prob.solve(COINMP_DLL(msg=0))
    # prob.solve(COIN_CMD(msg=0))
    prob.solve(PULP_CBC_CMD(msg=0))
    # print("Status:", LpStatus[prob.status])
    # for v in prob.variables():
    #     if v.varValue>0:
    #         print(v.name, "=", v.varValue)

    obj = value(prob.objective)
    # print("The total cost of this balanced diet is: ${}".format(round(obj,2)))

    
    indices = []
    # for i, v in enumerate(prob.variables()):
    # # print(v.name)
    #     if v.varValue>0:
    #         _indices = list(map(int, re.findall(r"([0-9]+)", v.name)))
    #         indices.append(_indices)
    #         # print(f"{v.name}, '=', {v.varValue}, cost={costs[_indices[0], _indices[1]]}")
    #     # print(np.array(indices))
    # indices = np.array(indices)
    
    return indices


def ilp_solver_pulp(cost_matrix):
    n_mechanisms = cost_matrix.shape[0]
    p = cost_matrix.shape[1]
    num_slots = int(np.sqrt(p))
    costs_flattened = np.reshape(cost_matrix, n_mechanisms * p)
    # costs_indices = list(product(range(n_mechanisms), range(p)))
    
    # using binary values resulting in an integer linear programming (NP complete)
    coeffs = lp.LpVariable.dicts("coeffs", ((i, j) for i in range(n_mechanisms) for j in range(p)), cat=LpBinary)
    costs = dict(zip(coeffs.keys(), list(costs_flattened)))

    prob = LpProblem("3_dim_matching", LpMinimize)
    prob += lpSum([costs[i,j]*coeffs[i,j] for i in range(n_mechanisms) for j in range(p)])
    # prob += lpSum([coeffs[i,j] for i in range(m) for j in range(p)]) = m, "Perfect matching of mechanisms"
    for i in range(n_mechanisms):
        prob += lpSum([coeffs[i,j] for j in range(p)]) == 1, f"{i}-Perfect matching of mechanisms"
    for k in range(num_slots):
        prob += lpSum([coeffs[i,j] for i in range(n_mechanisms) for j in range(k*num_slots, (k+1)*num_slots)]) <= 1, f"{k}-No duplicate slots at t"
    for k in range(num_slots):
        prob += lpSum([coeffs[i,j] for i in range(n_mechanisms) for j in range(k, p, num_slots)]) <= 1, f"{k}-No duplicate slots at t+1"  

    # prob += lpSum([coeffs[i,j] for i in range(m) for j in range(p)]) = m, "No duplicate slots at t"
    # prob += lpSum([coeffs[i,j] for i in range(m) for j in range(p)]) = m, "No duplicate slots at t+1"
    # LpStatus[prob.status]

    # prob.solve(COINMP_DLL(msg=0))
    # prob.solve(COIN_CMD(msg=0))
    
    # narval uses coin
    # solver_list = lp.listSolvers(onlyAvailable=True)
    # print(f"---------------------\nsolver_list:\n{solver_list}---------------------\n")
    prob.solve(COIN_CMD(msg=0))
    # prob.solve(PULP_CBC_CMD(msg=0))
    # print("Status:", LpStatus[prob.status])
    # for v in prob.variables():
    #     if v.varValue>0:
    #         print(v.name, "=", v.varValue)

    obj = value(prob.objective)
    # print("The total cost of this balanced diet is: ${}".format(round(obj,2)))

    
    indices = []
    for i, v in enumerate(prob.variables()):
    # print(v.name)
        if v.varValue>0:
            _indices = list(map(int, re.findall(r"([0-9]+)", v.name)))
            indices.append(_indices)
            # print(f"{v.name}, '=', {v.varValue}, cost={costs[_indices[0], _indices[1]]}")
        # print(np.array(indices))
    indices = np.array(indices)
    
    return indices


def lp_solver_pulp_unconstrained(cost_matrix):
    n_mechanisms = cost_matrix.shape[0]
    num_slots = cost_matrix.shape[1]
    p = num_slots
    costs_flattened = np.reshape(cost_matrix, n_mechanisms * p)
    # costs_indices = list(product(range(n_mechanisms), range(p)))
    coeffs = lp.LpVariable.dicts("coeffs", ((i, j) for i in range(n_mechanisms) for j in range(p)), lowBound=0.0, upBound=1.0) #, cat=LpBinary)
    costs = dict(zip(coeffs.keys(), list(costs_flattened)))

    prob = LpProblem("3_dim_matching", LpMinimize)
    prob += lpSum([costs[i,j]*coeffs[i,j] for i in range(n_mechanisms) for j in range(p)])
    # prob += lpSum([coeffs[i,j] for i in range(m) for j in range(p)]) = m, "Perfect matching of mechanisms"
    for i in range(n_mechanisms):
        prob += lpSum([coeffs[i,j] for j in range(p)]) == 1, f"{i}-Perfect matching of mechanisms"
    

    prob.solve(PULP_CBC_CMD(msg=0))
    obj = value(prob.objective)

    
    indices = []
    for i, v in enumerate(prob.variables()):

        if v.varValue>0:
            _indices = list(map(int, re.findall(r"([0-9]+)", v.name)))
            indices.append(_indices)
            # print(f"{v.name}, '=', {v.varValue}, cost={costs[_indices[0], _indices[1]]}")

    indices = np.array(indices)
    
    return indices


import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

def lp_solver_cvxpy(n_mechanisms, num_slots):

    p = num_slots ** 2
    
    # what are the variables?
    w = cp.Variable((n_mechanisms, p))
    # w = cp.Variable((n_mechanisms, p), boolean=True)

    # what are the parameters? it's only the cost matrix
    C = cp.Parameter((n_mechanisms, p))

    # # scalarized constraints

    # # constraints:
    # constraints = []
    # for i in range(n_mechanisms):
    #     # Perfect matching of mechanisms in rows
    #     constraints.extend([cp.sum([w[i,j] for j in range(p)]) == 1])
    # for k in range(num_slots):
    #     # No duplicate slots at t
    #     constraints.extend([cp.sum([w[i,j] for i in range(n_mechanisms) for j in range(k*num_slots, (k+1)*num_slots)]) <= 1])
    # for k in range(num_slots):
    #     # No duplicate slots at t+1
    #     constraints.extend([cp.sum([w[i,j] for i in range(n_mechanisms) for j in range(k, p, num_slots)]) <= 1])


    # vectorized constraints
    constraints = []
    # Perfect matching of mechanisms in rows
    constraints.extend([cp.sum(w, axis=1) == np.ones((n_mechanisms,))])

    # No duplicate slots at t

    # define some mask matrix the same as cost
    # define a vector whose entries should be the sum of entries of mask_i * w
    # then the constraint would be: vector <= 1
    # it is now vectorized

    # constraint_vector = np.zeros((num_slots, 1))
    constraint = []
    for k in range(num_slots):
        temp_mask = np.zeros((n_mechanisms, p))
        for i in range(n_mechanisms):
            for j in range(k*num_slots, (k+1)*num_slots):
                temp_mask[i,j] = 1.0
        
        temp_mask = (temp_mask == 1.0)
        constraint.append(cp.sum(w[temp_mask]))

    constraint = cp.vstack(constraint)
    constraints.extend([constraint <= np.ones((num_slots, 1))])


    # No duplicate slots at t+1

    # define some mask matrix the same as cost
    # define a vector whose entries should be the sum of entries of mask_i * w
    # then the constraint would be: vector <= 1
    # it is now vectorized

    # constraint_vector = np.zeros((num_slots, 1))
    constraint = []
    for k in range(num_slots):
        temp_mask = np.zeros((n_mechanisms, p))
        for i in range(n_mechanisms):
            for j in range(k, p, num_slots):
                temp_mask[i,j] = 1.0

        temp_mask = (temp_mask == 1.0)
        constraint.append(cp.sum(w[temp_mask]))
        
        # constraint.append(cp.sum(cp.multiply(temp_mask, w)))

        # constraint_vector[k] = cp.sum(cp.multiply(temp_mask, w))

    constraint = cp.vstack(constraint)
    constraints.extend([constraint <= np.ones((num_slots, 1))])

    # weights should be between zero and one (no integer forcing to avoid ILP complexity)
    constraints.extend([w >= 0.001])
    constraints.extend([w <= 1.0])

    # problem: objective and the problem:
    objective = cp.Minimize(cp.sum(cp.multiply(w, C)))
    # objective = cp.Minimize(cp.sum(cp.multiply(w, C))**2)
    problem = cp.Problem(objective, constraints)

    assert problem.is_dpp()
    cvxpylayer = CvxpyLayer(problem, parameters=[C], variables=[w])

    return cvxpylayer


def lp_solver_cvxpy_unconstrained(n_mechanisms, num_slots):
    
    # what are the variables?
    w = cp.Variable((n_mechanisms, num_slots))

    # what are the parameters? it's only the cost matrix
    C = cp.Parameter((n_mechanisms, num_slots))

    # vectorized constraints
    constraints = []
    # Perfect matching of mechanisms in rows
    constraints.extend([cp.sum(w, axis=1) == np.ones((n_mechanisms,))])

    # weights should be between zero and one (no integer forcing to avoid ILP complexity)
    constraints.extend([w >= 0.001])
    constraints.extend([w <= 1.0])

    objective = cp.Minimize(cp.sum(cp.multiply(w, C))**2)
    problem = cp.Problem(objective, constraints)

    assert problem.is_dpp()
    cvxpylayer = CvxpyLayer(problem, parameters=[C], variables=[w])

    return cvxpylayer
