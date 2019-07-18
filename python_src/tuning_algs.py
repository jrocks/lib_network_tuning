import sys
import network_solver as ns
import numpy.random as rand
import numpy as np
import numpy.random as rand
import numpy.linalg as la
import scipy.sparse.linalg as sparsela
import scipy.optimize as spo
import network_solver as ns
import itertools as it




def tune_disc_lin_greedy(solver, obj_func, K_disc_init, K_min, K_max, meas_func=lambda x: x, K_fix = set(), NDISC=1, NCONVERGE=1, tol=1e-8, verbose=True, offset=True):
                
    #Set initial response ratio
    K_disc_curr = np.copy(K_disc_init)
    K_curr = (K_max-K_min) * K_disc_curr / NDISC + K_min
    solver.setK(K_curr)
        
    state = solver.getSolverState()
    
    result = solver.solve(state)
    solver.computeMeas(result)
    
    meas_init = meas_func(result.meas)
    
    if verbose:
        print("Initial meas:")
        print(result.meas)
        print(meas_init)

    if offset:
        obj_func.setOffset(meas_init)
    
    # Calculate initial response
    obj_prev = obj_func.evalFunc(meas_init)
    obj_curr = obj_prev
    K_prev = np.copy(K_curr)
    K_disc_prev = np.copy(K_disc_curr)

    if verbose:
        print("Initial objective function:", obj_prev)
    
    NE = solver.nw.NE
    
    move_list = [[] for b in range(NE)]
    for b in range(NE):
        
        if b in K_fix:
            continue
        
        if K_disc_curr[b] == NDISC: 
            move_list[b].append(ns.LinUpdate(1, [b], np.array([-(K_max[b]-K_min[b]) / NDISC])))
        elif K_disc_curr[b] == 0:
            move_list[b].append(ns.LinUpdate(1, [b], np.array([(K_max[b]-K_min[b]) / NDISC])))
        else:
            move_list[b].append(ns.LinUpdate(1, [b], np.array([-(K_max[b]-K_min[b]) / NDISC])))
            move_list[b].append(ns.LinUpdate(1, [b], np.array([(K_max[b]-K_min[b]) / NDISC])))
        
    n_iter = 0
    converge_count = 0
    
    executed_moves = []
    
    while True:

#         if obj_curr == 0.0:
        if obj_curr <= tol:
            break

        obj_list = []
        valid_move_list = []
        meas_list = []
        raw_meas_list =[]

        n_zero = 0

        for up in it.chain.from_iterable(move_list):
            
            result = solver.solve(up, state)
                        
            if not result.success:
                n_zero += 1
                continue
            
            solver.computeMeas(result)
            meas = meas_func(result.meas)
            obj = obj_func.evalFunc(meas)
                        
            obj_list.append(obj)
            valid_move_list.append(up)
            meas_list.append(meas)
            raw_meas_list.append(result.meas)
            
#             if n_iter == 450:
#                 print(result.meas)
                 
                
        if verbose:
            print("Removing", n_zero, "/", np.sum(K_disc_curr), "/",solver.nw.NE, "bonds would create zero modes...")
                
        if len(valid_move_list) == 0:
            if verbose:
                print("No solution found.")
            break


        args = np.argsort(obj_list)   

        min_list = []
        for i in range(len(obj_list)):
#             if obj_list[args[i]] == 0.0:
            if obj_list[args[i]] < tol:
                min_list.append(args[i])
            else:
                break

        if len(min_list) > 0:
            
            rand.shuffle(min_list)
            index = min_list[0]
        else:
            index = args[0]

        min_move = valid_move_list[index]
        obj_curr = obj_list[index]

        if verbose:
            # print(min_move.dK_edges, min_move.dK)
            print("Meas:")
            print(raw_meas_list[index])
            print(meas_list[index])
            
        K_disc_curr[min_move.dK_edges] += int(np.rint(min_move.dK / (K_max[min_move.dK_edges]-K_min[min_move.dK_edges]) * NDISC))
              
        K_curr = (K_max-K_min) * K_disc_curr / NDISC  + K_min
        
        if verbose:
            print(n_iter, ":", "Move", min_move.dK_edges, min_move.dK)
            print(n_iter, ":", "Objective function:", obj_curr, "Change:", obj_curr - obj_prev, "Percent:", 100 * (obj_curr - obj_prev) / np.abs(obj_prev), "%")
                        
        if (obj_curr - obj_prev) / np.abs(obj_prev) > -tol:
#         if (obj_curr - obj_prev) > -1e-8:
            converge_count += 1
            print("Steps Backwards", converge_count, "/", NCONVERGE)
            if converge_count >= NCONVERGE:
                if verbose:
                    print("Stopped making progress.")
                break
        else:            
            obj_prev = obj_curr
            K_prev = np.copy(K_curr)
            K_disc_prev = np.copy(K_disc_curr)
            converge_count = 0
            executed_moves.append(dict(zip(min_move.dK_edges, min_move.dK)))


        solver.updateSolverState(min_move, state)         
        
        bond = min_move.dK_edges[0]
        move_list[bond] = []
        if K_disc_curr[bond] == NDISC: 
            move_list[bond].append(ns.LinUpdate(1, [bond], np.array([-(K_max[bond]-K_min[bond]) / NDISC])))
        elif K_disc_curr[bond] == 0:
            move_list[bond].append(ns.LinUpdate(1, [bond], np.array([(K_max[bond]-K_min[bond]) / NDISC])))
        else:
            move_list[bond].append(ns.LinUpdate(1, [bond], np.array([-(K_max[bond]-K_min[bond]) / NDISC])))
            move_list[bond].append(ns.LinUpdate(1, [bond], np.array([(K_max[bond]-K_min[bond]) / NDISC])))

        n_iter += 1            
            
    obj = obj_prev
    K = K_prev
        
    solver.setK(K)
    result = solver.solve()
    solver.computeMeas(result)
    meas_final = meas_func(result.meas)
    obj_real = obj_func.evalFunc(meas_final)
    
#     if verbose:   
#         evals = sparsela.eigsh(solver.H, k=16, return_eigenvectors=False, which='SM')
#         print(evals)

    # res = self.obj_func.projMeas(meas)
    if verbose:
        print("Abs Obj Error:", obj - obj_real)
        print("Init Measure:", meas_init)
        print("Final Measure:", meas_final)

    if verbose:
        print("Rel Change:", (meas_final - meas_init) / (1e-8 + meas_init))
        print("Abs Change:", meas_final - meas_init)

    data = dict()
        
    data['niter'] = n_iter
    # data['min_eval'] = evals[3]
    data['K'] = K
    data['K_disc'] = K_disc_prev
    data['K_init'] = (K_max-K_min) * K_disc_init / NDISC + K_min
    # data['NR'] = self.net.NE - np.sum(K_disc_prev)
    # data['DZ_final'] = 2.0 * np.sum(K_disc_prev) / self.net.NN - 2.0 * (self.net.DIM - 1.0 * self.net.DIM / self.net.NN)
    data['obj_err'] = obj - obj_real
    data['obj_func'] = obj
    # data['obj_func_terms'] = obj_terms
    # data['max_error'] = max_error
    # data['condition'] = self.solver.getConditionNum()
    # data['obj_res'] = res
    data['moves'] = executed_moves

    if obj < tol:
#     if obj == 0.0:
        data['success_flag'] = 0
        data['result_msg'] = "Valid solution found."
    else:
        data['success_flag'] = 1
        data['result_msg'] = "No valid solution found."

    return data   
        


def tune_cont_lin(solver, obj_func, K_init, K_min, K_max, tol=1e-8, verbose=True):
    
    solver.setK(K_init)
    result = solver.solveMeas()
    meas_init = np.array(result.meas)
    obj_func.setOffset(meas_init)
    
    
    bounds = [(K_min[i], K_max[i]) for i in range(len(K_min))]   
          
    def func(K):
        solver.setK(K)
        result = solver.solveMeasGrad()
        
        f = obj_func.evalFunc(np.array(result.meas))
        
        return f
    
    def grad(K):
        solver.setK(K)
        result = solver.solveMeasGrad()        
        g = obj_func.evalGrad(np.array(result.meas), np.array(result.meas_grad))
        
        return g
        
    print(spo.check_grad(func, grad, K_max))
        
    def func_grad(K):
        solver.setK(K)
        result = solver.solveMeasGrad()
        
        f = obj_func.evalFunc(np.array(result.meas))
        
        g = obj_func.evalGrad(np.array(result.meas), np.array(result.meas_grad))
        
        return (f, g)
    
    # print(func(K_init))
    
    
    res = spo.minimize(func_grad, K_max, method='L-BFGS-B', callback=None,
                               jac=True, options={'disp': True, 'gtol': tol, 'ftol': 0.0},
                              bounds=bounds)
    
    print(res)

    print(0.5*np.sum((res.x-K_init)**2))
    
    def func_grad(K):
        solver.setK(K)
        result = solver.solveMeasGrad()
        
        mu = 1e-6
        
        f = obj_func.evalFunc(np.array(result.meas)) + mu*0.5*np.sum((K-K_init)**2)
        
        g = obj_func.evalGrad(np.array(result.meas), np.array(result.meas_grad)) + mu*(K-K_init)
        
        return (f, g)
        
    res = spo.minimize(func_grad, res.x, method='L-BFGS-B', callback=None,
                               jac=True, options={'disp': True, 'gtol': tol, 'ftol': 0.0, 'maxls': 1000},
                              bounds=bounds)
    
    print(res)
        
    print(0.5*np.sum((res.x-K_init)**2))
          
    solver.setK(res.x)
    result = solver.solveMeas()
    f = obj_func.evalFunc(np.array(result.meas))
          
    print("func:", f)
        
    data = {}
    data['K'] = res.x
    data['obj_func'] = f
    
    data['success_flag'] = (f == 0.0)
    data['result_msg'] = res.message
    
    return data

 
    

        
        
        
def tune_disc_lin_greedy_cplex(solver, obj_func, K_max, K_disc, K_fix = set(), NDISC=1, NCONVERGE=1, tol=1e-8, verbose=True):
                
    #Set initial response ratio
    K_disc_curr = np.copy(K_disc)
    K_curr = K_max * K_disc_curr / NDISC
    solver.setK(K_curr)
        
    result = solver.solve()
    solver.computeMeas(result)
     
    meas_init = np.array(result.meas)
    
    if verbose:
        print("Initial meas:")
        print(meas_init)

    obj_func.setOffset(meas_init)
    
    # Calculate initial response
    obj_prev = obj_func.evalFunc(meas_init)
    obj_curr = obj_prev
    K_prev = np.copy(K_curr)
    K_disc_prev = np.copy(K_disc_curr)

    if verbose:
        print("Initial objective function:", obj_prev)
    
    NE = solver.nw.NE
    
    move_list = [[] for b in range(NE)]
    for b in range(NE):
        
        if b in K_fix:
            continue
        
        if K_disc_curr[b] == NDISC: 
            move_list[b].append(ns.LinUpdate(1, [b], np.array([-K_max[b] / NDISC])))
        elif K_disc_curr[b] == 0:
            move_list[b].append(ns.LinUpdate(1, [b], np.array([K_max[b] / NDISC])))
        else:
            move_list[b].append(ns.LinUpdate(1, [b], np.array([-K_max[b] / NDISC])))
            move_list[b].append(ns.LinUpdate(1, [b], np.array([K_max[b] / NDISC])))
        
    n_iter = 0
    converge_count = 0
    
    executed_moves = []
    
    while True:

        if obj_curr == 0.0:
            break

        obj_list = []
        valid_move_list = []
        meas_list = []

        n_zero = 0

        for i, up in enumerate(it.chain.from_iterable(move_list)):
            
            if i % 100 == 0:
                print(i)
        
            result = solver.solve(up)                                    
            
            solver.computeMeas(result)
            meas = np.array(result.meas)
            obj = obj_func.evalFunc(meas)
            
            obj_list.append(obj)
            valid_move_list.append(up)
            meas_list.append(meas)
                 
                
        if verbose:
            print("Removing", n_zero, "/", np.sum(K_disc_curr), "/",solver.nw.NE, "bonds would create zero modes...")
                
        if len(valid_move_list) == 0:
            if verbose:
                print("No solution found.")
            break


        args = np.argsort(obj_list)   

        min_list = []
        for i in range(len(obj_list)):
            if obj_list[args[i]] == 0.0:
                min_list.append(args[i])
            else:
                break

        if len(min_list) > 0:
            
            rand.shuffle(min_list)
            index = min_list[0]
        else:
            index = args[0]

        min_move = valid_move_list[index]
        obj_curr = obj_list[index]

        if verbose:
            # print(min_move.dK_edges, min_move.dK)
            print("Meas", meas_list[index])

        K_disc_curr[min_move.dK_edges] += int(min_move.dK / K_max[min_move.dK_edges])
        K_curr = K_max * K_disc_curr / NDISC
        
        if verbose:
            print(n_iter, ":", "Move", min_move.dK_edges, min_move.dK)
            print(n_iter, ":", "Objective function:", obj_curr, "Change:", obj_curr - obj_prev, "Percent:", 100 * (obj_curr - obj_prev) / np.abs(obj_prev), "%")
                        
        if (obj_curr - obj_prev) / np.abs(obj_prev) > -tol:
            converge_count += 1
            print("Steps Backwards", converge_count, "/", NCONVERGE)
            if converge_count >= NCONVERGE:
                if verbose:
                    print("Stopped making progress.")
                break
        else:            
            obj_prev = obj_curr
            K_prev = np.copy(K_curr)
            K_disc_prev = np.copy(K_disc_curr)
            converge_count = 0
            executed_moves.append(dict(zip(min_move.dK_edges, min_move.dK)))
            
        solver.setUpdate(min_move, False)       
        
        bond = min_move.dK_edges[0]
        move_list[bond] = []
        if K_disc_curr[bond] == NDISC: 
            move_list[bond].append(ns.LinUpdate(1, [bond], np.array([-K_max[bond] / NDISC])))
        elif K_disc_curr[bond] == 0:
            move_list[bond].append(ns.LinUpdate(1, [bond], np.array([K_max[bond] / NDISC])))
        else:
            move_list[bond].append(ns.LinUpdate(1, [bond], np.array([-K_max[bond] / NDISC])))
            move_list[bond].append(ns.LinUpdate(1, [bond], np.array([K_max[bond] / NDISC])))

        n_iter += 1            
            
    obj = obj_prev
    K = K_prev
    
    
    solver.setK(K)
    result = solver.solve()
    solver.computeMeas(result)
    meas_final = np.array(result.meas)
    obj_real = obj_func.evalFunc(meas_final)
    
    if verbose:   
        evals = sparsela.eigsh(solver.H, k=16, return_eigenvectors=False, which='SM')
        print(evals)

    # res = self.obj_func.projMeas(meas)
    if verbose:
        print("Abs Obj Error:", obj - obj_real)
        print("Init Measure:", meas_init)
        print("Final Measure:", meas_final)

    if verbose:
        print("Rel Change:", (meas_final - meas_init) / meas_init)
        print("Abs Change:", meas_final - meas_init)

    data = dict()
        
    data['niter'] = n_iter
    # data['min_eval'] = evals[3]
    data['K'] = K
    data['K_disc'] = K_disc_prev
    # data['NR'] = self.net.NE - np.sum(K_disc_prev)
    # data['DZ_final'] = 2.0 * np.sum(K_disc_prev) / self.net.NN - 2.0 * (self.net.DIM - 1.0 * self.net.DIM / self.net.NN)
    data['obj_err'] = obj - obj_real
    data['obj_func'] = obj
    # data['obj_func_terms'] = obj_terms
    # data['max_error'] = max_error
    # data['condition'] = self.solver.getConditionNum()
    # data['obj_res'] = res
    data['moves'] = executed_moves

    if obj == 0.0:
        data['success_flag'] = 0
        data['result_msg'] = "Valid solution found."
    else:
        data['success_flag'] = 1
        data['result_msg'] = "No valid solution found."

    return data   
        
        
    