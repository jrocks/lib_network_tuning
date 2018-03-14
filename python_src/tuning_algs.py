import sys
import network_solver as ns
import numpy.random as rand
import numpy as np
import numpy.random as rand
import numpy.linalg as la
import scipy.sparse.linalg as sparsela
import network_solver as ns
import itertools as it


 
    
def tune_disc_lin_greedy(solver, obj_func, K_max, K_disc, NDISC=1, NCONVERGE=1, tol=1e-8, verbose=True):
                
    #Set initial response ratio
    K_disc_curr = np.copy(K_disc)
    K_curr = K_max * K_disc_curr / NDISC
    solver.setK(K_curr)
    
    state = solver.getSolverState()
    
    result = solver.solve(state)
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

        for up in it.chain.from_iterable(move_list):
            
            result = solver.solve(up, state)
                        
            if not result.success:
                n_zero += 1
                continue
            
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


        solver.updateSolverState(min_move, state)         
        
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
        
        
        
        
        
        
        
#     self.obj_func.setRatioInit(meas)

#     # Calculate initial respons e
#     obj_prev = self.func(K_curr)
#     obj_curr = obj_prev
#     K_prev = np.copy(K_curr)
#     K_disc_prev = np.copy(self.K_disc)

#     if verbose:
#         print "Initial objective function:", obj_prev        

#     move_list = []
#     for b in range(self.net.NE):
#         move_list.append({'bond': b, 'disc': np.min([self.NDISC, self.K_disc[b]+1])})
#         move_list.append({'bond': b, 'disc': np.max([0, self.K_disc[b] - 1])})


#     up_list = []
#     for move in move_list:
#         up = Update()
#         up.setStretchMod(1, [move['bond']], [self.K_max[move['bond']] * move['disc'] / self.NDISC])
#         up_list.append(up.getCyUpdate())

#     self.solver.prepareUpdateList(up_list)


#     # tol = np.sqrt(np.finfo(float).eps)
#     # tol = 1e-10
#     tol = 1e-8

#     rem_set = set()

#     max_error = 0

#     n_iter = 0
#     converge_count = 0
#     target_NE = np.sum(self.K_disc)
#     current_NE = np.sum(self.K_disc)

#     bond_prev = -1

#     while True:

#         if obj_curr == 0.0:
#             break

#         obj_list = []
#         valid_move_list = []
#         meas_list = []

#         n_zero = 0

#         for i, move in enumerate(move_list):


#             bond = move['bond']
#             disc = move['disc']

#             if self.fix_NE and current_NE != target_NE:
#                 if current_NE + disc - self.K_disc[bond] != target_NE:
#                     continue

#             if self.K_disc[bond] == disc or bond == bond_prev:
#                 continue

#             bi = self.net.edgei[bond]
#             bj = self.net.edgej[bond]

#             (condition, meas) = self.solver.solveMeasUpdate(i)

#             if condition < 0.0:
#                 n_zero += 1                    
#                 continue

#             obj = self.obj_func.objFunc(np.concatenate(meas))

#             # print move, obj

#             obj_list.append(obj)
#             valid_move_list.append(i)
#             meas_list.append(meas)


#         if verbose:
#             print "Removing", n_zero, "/", np.sum(self.K_disc), "/",self.net.NE, "bonds would create zero modes..."

#         if len(valid_move_list) == 0:
#             if verbose:
#                 print "No solution found."
#             break


#         args = np.argsort(obj_list)   

#         min_list = []
#         for i in range(len(obj_list)):
#             if obj_list[args[i]] == 0.0:
#                 min_list.append(args[i])
#             else:
#                 break

#         if len(min_list) > 0:
#             rand.shuffle(min_list)
#             index = min_list[0]
#         else:
#             index = args[0]

#         min_move = move_list[valid_move_list[index]]
#         obj_curr = obj_list[index]

#         if verbose:
#             print min_move

#         print obj_curr, meas_list[index]

#         self.K_disc[min_move['bond']] = min_move['disc']
#         K_curr = self.K_max * self.K_disc / self.NDISC

#         if verbose:
#             print n_iter, "Objective function:", obj_curr, "Change:", obj_curr - obj_prev, "Percent:", 100 * (obj_curr - obj_prev) / np.abs(obj_prev), "%"

#         if (obj_curr - obj_prev) / np.abs(obj_prev) > -tol:
#             converge_count += 1
#             print "Steps Backwards", converge_count, "/", self.NCONVERGE
#             if converge_count >= self.NCONVERGE:
#                 if verbose:
#                     print "Stopped making progress."
#                 break
#         else:
#             obj_prev = obj_curr
#             K_prev = np.copy(K_curr)
#             K_disc_prev = np.copy(self.K_disc)
#             converge_count = 0



#         (error, meas) = self.solver.setUpdate(valid_move_list[index])                           

#         if error > max_error:
#             max_error = error

#         bond = min_move['bond']              
#         bi = self.net.edgei[bond]
#         bj = self.net.edgej[bond]

#         K_up = np.min([self.NDISC, min_move['disc']+1])
#         K_down = np.max([0, min_move['disc']-1])

#         replace1 = Update()
#         replace1.setStretchMod(1, [min_move['bond']], [self.K_max[min_move['bond']] * K_up / self.NDISC])

#         replace2 = Update()
#         replace2.setStretchMod(1, [min_move['bond']], [self.K_max[min_move['bond']] * K_down / self.NDISC])

#         move_list[2*min_move['bond']]['disc'] =  K_up
#         move_list[2*min_move['bond']+1]['disc'] =  K_down

#         self.solver.replaceUpdates([2*min_move['bond'], 2*min_move['bond']+1], [replace1.getCyUpdate(), replace2.getCyUpdate()])

#         current_NE = np.sum(self.K_disc)

#         n_iter += 1

#         bond_prev = bond

#     obj = obj_prev
#     K = K_prev

#     # evals = self.solver.getEigenvals()
#     # if verbose:   
#     #     print evals[0:6]

#     result = dict()

#     self.solver.setIntStrengths(K)
#     meas = self.solver.solveMeas()
#     obj_real = self.obj_func.objFunc(meas)

#     res = self.obj_func.projMeas(meas)
#     if verbose:
#         print "Abs Obj Error:", obj - obj_real


#         print "Init Measure:", meas_init
#         print "Final Measure:", meas
#     meas_final = meas

#     if verbose:
#         print "Rel Change:", (meas_final - meas_init) / meas_init
#         print "Abs Change:", meas_final - meas_init

#     result['niter'] = n_iter
#     # result['min_eval'] = evals[3]
#     result['K'] = K
#     result['K_disc'] = K_disc_prev
#     result['NR'] = self.net.NE - np.sum(K_disc_prev)
#     result['DZ_final'] = 2.0 * np.sum(K_disc_prev) / self.net.NN - 2.0 * (self.net.DIM - 1.0 * self.net.DIM / self.net.NN)
#     result['obj_err'] = obj - obj_real
#     result['obj_func'] = obj
#     # result['obj_func_terms'] = obj_terms
#     result['max_error'] = max_error
#     result['condition'] = self.solver.getConditionNum()
#     result['obj_res'] = res

#     if obj == 0.0:
#         result['success_flag'] = 0
#         result['result_msg'] = "Valid solution found."
#     else:
#         result['success_flag'] = 1
#         result['result_msg'] = "No valid solution found."

#     return result   

    