import sys
import mech_network_solver as mns
import network
import numpy as np
import numpy.random as rand
import numpy.linalg as la
import scipy.optimize as spo
import time as time
import copy
import network


class Perturb(object):
    def __init__(self):
        self.NIstrain = 0
        self.istrain_nodesi = np.array([0], dtype=np.int32)
        self.istrain_nodesj = np.array([0], dtype=np.int32)
        self.istrain_bonds = np.array([0], dtype=np.int32)
        self.istrain = np.array([0], dtype=np.double)
        self.istrain_vec = np.array([0], dtype=np.double) 
        
        self.NIstress = 0
        self.istress_bonds = np.array([0], dtype=np.int32)
        self.istress = np.array([0], dtype=np.double)
        
        self.apply_affine_strain = False
        self.strain_tensor = np.array([0], dtype=np.double)
        
        self.apply_affine_stress = False
        self.stress_tensor = np.array([0], dtype=np.double)
        
        self.NFix = 0
        self.fixed_nodes = np.array([0], dtype=np.int32)

    def setInputStrain(self, NIstrain, istrain_nodesi, istrain_nodesj, istrain_bonds, istrain, istrain_vec):
        self.NIstrain = NIstrain
        self.istrain_nodesi = istrain_nodesi
        self.istrain_nodesj = istrain_nodesj
        self.istrain_bonds = istrain_bonds
        self.istrain = istrain
        self.istrain_vec = istrain_vec
        
    def setInputStress(self, NIstress, istress_bonds, istress):
        self.NIstress = NIstress
        self.istress_bonds = istress_bonds
        self.istress = istress
        
    def setAffineStrain(self, apply_affine_strain, strain_tensor):
        self.apply_affine_strain = apply_affine_strain
        self.strain_tensor = strain_tensor
        
    def setAffineStress(self, apply_affine_stress, stress_tensor):
        self.apply_affine_stress = apply_affine_stress
        self.stress_tensor = stress_tensor
        
    def setFixedNodes(self, NFix, fixed_nodes):
        self.NFIx = NFix
        self.fixed_nodes = fixed_nodes
        
    def getCyPert(self):
        return mns.CyPerturb(self.NIstrain, self.istrain_nodesi, self.istrain_nodesj, self.istrain_bonds,
                             self.istrain, self.istrain_vec,
                             self.NIstress, self.istress_bonds, self.istress,
                             self.apply_affine_strain, self.strain_tensor, self.apply_affine_stress, self.stress_tensor,
                            self.NFix, self.fixed_nodes)
    
class Measure(object):
    def __init__(self):
        
        self.NOstrain = 0
        self.ostrain_nodesi = np.array([0], dtype=np.int32)
        self.ostrain_nodesj = np.array([0], dtype=np.int32)
        self.ostrain_bonds = np.array([0], dtype=np.int32)
        self.ostrain_vec = np.array([0], dtype=np.double) 
        
        self.NOstress = 0
        self.ostress_bonds = np.array([0], dtype=np.int32)
                
        self.measure_affine_strain = False
        self.affine_strain_target = np.array([0], dtype=np.double) 
        
        self.measure_affine_stress = False
        self.affine_stress_target = np.array([0], dtype=np.double) 
        
    def setOutputStrain(self, NOstrain, ostrain_nodesi, ostrain_nodesj, ostrain_bonds, ostrain_vec):
        self.NOstrain = NOstrain
        self.ostrain_nodesi = ostrain_nodesi
        self.ostrain_nodesj = ostrain_nodesj
        self.ostrain_bonds = ostrain_bonds
        self.ostrain_vec = ostrain_vec
        
    def setOutputStress(self, NOstress, ostress_bonds):
        self.NOstress = NOstress
        self.ostress_bonds = ostress_bonds
        
    def setAffineStrain(self, measure_affine_strain, affine_strain_target):
        self.measure_affine_strain = measure_affine_strain
        self.affine_strain_target = affine_strain_target
        
    def getCyMeas(self):
        return mns.CyMeasure(self.NOstrain, self.ostrain_nodesi, self.ostrain_nodesj, self.ostrain_bonds, self.ostrain_vec,
                              self.NOstress, self.ostress_bonds, self.measure_affine_strain, self.measure_affine_stress)
    

    
class Update(object):
    def __init__(self):
        self.NSM = 0
        self.sm_bonds = np.array([0], dtype=np.int32)
        self.stretch_mod = np.array([0], dtype=np.double)
        
    def setStretchMod(self, NSM, sm_bonds, stretch_mod):
        self.NSM = NSM
        self.sm_bonds = sm_bonds
        self.stretch_mod = stretch_mod
        
    def getCyUpdate(self):
        return mns.CyUpdate(self.NSM, self.sm_bonds, self.stretch_mod)
    
    
    
    
    
    
    
class TuneDiscLin(object):
    def __init__(self, net, pert, meas, obj_func, K_max, K_disc, NDISC=1, NCONVERGE=1, fix_NE=False):
        
        self.net = net
        self.pert = pert
        self.meas = meas
        
        cypert = []
        for p in pert:
            cypert.append(p.getCyPert())
            
        cymeas = []
        for m in meas:
            cymeas.append(m.getCyMeas())
        
        self.solver = mns.CyLinSolver(net.getCyNetwork(), len(cypert), cypert, cymeas)
                
        self.obj_func = obj_func
        
        self.K_max = np.copy(K_max)
        
        self.K_disc = np.copy(K_disc)
        
        self.NDISC = NDISC
        
        self.NCONVERGE = NCONVERGE
        self.fix_NE = fix_NE
        
        
    def func(self, K):
        
        self.solver.setIntStrengths(K)
        
        meas = self.solver.solveMeas()
        
        return self.obj_func.objFunc(meas)
    
    def func_terms(self, K):
        
        self.solver.setIntStrengths(K)
        
        meas = self.solver.solveMeas()
        
        return self.obj_func.objFuncTerms(meas)
      
        
        
    def tune(self, verbose=True):
                
        #Set initial response ratio
        K_curr = self.K_max * self.K_disc
        self.solver.setIntStrengths(K_curr)
        meas = self.solver.solveMeas()
        
        meas_init = meas
        
        # print meas_init
        
        self.obj_func.setRatioInit(meas)
        
        # Calculate initial response
        obj_prev = self.func(K_curr)
        obj_curr = obj_prev
        K_prev = np.copy(K_curr)
        K_disc_prev = np.copy(self.K_disc)
        
        if verbose:
            print "Initial objective function:", obj_prev        
        
        move_list = []
        for b in range(self.net.NE):
            move_list.append({'bond': b, 'disc': np.min([self.NDISC, self.K_disc[b]+1])})
            move_list.append({'bond': b, 'disc': np.max([0, self.K_disc[b] - 1])})
            
        
        up_list = []
        for move in move_list:
            up = Update()
            up.setStretchMod(1, [move['bond']], [self.K_max[move['bond']] * move['disc'] / self.NDISC])
            up_list.append(up.getCyUpdate())

        self.solver.prepareUpdateList(up_list)
        
        
        # tol = np.sqrt(np.finfo(float).eps)
        # tol = 1e-10
        tol = 1e-8
        
        rem_set = set()
        
        max_error = 0
        
        n_iter = 0
        converge_count = 0
        target_NE = np.sum(self.K_disc)
        current_NE = np.sum(self.K_disc)
        
        bond_prev = -1
        
        while True:
            
            if obj_curr == 0.0:
                break
            
            obj_list = []
            valid_move_list = []
            meas_list = []
            
            n_zero = 0

            for i, move in enumerate(move_list):
                    
                    
                bond = move['bond']
                disc = move['disc']
                
                if self.fix_NE and current_NE != target_NE:
                    if current_NE + disc - self.K_disc[bond] != target_NE:
                        continue
                
                if self.K_disc[bond] == disc or bond == bond_prev:
                    continue
                    
                bi = self.net.edgei[bond]
                bj = self.net.edgej[bond]
                
                (condition, meas) = self.solver.solveMeasUpdate(i)
                
    
                if condition < 0.0:
                    n_zero += 1                    
                    continue
                    
                obj = self.obj_func.objFunc(np.concatenate(meas))
                    
                # print move, obj
                    
                obj_list.append(obj)
                valid_move_list.append(i)
                meas_list.append(meas)
            
            
            if verbose:
                print "Removing", n_zero, "/", np.sum(self.K_disc), "/",self.net.NE, "bonds would create zero modes..."
                        
            if len(valid_move_list) == 0:
                if verbose:
                    print "No solution found."
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
            
            min_move = move_list[valid_move_list[index]]
            obj_curr = obj_list[index]
            
            if verbose:
                print min_move
            
            # print obj_curr, meas_list[index]
            
            self.K_disc[min_move['bond']] = min_move['disc']
            K_curr = self.K_max * self.K_disc / self.NDISC
                
            if verbose:
                print n_iter, "Objective function:", obj_curr, "Change:", obj_curr - obj_prev, "Percent:", 100 * (obj_curr - obj_prev) / np.abs(obj_prev), "%"
            
            if (obj_curr - obj_prev) / np.abs(obj_prev) > -tol:
                converge_count += 1
                print "Steps Backwards", converge_count, "/", self.NCONVERGE
                if converge_count >= self.NCONVERGE:
                    if verbose:
                        print "Stopped making progress."
                    break
            else:
                obj_prev = obj_curr
                K_prev = np.copy(K_curr)
                K_disc_prev = np.copy(self.K_disc)
                converge_count = 0
                
            
                
            (error, meas) = self.solver.setUpdate(valid_move_list[index])                           
                
            if error > max_error:
                max_error = error
                        
            bond = min_move['bond']              
            bi = self.net.edgei[bond]
            bj = self.net.edgej[bond]
            
            K_up = np.min([self.NDISC, min_move['disc']+1])
            K_down = np.max([0, min_move['disc']-1])
                            
            replace1 = Update()
            replace1.setStretchMod(1, [min_move['bond']], [self.K_max[min_move['bond']] * K_up / self.NDISC])
            
            replace2 = Update()
            replace2.setStretchMod(1, [min_move['bond']], [self.K_max[min_move['bond']] * K_down / self.NDISC])
            
            move_list[2*min_move['bond']]['disc'] =  K_up
            move_list[2*min_move['bond']+1]['disc'] =  K_down
            
            self.solver.replaceUpdates([2*min_move['bond'], 2*min_move['bond']+1], [replace1.getCyUpdate(), replace2.getCyUpdate()])
            
            current_NE = np.sum(self.K_disc)
            
            n_iter += 1
            
            bond_prev = bond
            
        obj = obj_prev
        K = K_prev
        
        # evals = self.solver.getEigenvals()
        # if verbose:   
        #     print evals[0:6]
                                
        result = dict()
        
        self.solver.setIntStrengths(K)
        meas = self.solver.solveMeas()
        obj_real = self.obj_func.objFunc(meas)
        
        res = self.obj_func.projMeas(meas)
        if verbose:
            print "Abs Obj Error:", obj - obj_real
        
        
            print "Init Measure:", meas_init
            print "Final Measure:", meas
        meas_final = meas
        
        if verbose:
            print "Rel Change:", (meas_final - meas_init) / meas_init
            print "Abs Change:", meas_final - meas_init
        
        result['niter'] = n_iter
        # result['min_eval'] = evals[3]
        result['K'] = K
        result['K_disc'] = K_disc_prev
        result['NR'] = self.net.NE - np.sum(K_disc_prev)
        result['DZ_final'] = 2.0 * np.sum(K_disc_prev) / self.net.NN - 2.0 * (self.net.DIM - 1.0 * self.net.DIM / self.net.NN)
        result['obj_err'] = obj - obj_real
        result['obj_func'] = obj
        # result['obj_func_terms'] = obj_terms
        result['max_error'] = max_error
        result['condition'] = self.solver.getConditionNum()
        result['obj_res'] = res
        
        if obj == 0.0:
            result['success_flag'] = 0
            result['result_msg'] = "Valid solution found."
        else:
            result['success_flag'] = 1
            result['result_msg'] = "No valid solution found."
        
        return result
    
    
    
    
    
#     def tuneEdge(self):
                
#         #Set initial response ratio
#         self.solver.setIntStrengths(self.K_init)
        
# #         (disp, strain) = self.solver.solveDOF()
# #         disp = disp[0]
        
        
# #         for b in range(self.nw.NE):
# #             bi = self.nw.edgei[b]
# #             bj = self.nw.edgej[b]
# #             posi = self.nw.node_pos[self.nw.DIM*bi:self.nw.DIM*bi+self.nw.DIM]
# #             posj = self.nw.node_pos[self.nw.DIM*bj:self.nw.DIM*bj+self.nw.DIM]
# #             bvec = posj - posi
# #             bvec -= np.round(bvec/self.nw.L)*self.nw.L
            
# #             l0 = la.norm(bvec)
# #             bvec /= l0
            
# #             e = bvec.dot(disp[self.nw.DIM*bj:self.nw.DIM*bj+self.nw.DIM] - disp[self.nw.DIM*bi:self.nw.DIM*bi+self.nw.DIM])/l0
            
# #             print b, e
        
        
#         meas = self.solver.solveMeas()
        
#         print "true init", meas
        
        
#         meas =  np.concatenate(self.solver.initEdgeCalc())
                
#         print "init", meas
                
#         self.obj_func.setRatioInit(meas)

#         # Calculate initial response
#         K_curr = np.copy(self.K_init)
#         obj_prev = self.obj_func.objFunc(meas)
#         obj_curr = obj_prev
        
#         print "Initial objective function:", obj_prev        
        
#         valid_set = set(range(self.nw.NE))
#         for b in self.pert[0].istrain_bonds:
#             valid_set.remove(b)
#         for b in self.meas[0].ostrain_bonds:
#             valid_set.remove(b)        
        
        
#         # tol = np.sqrt(np.finfo(float).eps)
#         # tol = 1e-10
#         tol = 1e-8        
        
#         max_error = 0
        
#         n_iter = 0
#         while True:
            
#             obj_list = []
#             valid_move_list = []
#             meas_list = []
            
#             n_zero = 0
            
#             zero_set = set()
#             for b in sorted(valid_set):
                
#                 (condition, meas) = self.solver.solveEdgeUpdate(b)
    
#                 if condition < 0.0:
#                     print "Inverting Detected Zero", b
#                     n_zero += 1
#                     zero_set.add(b)
                    
#                     continue
                    
#                 obj = self.obj_func.objFunc(np.concatenate(meas))
                    
#                 # print b, meas, obj
                    
#                 obj_list.append(obj)
#                 valid_move_list.append(b)
#                 meas_list.append(np.concatenate(meas))
            
#             # print obj_list
            
#             # print zero_set
            
#             print "Removing", n_zero, "/", len(valid_set), "/", self.nw.NE, "bonds would create zero modes..."
            
#             if len(valid_move_list) == 0:
#                 print "No solution found."
#                 break
            
                
#             args = np.argsort(obj_list)   
            
#             min_list = []
#             for i in range(len(obj_list)):
#                 if obj_list[args[i]] < tol:
#                     min_list.append(args[i])
#                 else:
#                     break
                   
#             if len(min_list) > 0:
#                 rand.shuffle(min_list)
#                 index = min_list[0]
#             else:
#                 index = args[0]
            
#             bmin = valid_move_list[index]
#             obj_min = obj_list[index]
            
#             print "Removing", bmin
            
#             print n_iter, "Objective function:", obj_min, "Change:", obj_min - obj_prev
            
#             if (obj_min - obj_prev) / np.abs(obj_prev) > -tol:
#                 print "Stopped making progress."
#                 break
                         
#             obj_curr = obj_min
#             obj_prev = obj_curr
            
#             K_curr[bmin] = 0.0            
            
#             (error, meas) = self.solver.removeEdge(bmin)
#             # obj = self.obj_func.objFunc(np.concatenate(meas))
#             # print obj_min, obj
            
# #             print meas_list[index]
            
#             print meas
            
#             if error > max_error:
#                 max_error = error
            
#             valid_set.remove(bmin)
            
#             if obj_curr < tol**2:
#                 break
            
          
            
#             n_iter += 1
            
        
        
#         self.solver.setIntStrengths(K_curr)
#         meas = self.solver.solveMeas()
        
#         obj = self.func(K_curr)
        
#         print "obj", obj
#         print "meas", meas
        
        
        
#         evals = self.solver.getEigenvals()
            
#         print evals[0:6]
                    
#         result = dict()

        
#         result['min_eval'] = evals[3]
#         result['K'] = K_curr
#         result['obj_func'] = obj_curr
#         # result['obj_func_terms'] = obj_terms
#         result['error'] = max_error

#         if obj_curr < tol:
#             result['success_flag'] = 0
#             result['result_msg'] = "Valid solution found."
#         else:
#             result['success_flag'] = 1
#             result['result_msg'] = "No valid solution found."

#         print "Result:"
#         print result
        
#         return result
    
    
    
    
    
    
    
# class IneqRatioChangeObjFunc(object):
#     def __init__(self, Nterms, Ngrad, ratio_init, delta_ratio_target):
        
#         self.Nterms = Nterms
#         self.Ngrad = Ngrad
#         self.ratio_init = np.array(ratio_init)
#         self.delta_ratio_target = np.array(delta_ratio_target)
        
#     def getCyObjFunc(self):
#         return mns.CyIneqRatioChangeObjFunc(self.Nterms, self.Ngrad, self.ratio_init, self.delta_ratio_target)
       
    
    
    
    
    
class TuneContLin(object):
    def __init__(self, nw, pert, meas, obj_func, K_init, alg="ADAM"):
        
        self.nw = nw
        
        cypert = []
        for p in pert:
            cypert.append(p.getCyPert())
            
        cymeas = []
        for m in meas:
            cymeas.append(m.getCyMeas())
        
        self.solver = mns.CyLinSolver(nw.getCyNetwork(), len(cypert), cypert, cymeas)
        
        self.obj_func = obj_func
        
        self.K_init = K_init
        
        self.alg = alg
        
       
    
    
    def func(self, K):
        
        self.solver.setIntStrengths(K)
                
        meas = self.solver.solveMeas()
        
        obj = self.obj_func.objFunc(meas)
                
        return obj
    
    def grad(self, K):
        
        self.solver.setIntStrengths(K)
        
        (meas, grad) = self.solver.solveMeasGrad()
        
        return self.obj_func.objFuncGrad(meas, grad)
    
    def func_terms(self, K):
                
        self.solver.setIntStrengths(K)
        
        meas = self.solver.solveMeas()
        
        return self.obj_func.objFuncTerms(meas)
        

    def projMeas(self, K):
        
        self.solver.setIntStrengths(K)
        
        meas = self.solver.solveMeas()
        
        return self.obj_func.projMeas(meas)
    
    
    def projGrad(self, K):
        
        self.solver.setIntStrengths(K)
        
        (meas, grad) = self.solver.solveMeasGrad()
        
        return self.obj_func.projGrad(meas, grad)
    
    def evals(self, K):
        self.solver.setIntStrengths(K)
        
        evals = self.solver.getEigenvals()
        
        return evals
      
    def minimizeGrad(self, K_curr):
        
        max_it = 1000000
        
        x = K_curr.copy()
        obj_prev = self.func(x)
        obj = self.func(x)
        
        upper = self.K_init
        
        low_bound = 10*np.finfo(float).eps
        # low_bound = 0.1
        lower = low_bound * self.K_init
        
        tol = np.sqrt(np.finfo(float).eps)
        
        msg = "Hit max iter: {:d}".format(max_it)
        flag = 2
        
        niter = 0
        
        def proj(x0):
            over = np.where(x0 > upper)[0]
            under = np.where(x0 < lower)[0]

            return np.clip(x0, lower, upper)
            
            
        def projFunc(x0):
            over = np.where(x0 > upper)[0]
            under = np.where(x0 < lower)[0]

            x_proj = np.clip(x0, lower, upper)
            
            # print self.func(x_proj)
            return self.func(x_proj)
        
        
        def projGrad(x0):
            over = np.where(x0 >= upper)[0]
            under = np.where(x0 <= lower)[0]

            x_proj = np.clip(x0, lower, upper)
            pgrad = self.grad(x_proj)

            pgrad[over[pgrad[over] < 0]] = 0.0
            pgrad[under[pgrad[under] > 0]] = 0.0
            
            return pgrad
        
        def goldenSearch(x0, p):
            phi = (np.sqrt(5.0) + 1.0) / 2.0
            
            a = 0
            fa = projFunc(x0)
            b = 0.1
            fb = projFunc(x0+b*p)
            
            f0 = fa
            
            # print "Init: ", a, c, d, b
            # print fa, fc, fd, fb
            
            niter = 0
            
            # if fa > fb:
            #     return b
                        
#             while fa > fb:
#                 niter += 1
#                 b *= 2.0
#                 fb = projFunc(x0+b*p)
                
#                 print "Expanding: ", a, b, fa, fb
            
            c = b - (b - a) / phi
            d = a + (b - a) / phi 
            
            # while not ((c > 1e-3 and np.abs(c - d) < 1e-4) or c < 10*np.finfo(float).eps):
            while np.abs(c - d) > 10*np.finfo(float).eps:
                niter += 1
                fc = projFunc(x0+c*p)
                fd = projFunc(x0+d*p)
                
                if fc > fa and fd > fa:
                    # print "a"
                    b = c
                    fb = fc
                elif fc <= fd:
                    # print "b"
                    b = d
                    fb = fd
                else:
                    # print "c"
                    a = c
                    fa = fc

                c = b - (b - a) / phi
                d = a + (b - a) / phi
                
                # print "Searching: ", a, b, fa, fb
                
            print "Line Search: ", niter, (b + a) / 2, a, b, fa, fb
                
            alpha = (b + a) / 2.0
            if projFunc(x0+alpha*p) < f0:
                
                return (True, alpha)
            else:
                
                return (False, alpha)
            
            
        x_prev = x
        dx_prev = -self.grad(x)
        p_prev = dx_prev
        (success, alpha) = goldenSearch(x_prev, dx_prev)
        
        x += alpha * p_prev
            
        over = np.where(x > upper)[0]
        under = np.where(x < lower)[0]

        x = np.clip(x, lower, upper)
        
        for t in range(max_it):

            
            niter += 1
                     
            grad = self.grad(x)
            dx = -grad
            beta = dx.dot(dx - dx_prev) / dx_prev.dot(dx_prev)
            
            p = dx + beta * p_prev
            
            # (alpha, fc, gc, new_fval, old_fval, new_slope) = spo.line_search(ProjFunc, ProjGrad, 
            #                                                                  x, -grad, gfk=grad, 
            #                                                                  old_fval=obj,
            #                                                                 amax = 0.1, c1=1e-8, c2=0.99)
            
            
            # print alpha, fc, gc, new_fval, old_fval
            
            # if alpha == None:
            #     print "Fail"
            #     return
            
            (success, alpha) = goldenSearch(x, p)
            
            if not success:
                print alpha
                print "CG direction failed, using GD..."
                p = -grad
                dx = -grad
                (success, alpha) = goldenSearch(x, p)
                if not success:
                    print "GD max precision"
                
                
            x += alpha * p
            
            over = np.where(x > upper)[0]
            under = np.where(x < lower)[0]

            x = np.clip(x, lower, upper)
            
            x_prev = x
            dx_prev = dx
            p_prev = p
            
            
            pgrad = grad.copy()

            pgrad[over[pgrad[over] < 0]] = 0.0
            pgrad[under[pgrad[under] > 0]] = 0.0

            # break if energy is zero or projected gradient
            # is zero
            
            imax = np.argmax(np.abs(pgrad))
            pgrad_max = np.abs(pgrad)[imax]
            
            obj_prev = obj
            (obj, obj_terms) = self.func_terms(x)
            
            print "Step: ", t, alpha
            print "PGrad: ", imax, pgrad_max, la.norm(pgrad)
            print "Obj: ", obj
            print "Eval: ", self.evals(x)[2:8]
            
            
            if pgrad_max < tol:
                msg = "Projected gradient tolerance satisfied: max(abs(proj_grad)) < {1:.1E}".format(pgrad_max, tol)
                flag = 1
                break
            
            elif (obj_terms < 10*np.finfo(float).eps).all():
                msg = "Objective function converged to zero."
                flag = 0
                break
               
                                
        return (x, tol, msg, flag, niter)
    
    def minimizeModFIRE(self, K_curr):
        
        N_min = 5
        f_inc = 1.01
        f_dec = 0.5
        alpha_start = 0.1
        f_alpha = 0.99
        dt_max = 0.3
        max_it = 1000000
        
        x = K_curr.copy()
        v = np.zeros(self.nw.NE, float)
        grad = self.grad(x)
        obj_prev = self.func(x)
        
        dt = dt_max / 10
        N_neg = 0
        alpha = alpha_start
        
        upper = self.K_init
        
        # low_bound = 10*np.finfo(float).eps
        low_bound = 1e-2
        lower = low_bound * self.K_init
        
        tol = np.sqrt(np.finfo(float).eps)
        
        msg = "Hit max iter: {:d}".format(max_it)
        flag = 2
        
        niter = 0
        for t in range(max_it):
            niter += 1
                        
            # step positions
            x = x + v*dt + 0.5*(-grad)*dt**2.0
            
            # apply boundary conditions on positions
            over = np.where(x > upper)[0]
            under = np.where(x < lower)[0]
            x = np.clip(x, lower, upper)
            
            (obj, obj_terms) = self.func_terms(x)
            
            delta_obj = obj - obj_prev
            
            # first velocity half step at old acceleration
            v += 0.5*(-grad)*dt
            # calculate new gradient
            grad = self.grad(x)
            # apply second half of stop
            v += 0.5*(-grad)*dt
            
            #check for convergence
            
            # break if energy is zero or projected gradient
            # is zero
            pgrad = grad.copy()
            pgrad[over[pgrad[over] < 0]] = 0.0
            pgrad[under[pgrad[under] > 0]] = 0.0
            
            
            pgrad_max = np.max(np.abs(pgrad))
           
            if t % 10 == 0:
                print "Step: ", t, obj, delta_obj, dt, N_neg
                ipgrad_max = np.argmax(np.abs(pgrad))
                print "PGrad: ", ipgrad_max, x[ipgrad_max], -pgrad_max, v[ipgrad_max]
            
            
            
            if pgrad_max < tol:
                msg = "Projected gradient tolerance satisfied: max(abs(proj_grad)) < {1:.1E}".format(pgrad_max, tol)
                flag = 1
                break
            
            elif (obj_terms < 10*np.finfo(float).eps).all():
                msg = "Objective function converged to zero."
                flag = 0
                break
                
                
            #continue step
            
            v = (1.0-alpha)*v + alpha * (-grad) / la.norm(-grad) * la.norm(v)
            
            v[over[v[over] > 0]] = 0.0
            v[under[v[under] < 0]] = 0.0
            
            if delta_obj <= 0:
                N_neg += 1
                
                if N_neg > N_min:
                    dt = np.min([dt*f_inc, dt_max])
                    alpha = f_alpha * alpha
                    
                x_prev = x
                v_prev = v
                grad_prev = grad
                obj_prev = obj
                
                # print "accept"
            
            else:
                N_neg = 0
                dt = dt*f_dec
                alpha = alpha_start
                # v = np.zeros(self.nw.NE, float)
                
                v[np.where(v*grad > 0)[0]] = 0
                
                x = x_prev
                grad = grad_prev
                obj = obj_prev
                delta_obj = 0.0
                
            if t % 10 == 0:
                    print "Obj: ", obj 
            
            
        return (x, tol, msg, flag, niter)
    
    
    def minimizeFIRE(self, K_curr):
        
        N_min = 5
        f_inc = 1.01
        f_dec = 0.5
        alpha_start = 0.1
        f_alpha = 0.99
        dt_max = 0.3
        max_it = 1000000
        
        x = K_curr.copy()
        v = np.zeros(self.nw.NE, float)
        grad = self.grad(x)
        obj_prev = self.func(x)
        
        dt = dt_max / 10
        N_neg = 0
        alpha = alpha_start
        
        upper = self.K_init
        
        # low_bound = 10*np.finfo(float).eps
        low_bound = 1e-2
        lower = low_bound * self.K_init
        
        tol = np.sqrt(np.finfo(float).eps)
        
        msg = "Hit max iter: {:d}".format(max_it)
        flag = 2
        
        niter = 0
        for t in range(max_it):
            niter += 1
                        
            # step positions
            x = x + v*dt + 0.5*(-grad)*dt**2.0
            
            # apply boundary conditions on positions
            over = np.where(x > upper)[0]
            under = np.where(x < lower)[0]
            x = np.clip(x, lower, upper)
            
            (obj, obj_terms) = self.func_terms(x)
            
            delta_obj = obj - obj_prev
            
            # first velocity half step at old acceleration
            v += 0.5*(-grad)*dt
            # calculate new gradient
            grad = self.grad(x)
            # apply second half of stop
            v += 0.5*(-grad)*dt
            
            
            
            #check for convergence
            
            # break if energy is zero or projected gradient
            # is zero
            pgrad = grad.copy()
            pgrad[over[pgrad[over] < 0]] = 0.0
            pgrad[under[pgrad[under] > 0]] = 0.0
            
            pv = v.copy()
            pv[over[pv[over] > 0]] = 0.0
            pv[under[pv[under] < 0]] = 0.0
            
            
            P = (-pgrad).dot(pv)
            
            
            pgrad_max = np.max(np.abs(pgrad))
           
            if t % 10 == 0:
                print "Step: ", t, obj, delta_obj, dt, P, N_neg
                ipgrad_max = np.argmax(np.abs(pgrad))
                print "PGrad: ", ipgrad_max, x[ipgrad_max], -pgrad_max, v[ipgrad_max]
            
            
            
            if pgrad_max < tol:
                msg = "Projected gradient tolerance satisfied: max(abs(proj_grad)) < {1:.1E}".format(pgrad_max, tol)
                flag = 1
                break
            
            elif (obj_terms < 10*np.finfo(float).eps).all():
                msg = "Objective function converged to zero."
                flag = 0
                break
                
                
            #continue step
            v[over[v[over] > 0]] = 0.0
            v[under[v[under] < 0]] = 0.0
            
            
            v = (1.0-alpha)*v + alpha * (-grad) / la.norm(-grad) * la.norm(v)
            
            v[over[v[over] > 0]] = 0.0
            v[under[v[under] < 0]] = 0.0
            
            if P > 0:
                N_neg += 1
                
                if N_neg > N_min:
                    dt = np.min([dt*f_inc, dt_max])
                    alpha = f_alpha * alpha
                    
                x_prev = x
                v_prev = v
                grad_prev = grad
                obj_prev = obj
                obj_min = obj
                
                # print "accept"
            
            else:
                N_neg = 0
                dt = dt*f_dec
                alpha = alpha_start
                v = np.zeros(self.nw.NE, float)
                
                x = x_prev
                grad = grad_prev
                obj = obj_prev
                delta_obj = 0.0
            
            if t % 10 == 0:
                print "Obj: ", obj 
        
        
        return (x, tol, msg, flag, niter)
    
    def minimizeMonoADAM(self, K_curr, alpha=0.001, low_bound=10*np.finfo(float).eps):
        
        # alpha = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        eps = 1e-8
        
        max_it = 100000
        
        m = np.zeros(self.nw.NE, float)
        v = np.zeros(self.nw.NE, float)
        
        x = K_curr.copy()
        
        upper = self.K_init
        
        # low_bound = 10*np.finfo(float).eps
        # low_bound = 1e-3
        lower = low_bound * self.K_init
        
        tol = np.sqrt(np.finfo(float).eps)
        
        msg = "Hit max iter: {:d}".format(max_it)
        flag = 2
        
        niter = 0
           
        updated = True
            
        # for t in range(max_it):
        t = 0
        
        obj_prev = self.func(x)
        x_prev = x
        
        while True:


            niter += 1

            grad = self.grad(x)
            # p = grad + rand.normal(scale=0.01 / (1+t)**0.55, size=self.nw.NE)
            m = beta_1 * m + (1.0 - beta_1) * grad
            v = beta_2 * v + (1.0 - beta_2) * grad**2.0

            mhat = m / (1.0 - beta_1**(t+1))
            vhat = v / (1.0 - beta_2**(t+1))

            x = x - alpha *mhat / (np.sqrt(vhat) + eps) 

            over = np.where(x > upper)[0]
            under = np.where(x < lower)[0]

            x = np.clip(x, lower, upper)

            pgrad = grad.copy()

            pgrad[over[pgrad[over] < 0]] = 0.0
            pgrad[under[pgrad[under] > 0]] = 0.0

            # break if energy is zero or projected gradient
            # is zero

            pgrad_max = np.max(np.abs(pgrad))

            (obj, obj_terms) = self.func_terms(x)
            
            if niter % 1 == 0:
                print "Step: ", t, obj,(1.0 - beta_1**(t+1)),  (1.0 - beta_2**(t+1))
                ipgrad_max = np.argmax(np.abs(pgrad))
                print "PGrad: ", ipgrad_max, -pgrad_max
                
            if obj > obj_prev:
                t = 0
                m = np.zeros(self.nw.NE, float)
                v = np.zeros(self.nw.NE, float)
                x = x_prev
                obj = obj_prev
                
                
            if pgrad_max < tol:
                msg = "Projected gradient tolerance satisfied: max(abs(proj_grad)) < {1:.1E}".format(pgrad_max, tol)
                flag = 1
                break

            elif (obj_terms < 10*np.finfo(float).eps).all():
                msg = "Objective function converged to zero."
                flag = 0
                break
            
            # if t % 10 == 0:
            #     print "Step: ", t, obj,(1.0 - beta_1**(t+1)),  (1.0 - beta_2**(t+1))
            #     ipgrad_max = np.argmax(np.abs(pgrad))
            #     print "PGrad: ", ipgrad_max, -pgrad_max
            #     # print "Eval: ", self.evals(x)[2:8]
            #     # print "XMax: ", np.max(x)
              
            obj_prev = obj
            
            t += 1
                       

        return (x, tol, msg, flag, niter)
    
    
    def minimizeADAM(self, K_curr, alpha=0.001, low_bound=10*np.finfo(float).eps):
        
        # alpha = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        eps = 1e-8
        
        max_it = 100000
        
        m = np.zeros(self.nw.NE, float)
        v = np.zeros(self.nw.NE, float)
        
        x = K_curr.copy()
        
        upper = 1e6*self.K_init
        
        # low_bound = 10*np.finfo(float).eps
        # low_bound = 1e-3
        lower = low_bound * self.K_init
        
        tol = np.sqrt(np.finfo(float).eps)
        
        msg = "Hit max iter: {:d}".format(max_it)
        flag = 2
        
        niter = 0
          
        x_best = x
        obj_best = self.func(x)
        pgrad_max_best = -1.0
        pgrad_max_norm_best = -1.0
           
        updated = True
            
        inactive_set = set()
            
        # for t in range(max_it):
        t = 0
        while True:


            niter += 1

            grad = self.grad(x)
            # p = grad + rand.normal(scale=0.01 / (1+t)**0.55, size=self.nw.NE)
            m = beta_1 * m + (1.0 - beta_1) * grad
            v = beta_2 * v + (1.0 - beta_2) * grad**2.0

            mhat = m / (1.0 - beta_1**(t+1))
            vhat = v / (1.0 - beta_2**(t+1))

            x = x - alpha *mhat / (np.sqrt(vhat) + eps) 

            over = np.where(x > upper)[0]
            under = np.where(x < lower)[0]

            x = np.clip(x, lower, upper)

            pgrad = grad.copy()

            pgrad[over[pgrad[over] < 0]] = 0.0
            pgrad[under[pgrad[under] > 0]] = 0.0

            # break if energy is zero or projected gradient
            # is zero

            pgrad_max = np.max(np.abs(pgrad))
            
            pgrad2 = grad.copy()
            pgrad2[over[pgrad2[over] < 0]] = 1e10
            pgrad2[under[pgrad2[under] > 0]] = 1e10

            (obj, obj_terms) = self.func_terms(x)


            if obj < obj_best:
                x_best = x
                obj_best = obj
                pgrad_max_best = pgrad_max
                pgrad_max_norm_best = la.norm(pgrad)
                updated = True
                
                
                if pgrad_max < tol:
                    msg = "Projected gradient tolerance satisfied: max(abs(proj_grad)) < {1:.1E}".format(pgrad_max, tol)
                    flag = 1
                    break

                elif (obj_terms < 10*np.finfo(float).eps).all():
                    msg = "Objective function converged to zero."
                    flag = 0
                    break
            
            if t % 10 == 0:
                print "Step: ", t, obj,(1.0 - beta_1**(t+1)),  (1.0 - beta_2**(t+1))
                ipgrad_max = np.argmax(np.abs(pgrad))
                print "PGrad Max: ", ipgrad_max, pgrad_max
                print "PGrad Min: ", np.sort(np.abs(pgrad2))[0::10]
                # print "Eval: ", self.evals(x)[2:8]
                # print "XMax: ", np.max(x)

            if t % 1000 == 0:
                print t, "Best:", obj_best, "alpha", alpha, "pgrad_max", pgrad_max_best, "pgrad_norm", pgrad_max_norm_best

                if not updated:
                    break
                else:
                    updated = False
                
            
            t += 1
                       
                        
        print "Best:", obj_best, "alpha", alpha, "pgrad_max", pgrad_max_best, "pgrad_norm", pgrad_max_norm_best
        print "Eval: ", self.evals(x_best)[2:8]
        print "L1Norm: ", np.sum(x_best)
        
        return (x_best, tol, msg, flag, niter, under)


    def minimizeLBFGSB(self, K_curr):
        
        obj_prev = self.func(K_curr)
        
        low_bound = 1e-4
        alpha = 1e-4
        while True:

            tol = np.max([low_bound, 10*np.finfo(float).eps])

            print "Lower bound:", low_bound
            bounds = [(low_bound*k, 1e0*k) for k in self.K_init]   
            t0 = time.time()
            res = spo.minimize(self.func, K_curr, method='L-BFGS-B', callback=None,
                               jac=self.grad, options={'ftol': tol, 'gtol': tol,'maxcor': 10,
                                                              'maxiter':10000000, 'maxfun': 10000000, 'maxls': 10000000},
                              bounds=bounds)
            
#             (x, f, d) = spo.fmin_l_bfgs_b(self.func, K_curr, fprime=self.grad, bounds=bounds,
#                                    m=10, pgtol=10*np.finfo(float).eps, iprint=10, factr=0.0, maxfun=1000000, maxiter=1000000,
#                                    maxls=100)
                
            t1 = time.time()
            print "Time", t1-t0

            K_curr = res.x

            #Calculate final response
            obj_curr = self.func(K_curr)

            print "Objective function:", obj_prev, "-->", obj_curr, "Change:", np.abs(obj_curr - obj_prev)

            if obj_curr < 10*np.finfo(float).eps or np.abs(obj_curr - obj_prev) < 10*np.finfo(float).eps:
                break
            else:
                if alpha*low_bound > 10*np.finfo(float).eps:
                    low_bound *= alpha
                    obj_prev = obj_curr
                else:
                    break
                                
            
        if res.success:
            flag = 1
        else:
            flag = 2
         
        return (K_curr, low_bound, res.message, flag)
    
    
    def minimizeSciPyMin(self, K_curr):
        
        obj_prev = self.func(K_curr)
        
        low_bound = 1e-4
        # low_bound = 1e-4
        alpha = 1e-4
        while True:

            # tol = np.max([low_bound, 10*np.finfo(float).eps])
            
            tol = np.sqrt(np.finfo(float).eps)

            print "Lower bound:", low_bound
            bounds = [(low_bound*k, 1e0*k) for k in self.K_init]   
            t0 = time.time()
            res = spo.minimize(self.func, K_curr, method='L-BFGS-B', callback=None,
                               jac=self.grad, options={'ftol': 0.0, 'gtol': tol,'maxcor': 10, 'disp': True,
                                                              'maxiter':10000000, 'maxfun': 10000000, 'maxls': 10000000},
                              bounds=bounds)
            
            
#             res = spo.minimize(self.func, K_curr, method='TNC', callback=None,
#                                jac=self.grad, options={'ftol': 0.0, 'gtol': tol, 'xtol': tol, 'disp': True,
#                                                               'maxiter':10000000, 'maxCGit':-1, 'rescale': 0},
#                               bounds=bounds)
            
            
            # res = spo.minimize(self.func, K_curr, method='SLSQP', callback=None,
            #                    jac=self.grad, options={'ftol': 0.0, 'disp': True,
            #                                                   'maxiter':10000000},
            #                   bounds=bounds)
            


            t1 = time.time()
            print "Time", t1-t0

            K_curr = res.x
            
            
            print "Func:", self.func(K_curr)
            print "Grad:", self.grad(K_curr)
                

            #Calculate final response
            obj_curr = self.func(K_curr)

            print "Objective function:", obj_prev, "-->", obj_curr, "Change:", np.abs(obj_curr - obj_prev)

            if obj_curr < 10*np.finfo(float).eps or np.abs(obj_curr - obj_prev) < 10*np.finfo(float).eps:
                break
            else:
                if alpha*low_bound > 10*np.finfo(float).eps:
                    low_bound *= alpha
                    obj_prev = obj_curr
                else:
                    break
                    
                 
            break
            
        if res.success:
            flag = 1
        else:
            flag = 2
                  
        return (K_curr, low_bound, res.message, flag)
    
    def minimizeSciPyLS(self, K_curr):
        
        obj_prev = self.func(K_curr)
        
        low_bound = 0.0
        # low_bound = 1e-6
        up_bound = 1e0
        alpha = 1e-1

        # tol = np.sqrt(np.finfo(float).eps)/ 100
        tol = np.sqrt(np.finfo(float).eps)
        
        print "Tolerance:", tol

        print "Lower bound:", low_bound
        bounds = [(low_bound*k, 1e0*k) for k in self.K_init]   
        t0 = time.time()


        for n in range(1):
            print "Gradient Check:", spo.check_grad(self.func, self.grad, K_curr)
            
            
            res = spo.least_squares(self.projMeas, K_curr, jac=self.projGrad, 
                                    bounds=(low_bound*self.K_init, up_bound*self.K_init),
                                   method='trf', ftol=0.0, xtol=tol, gtol=0.0, max_nfev=1000000000, 
                                   verbose=2, tr_solver='lsmr', tr_options={'show':False, 'damp':0.0}, x_scale='jac')


            K_curr = res.x
            
            t1 = time.time()
            print "Time", t1-t0

            print res.active_mask
            
           

            grad = self.grad(K_curr)
            mask = np.where(res.active_mask != 0)

            print mask
            print grad[mask]
            
            pgrad = grad.copy()
            pgrad[mask] = 0.0

            pgrad_max = np.max(np.abs(pgrad))
            print "PGrad Max:", pgrad_max

            (obj, obj_terms) = self.func_terms(K_curr)

            if pgrad_max < tol:
                msg = "Projected gradient tolerance satisfied: max(abs(proj_grad)) < {1:.1E}".format(pgrad_max, tol)
                flag = 1
                break

            elif (obj_terms < 10*np.finfo(float).eps).all():
                msg = "Objective function converged to zero."
                flag = 0
                break
                
                
            # break
                    
            
            
        if res.success:
            flag = 1
        else:
            flag = 2
                  
        return (K_curr, low_bound, res.message, flag)
        
        
    def tune(self):
        #Set initial response ratio
        self.solver.setIntStrengths(self.K_init)
        meas = self.solver.solveMeas()
                
        self.obj_func.setRatioInit(meas)
        
        K_curr = self.K_init.copy()
        
        niter = 0
        if self.alg == 'ADAM':
            # (K_curr, tol, msg, flag, niter) = self.minimizeGrad(K_curr)
            
            alpha = 0.01
            # low_bound = 10*np.finfo(float).eps
            low_bound = 1e0
            
            flag = 2
            
            while flag == 2 and alpha > 10*np.finfo(float).eps:
            
                (K_curr, tol, msg, flag, nit, under) = self.minimizeADAM(K_curr, alpha=alpha, low_bound=low_bound)
                niter += nit
                alpha *= 1e-1 
                
                break
                
                
        elif self.alg == 'LBFGSB':
            # (K_curr, tol, msg, flag) = self.minimizeSciPyLS(K_curr)
            
            (K_curr, tol, msg, flag) = self.minimizeSciPyMin(K_curr)
                    
        (obj, obj_terms) = self.func_terms(K_curr)
        
        if obj < 10*np.finfo(float).eps:
            flag = 0
        
        result = dict()

        result['alg'] = self.alg
        result['K'] = K_curr
        result['obj_func'] = obj
        result['obj_func_terms'] = obj_terms

        # result['rem_set'] = under
            
        print msg
        result['success_flag'] = flag
        result['result_msg'] = msg
        result['tol'] = tol
        result['niter'] = niter
        # result['low_bound'] = low_bound
        # result['NR'] = len(under)
    
        
        #establish quality of minimization
        #calucate final gradient with elements projected out if on boundary
        
        grad = self.grad(K_curr)
        pgrad = np.zeros(self.nw.NE)
        for i in range(self.nw.NE):
            if not ((K_curr[i] - tol*self.K_init[i]) <= np.finfo(float).eps and grad[i] > np.finfo(float).eps):
                pgrad[i] = grad[i]      
        

        result['pgrad_max'] = np.max(np.abs(pgrad))
        result['pgrad_norm'] = la.norm(pgrad)
        
        print result['pgrad_max'], result['pgrad_norm']
        
        # print "Result:"
        # print result
        
        return result
        
    

    
class TuneDiscNonlin(object):
    def __init__(self, nw, pert, meas, obj_func, K_init):
        
        self.nw = nw
        
        cypert = []
        for p in pert:
            cypert.append(p.getCyPert())
            
        cymeas = []
        for m in meas:
            cymeas.append(m.getCyMeas())
        
        self.solver = mns.CyNonlinSolver(nw.getCyNetwork(), len(cypert), cypert, cymeas)
        self.lin_solver = mns.CyLinSolver(nw.getCyNetwork(), len(cypert), cypert, cymeas)
        
        self.obj_func = obj_func
        
        self.K_init = np.copy(K_init)
        
        
    def func(self, K):
        
        self.solver.setIntStrengths(K)
        
        meas = self.solver.solveMeas()
        
        return self.obj_func.objFunc(np.concatenate(meas))
    
    def func_terms(self, K):
        
        self.solver.setIntStrengths(K)
        
        meas = self.solver.solveMeas()
        
        return self.obj_func.objFuncTerms(np.concatenate(meas))
        
    def tune(self):
                
        #Set initial response ratio
        self.solver.setIntStrengths(self.K_init)
        meas = self.solver.solveMeas()
        self.obj_func.setRatioInit(np.concatenate(meas))
        
        # Calculate initial response
        K_prev = np.copy(self.K_init)
        obj_prev = self.func(K_prev)
        obj_curr = obj_prev
        
        print "Initial objective function:", obj_prev        
        
        up_list = []
        for b in range(self.nw.NE):
            up = Update()
            up.setStretchMod(1, [b], [0.0])
            up_list.append(up.getCyUpdate())

        self.lin_solver.prepareUpdateList(up_list)
        
        rem_set = set()
        
        while True:
            
            obj_list = []
            b_list = []
            
            for b in range(self.nw.NE):
                if b % 25 == 0:
                    print b
                    sys.stdout.flush()
                    
                (success, meas) = self.lin_solver.solveMeasUpdate(b)
                
                if not success:
                    print "Removing bond", b, "would create zero mode. Skipping..."
                    sys.stdout.flush()
                    continue
                    
                    
                K_curr = np.copy(self.K_init)
                K_curr[list(rem_set)] = 0.0
                
                if b in rem_set:
                    K_curr[b] = self.K_init[b]
                else:
                    K_curr[b] = 0.0
                
                obj = self.func(K_curr)
                    
                # print b, meas, obj
                    
                obj_list.append(obj)
                b_list.append(b)
            
            # print obj_list
            
            if len(b_list) == 0:
                print "No solution found."
                break
            
                
            args = np.argsort(obj_list)   
            
            min_list = []
            for i in range(len(b_list)):
                if obj_list[args[i]] < 10*np.finfo(float).eps:
                    min_list.append(args[i])
                else:
                    break
                   
            if len(min_list) > 0:
                print min_list
                rand.shuffle(min_list)
                index = min_list[0]
            else:
                index = args[0]
                            
            bmin = b_list[index]
            obj_curr = obj_list[index]
            
            if obj_curr - obj_prev > -10*np.finfo(float).eps:
                print "Stopped making progress."
                break
                
            
            print "Objective function:", obj_curr, "Change:", np.abs(obj_curr - obj_prev)
            sys.stdout.flush()
            
            replace_up = Update()
            if bmin in rem_set:
                print "Adding", bmin
                sys.stdout.flush()
                rem_set.remove(bmin)
                # new update removes bmin
                replace_up.setStretchMod(1, [bmin], [0.0])
            else:
                print "Removing", bmin
                sys.stdout.flush()
                rem_set.add(bmin)
                # new update adds bmin
                replace_up.setStretchMod(1, [bmin], [self.K_init[bmin]])

            if obj_curr < 10*np.finfo(float).eps or np.abs(obj_curr - obj_prev) < 10*np.finfo(float).eps:
                break
            
            self.lin_solver.setUpdate(bmin, replace_up.getCyUpdate())
                
            obj_prev = obj_curr
                

        K_curr = np.copy(self.K_init)
        K_curr[list(rem_set)] = 0.0
        
        (obj, obj_terms) = self.func_terms(K_curr)
                    
        result = dict()

        result['K'] = K_curr
        result['obj_func'] = obj
        result['obj_func_terms'] = obj_terms

        if np.sum(obj) < 10*np.finfo(float).eps:
            result['success_flag'] = 0
            result['result_msg'] = "Valid solution found."
        else:
            result['success_flag'] = 1
            result['result_msg'] = "No valid solution found."

        result['rem_set'] = rem_set
        result['low_bound'] = 0.0

        print "Result:"
        print result
        
        return result
    
    
def chooseRandomEdges(net, NTS):
    
    NF = 1
    
    edgei = net.edgei
    edgej = net.edgej
    
    edge = range(net.NE)
    
    rand.shuffle(edge)
    
    inodesi = [[] for t in range(NF)]
    inodesj = [[] for t in range(NF)]
    
    b = edge.pop()

    inodesi[0].append(edgei[b])
    inodesj[0].append(edgej[b])

    onodesi = [[] for t in range(NF)]
    onodesj = [[] for t in range(NF)]
    
    for i in range(NTS):
        b = edge.pop()
    
        onodesi[0].append(edgei[b])
        onodesj[0].append(edgej[b])
        
    return (inodesi, inodesj, onodesi, onodesj)
    
#only works for one tuned function
def chooseFeasibleEdges(net, eta, NTS):

    DIM = net.DIM
    
    NF = 1

    edgei = net.edgei
    edgej = net.edgej
    
    edge = range(net.NE)
    
    rand.shuffle(edge)
    
    inodesi = [[] for t in range(NF)]
    inodesj = [[] for t in range(NF)]
    
    b = edge.pop()

    inodesi[0].append(edgei[b])
    inodesj[0].append(edgej[b])

    onodesi_curr = [[] for t in range(NF)]
    onodesj_curr = [[] for t in range(NF)]
    
    while len(edge) > 0:
        
        onodesi = copy.deepcopy(onodesi_curr)
        onodesj = copy.deepcopy(onodesj_curr)
        
        b = edge.pop()

        onodesi[0].append(edgei[b])
        onodesj[0].append(edgej[b])

#         print "inodes", inodesi, inodesj

#         print "onodes", onodesi, onodesj

        isvec = [[] for t in range(NF)]
        for t in range(NF):
            for (i, j) in zip(inodesi[t], inodesj[t]):
                posi = net.node_pos[DIM*i:DIM*i+DIM]
                posj = net.node_pos[DIM*j:DIM*j+DIM]
                bvec = posj - posi
                bvec -= np.round(bvec/net.L)*net.L
                isvec[t].extend(bvec) 

        istrain = [[] for t in range(NF)]
        istrain[0].append(1.0)

        osvec = [[] for t in range(NF)]
        ostrain = [[] for t in range(NF)]
        for t in range(NF):
            for (i, j) in zip(onodesi[t], onodesj[t]):
                posi = net.node_pos[DIM*i:DIM*i+DIM]
                posj = net.node_pos[DIM*j:DIM*j+DIM]
                bvec = posj - posi
                bvec -= np.round(bvec/net.L)*net.L
                osvec[t].extend(bvec) 

                r = rand.randint(2)
                ostrain[t].append((2*r-1) * eta)

        pert = []
        meas = []
        for t in range(NF):
            pert.append(Perturb())
            pert[t].setInputStrain(len(inodesi[t]), inodesi[t], inodesj[t], istrain[t], isvec[t])

            meas.append(Measure())
            meas[t].setOutputStrain(len(onodesi[t]), onodesi[t], onodesj[t], osvec[t])

        obj_func = mns.CyIneqRatioChangeObjFunc(len(np.concatenate(ostrain)), net.NE, 
                                                np.zeros(len(np.concatenate(ostrain)), float), np.concatenate(ostrain))    


        K_init = np.ones(net.NE, float) / net.eq_length


        cypert = []
        for p in pert:
            cypert.append(p.getCyPert())

        cymeas = []
        for m in meas:
            cymeas.append(m.getCyMeas())

        solver = mns.CyLinSolver(net.getCyNetwork(), len(cypert), cypert, cymeas)
        solver.setIntStrengths(K_init)

        (feasible, u, con_err) = solver.solveFeasability(obj_func)
        
        # print feasible
        # print con_err
        
        if feasible[0]:
            # print "Success:", b, "Remaining:", len(edge)
            
            onodesi_curr = copy.deepcopy(onodesi)
            onodesj_curr = copy.deepcopy(onodesj)
            
            if len(onodesi[0]) == NTS:
                
                return (True, inodesi, inodesj, onodesi_curr, onodesj_curr)            
            
        else:
            # print "Fail:", b, "Remaining:", len(edge)
            pass
            
    return (False, inodesi, inodesj, onodesi_curr, onodesj_curr)