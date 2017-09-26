print "Loading Mechanical Network Deformation Solver Module"

import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "network.hpp":
    cdef cppclass Network:
        Network() except +
        Network(int NN, vector[double] &node_pos, 
                int NE, vector[int] &edgei, vector[int] &edgej,
                int NGDOF, vector[double] &L, bool enable_affine,
                vector[double] &bvecij, vector[double] &eq_length, vector[double] &stretch_mod) except +
        
cdef extern from "perturb.hpp":
    cdef cppclass Perturb:
        Perturb() except +
        Perturb(int NIstrain, vector[int] &istrain_nodesi, vector[int] &istrain_nodesj,
                vector[int] &istrain_bonds,
                vector[double] &istrain, vector[double] &istrain_vec,
                int NIstress, vector[int] &istress_bonds, vector[double] &istress,
                bool apply_affine_strain, vector[double] &strain_tensor, 
                bool apply_affine_stress, vector[double] &stress_tensor,
               int NFix, vector[int] &fixed_nodes) except +
        
cdef extern from "measure.hpp":
    cdef cppclass Measure:
        Measure() except +
        Measure(int NOstrain, vector[int] &ostrain_nodesi, vector[int] &ostrain_nodesj, 
                vector[int] &ostrain_bonds,
                vector[double] &ostrain_vec,
                int NOstress, vector[int] &ostress_bonds,
                bool measure_affine_strain, bool measure_affine_stress) except +

cdef extern from "abstract_objective_function.hpp":
    cdef cppclass AbstractObjFunc:
        AbstractObjFunc() except+

cdef extern from "objective_function.hpp":
    
    cdef cppclass AugIneqRatioChangeObjFunc:
        AugIneqRatioChangeObjFunc() except +
        AugIneqRatioChangeObjFunc(vector[double] &delta_ratio_target) except +
        
        void initialize(LinSolver &solver, vector[double] &x_init)
        void setWeights(double a, double b, double c)
        void setRegularize(double mu, vector[double] &x_reg)
        void res(vector[double] &x, LinSolver &solver, vector[double] &res)
        void resGrad(vector[double] &x, LinSolver &solver, vector[double] &res,
                                        vector[int] &rows, vector[int] &cols, vector[double] &vals)
        
        void func(vector[double] &x, LinSolver &solver, double &obj)
        void funcGrad(vector[double] &x, LinSolver &solver, double &obj, vector[double] &obj_grad)
        void funcHess(vector[double] &x, LinSolver &solver,
                                        vector[int] &rows, vector[int] &cols, vector[double] &vals)
    
    cdef cppclass IneqRatioChangeObjFunc(AbstractObjFunc):
        IneqRatioChangeObjFunc() except +
        IneqRatioChangeObjFunc(int Nterms, int Ngrad, vector[double] &ratio_init, 
                               vector[double] &delta_ratio_target) except +
        
        void setRatioInit(vector[double] &ratio_init)
        void objFuncTerms(vector[double] &meas, double &obj, vector[double] &terms)
        void objFunc(vector[double] &meas, double &obj)
        void objFuncGrad(vector[double] &meas, vector[vector[double]] &meas_grad,
                                           vector[double] &obj_grad)
        
        void projMeas(vector[double] &meas, vector[double] &pmeas)
        void projGrad(vector[double] &meas, vector[vector[double]] &meas_grad,
                                           vector[vector[double]] &pgrad)
        
        
        void res(AbstractSolver &solver, vector[double] &res)
        void resGrad(AbstractSolver &solver, vector[double] &res, 
                             vector[vector[double] ] &res_grad)
        void func(AbstractSolver &solver, double &obj)
        void funcGrad(AbstractSolver &solver, double &obj, vector[double] &obj_grad)
        void funcHess(AbstractSolver &solver, double &obj, vector[double] &obj_grad, 
                              vector[vector[double] ] &obj_hess)
        
    cdef cppclass EqRatioChangeObjFunc(AbstractObjFunc):
        EqRatioChangeObjFunc() except +
        EqRatioChangeObjFunc(int Nterms, int Ngrad, vector[double] &ratio_init, 
                               vector[double] &delta_ratio_target) except +
        
        void setRatioInit(vector[double] &ratio_init)
        void objFuncTerms(vector[double] &meas, double &obj, vector[double] &terms)
        void objFunc(vector[double] &meas, double &obj)
        void objFuncGrad(vector[double] &meas, vector[vector[double]] &meas_grad,
                                           vector[double] &obj_grad)
        
        void projMeas(vector[double] &meas, vector[double] &pmeas)
        void projGrad(vector[double] &meas, vector[vector[double]] &meas_grad,
                                           vector[vector[double]] &pgrad)
        
        
    cdef cppclass IneqRatioObjFunc(AbstractObjFunc):
        IneqRatioObjFunc() except +
        IneqRatioObjFunc(int Nterms, int Ngrad, vector[double] &ratio_init, 
                               vector[double] &delta_ratio_target) except +
        
        void setRatioInit(vector[double] &ratio_init)
        void objFuncTerms(vector[double] &meas, double &obj, vector[double] &terms)
        void objFunc(vector[double] &meas, double &obj)
        void objFuncGrad(vector[double] &meas, vector[vector[double]] &meas_grad,
                                           vector[double] &obj_grad)
        
        void projMeas(vector[double] &meas, vector[double] &pmeas)
        void projGrad(vector[double] &meas, vector[vector[double]] &meas_grad,
                                           vector[vector[double]] &pgrad)
        
        
        void res(AbstractSolver &solver, vector[double] &res)
        void resGrad(AbstractSolver &solver, vector[double] &res, 
                             vector[vector[double] ] &res_grad)
        void func(AbstractSolver &solver, double &obj)
        void funcGrad(AbstractSolver &solver, double &obj, vector[double] &obj_grad)
        void funcHess(AbstractSolver &solver, double &obj, vector[double] &obj_grad, 
                              vector[vector[double] ] &obj_hess)
        
cdef extern from "abstract_solver.hpp":
    cdef cppclass AbstractSolver:
        AbstractSolver() except+
        
        void solveU(vector[vector[double] ] &u)
        void solveM(vector[double] &meas)
        void solveMGrad(vector[double] &meas, vector[vector[double] ] &grad)
        
cdef extern from "lin_solver.hpp":
    cdef cppclass Update:
        Update() except +
        Update(int NSM, vector[int] &sm_bonds, vector[double] &stretch_mod) except +
    
    cdef cppclass LinSolver(AbstractSolver):
        LinSolver() except +
        LinSolver(Network &nw, int NF, vector[Perturb] &pert, vector[Measure] &meas) except +
        
        void setIntStrengths(vector[double] &K)
    
        void solveDOF(vector[vector[double] ] &disp, vector[vector[double] ] &strain_tensor)
        
        void prepareUpdateList(vector[Update] &up_list)
        double solveMeasUpdate(int i, vector[vector[double] ] &meas)
        double setUpdate(int i, vector[vector[double] ] &meas)
        void replaceUpdates(vector[int] &replace_index, vector[Update] &replace_update)
        double getConditionNum()
        
        void solveFeasability(AbstractObjFunc &obj_func, vector[bool] &feasible,
                                        vector[vector[double] ] &u, vector[vector[double] ] &con_err)
        
        void getAugHessian(vector[vector[vector[double] ] ] &AH)
        void getAugMeasMat(vector[vector[vector[double] ] ] &AM)
        void getAugForce(vector[vector[double] ] &Af)
        
        void getEigenvals(vector[double] &evals, bool bordered)
        
        
        void initEdgeCalc(vector[vector[double] ] &meas);
        void calcEdgeResponse(vector[vector[double] ] &meas);
        double solveEdgeUpdate(int i, vector[vector[double] ] &meas);
        double removeEdge(int irem, vector[vector[double] ] &meas);



        
cdef extern from "nonlin_solver.hpp":
    cdef cppclass NonlinSolver:
        NonlinSolver() except +
        NonlinSolver(Network &nw, int NF, vector[Perturb] &pert, vector[Measure] &meas) except +

        void setIntStrengths(vector[double] &K)
        void setAmplitude(double amp)
        
        void solveAll(vector[vector[double] ] &u)
        void solveMeas(vector[vector[double] ] &meas)
        void solveDOF(vector[vector[double] ] &disp, vector[vector[double] ] &strain_tensor)

        void solveEnergy(int t, vector[double] &u, double &energy)
        void solveGrad(int t, vector[double] &u, vector[double] &grad)
        void solveCon(int t, vector[double] &u, double &con);
        void solveConGrad(int t, vector[double] &u, vector[double] &grad);
        
        
cdef class CyNetwork:
    cdef Network c_nw

    def __cinit__(self, NN, node_pos, NE, edgei, edgej, NGDOF, L, enable_affine, bvecij, eq_length, stretch_mod):
        
        cdef int c_NN = NN
        cdef vector[double] c_node_pos = np.ascontiguousarray(node_pos, dtype=np.double)
        cdef int c_NE = NE
        cdef vector[int] c_edgei = np.ascontiguousarray(edgei, dtype=np.int32)
        cdef vector[int] c_edgej = np.ascontiguousarray(edgej, dtype=np.int32)
        cdef int c_NGDOF = NGDOF
        cdef vector[double] c_L = np.ascontiguousarray(L, dtype=np.double)
        cdef bool c_enable_affine = enable_affine
        
        cdef vector[double] c_bvecij = np.ascontiguousarray(bvecij, dtype=np.double)
        cdef vector[double] c_eq_length = np.ascontiguousarray(eq_length, dtype=np.double)
        cdef vector[double] c_stretch_mod = np.ascontiguousarray(stretch_mod, dtype=np.double)

        self.c_nw = Network(c_NN, c_node_pos, c_NE, c_edgei, c_edgej, c_NGDOF, c_L, c_enable_affine, 
                            c_bvecij, c_eq_length, c_stretch_mod)


        
cdef class CyPerturb:
    cdef Perturb c_pert
    
    def __cinit__(self, NIstrain, istrain_nodesi, istrain_nodesj, istrain_bonds, istrain, istrain_vec,
                 NIstress, istress_bonds, istress,
                 apply_affine_strain, strain_tensor, apply_affine_stress, stress_tensor,
                 NFix, fixed_nodes):

        cdef int c_NIstrain = NIstrain
        
        cdef vector[int] c_istrain_nodesi = np.ascontiguousarray(istrain_nodesi, dtype=np.int32)
        cdef vector[int] c_istrain_nodesj = np.ascontiguousarray(istrain_nodesj, dtype=np.int32)
        cdef vector[int] c_istrain_bonds = np.ascontiguousarray(istrain_bonds, dtype=np.int32)
        cdef vector[double] c_istrain = np.ascontiguousarray(istrain, dtype=np.double)
        cdef vector[double] c_istrain_vec = np.ascontiguousarray(istrain_vec, dtype=np.double)

        cdef bool c_apply_affine_strain = apply_affine_strain
        cdef vector[double] c_strain_tensor = np.ascontiguousarray(strain_tensor, dtype=np.double)
        
        cdef int c_NIstress = NIstress
        
        cdef vector[int] c_istress_bonds = np.ascontiguousarray(istress_bonds, dtype=np.int32)
        cdef vector[double] c_istress = np.ascontiguousarray(istress, dtype=np.double)

        cdef bool c_apply_affine_stress = apply_affine_stress
        cdef vector[double] c_stress_tensor = np.ascontiguousarray(stress_tensor, dtype=np.double)
        
        cdef int c_NFix = NFix
        cdef vector[int] c_fixed_nodes = np.ascontiguousarray(fixed_nodes, dtype=np.int32)
        
        self.c_pert = Perturb(c_NIstrain, c_istrain_nodesi, c_istrain_nodesj, c_istrain_bonds, c_istrain, c_istrain_vec,
                             c_NIstress, c_istress_bonds, c_istress,
                              c_apply_affine_strain, c_strain_tensor,
                             c_apply_affine_stress, c_stress_tensor,
                             c_NFix, c_fixed_nodes)

        
cdef class CyMeasure:
    cdef Measure c_meas
    
    def __cinit__(self, NOstrain, ostrain_nodesi, ostrain_nodesj, ostrain_bonds, ostrain_vec,
                 NOstress, ostress_bonds,
                 measure_affine_stress, measure_affine_strain):

        cdef int c_NOstrain = NOstrain
        
        cdef vector[int] c_ostrain_nodesi = np.ascontiguousarray(ostrain_nodesi, dtype=np.int32)
        cdef vector[int] c_ostrain_nodesj = np.ascontiguousarray(ostrain_nodesj, dtype=np.int32)
        cdef vector[int] c_ostrain_bonds = np.ascontiguousarray(ostrain_bonds, dtype=np.int32)
        cdef vector[double] c_ostrain_vec = np.ascontiguousarray(ostrain_vec, dtype=np.double)

        cdef bool c_measure_affine_strain = measure_affine_strain   
        
        cdef int c_NOstress = NOstress
        
        cdef vector[int] c_ostress_bonds = np.ascontiguousarray(ostress_bonds, dtype=np.int32)

        cdef bool c_measure_affine_stress = measure_affine_stress 
        
        self.c_meas = Measure(c_NOstrain, c_ostrain_nodesi, c_ostrain_nodesj, c_ostrain_bonds,
                               c_ostrain_vec, c_NOstress, c_ostress_bonds,
                               c_measure_affine_strain, c_measure_affine_stress) 
        
        
cdef class CyUpdate:
    cdef Update c_up
    
    def __cinit__(self, NSM, sm_bonds, stretch_mod):
        
        cdef int c_NSM = NSM
        cdef vector[int] c_sm_bonds = np.ascontiguousarray(sm_bonds, dtype=np.int32)
        cdef vector[double] c_stretch_mod = np.ascontiguousarray(stretch_mod, dtype=np.double)
        
        self.c_up = Update(c_NSM,  c_sm_bonds, c_stretch_mod)
        
    
cdef class CyAugIneqRatioChangeObjFunc:
    cdef AugIneqRatioChangeObjFunc c_obj
    
    def __cinit__(self, delta_ratio_target):
        cdef vector[double] c_delta_ratio_target = np.ascontiguousarray(delta_ratio_target, dtype=np.double)
        
        self.c_obj = AugIneqRatioChangeObjFunc(c_delta_ratio_target)

    def initialize(self, CyLinSolver solver):
        cdef vector[double] c_x_init
        self.c_obj.initialize(solver.c_solver[0], c_x_init)
        
        return np.array(c_x_init)
    
    def setWeights(self, a, b, c):
        cdef double c_a = a
        cdef double c_b = b
        cdef double c_c = c
        self.c_obj.setWeights(c_a, c_b, c_c)
    
    def setRegularize(self, mu, x_reg):
        cdef double c_mu = mu
        cdef vector[double] c_x_reg = np.ascontiguousarray(x_reg, dtype=np.double)

        self.c_obj.setRegularize(c_mu, c_x_reg)
    
    def res(self, x, CyLinSolver solver):
        cdef vector[double] c_x = np.ascontiguousarray(x, dtype=np.double)
        cdef vector[double] res
        
        self.c_obj.res(x, solver.c_solver[0], res)
        
        return np.array(res)
    
    def resGrad(self, x, CyLinSolver solver):
        cdef vector[double] c_x = np.ascontiguousarray(x, dtype=np.double)
        cdef vector[double] res
        cdef vector[int] rows
        cdef vector[int] cols
        cdef vector[double] vals
         
        self.c_obj.resGrad(x, solver.c_solver[0], res, rows, cols, vals)
        
        return (np.array(res, dtype=np.double), np.array(rows, dtype=np.int), np.array(cols, dtype=np.int), np.array(vals, dtype=np.double))
        
       
    def func(self, x, CyLinSolver solver):
        cdef vector[double] c_x = np.ascontiguousarray(x, dtype=np.double)
        cdef double c_obj = 0.0
        
        self.c_obj.func(c_x, solver.c_solver[0], c_obj)
        
        return c_obj
        
    def funcGrad(self, x, CyLinSolver solver):
        cdef vector[double] c_x = np.ascontiguousarray(x, dtype=np.double)
        cdef double c_obj = 0.0
        cdef vector[double] c_obj_grad

        self.c_obj.funcGrad(c_x, solver.c_solver[0], c_obj, c_obj_grad)
        
        return (c_obj, np.array(c_obj_grad))
    
    
    def funcHess(self, x, CyLinSolver solver):
        cdef vector[double] c_x = np.ascontiguousarray(x, dtype=np.double)
        cdef vector[int] rows
        cdef vector[int] cols
        cdef vector[double] vals
         
        self.c_obj.funcHess(x, solver.c_solver[0], rows, cols, vals)
        
        return (np.array(rows, dtype=np.int), np.array(cols, dtype=np.int), np.array(vals, dtype=np.double))
    
cdef class CyIneqRatioChangeObjFunc:
    cdef IneqRatioChangeObjFunc c_obj
    
    def __cinit__(self, Nterms, Ngrad, ratio_init, delta_ratio_target):
        
        cdef int c_Nterms = Nterms
        cdef int c_Ngrad = Ngrad
        cdef vector[double] c_ratio_init = np.ascontiguousarray(ratio_init, dtype=np.double)
        cdef vector[double] c_delta_ratio_target = np.ascontiguousarray(delta_ratio_target, dtype=np.double)
        
        self.c_obj = IneqRatioChangeObjFunc(c_Nterms, c_Ngrad, c_ratio_init, c_delta_ratio_target)
        
        
    def setRatioInit(self, ratio_init):
        cdef vector[double] c_ratio_init = np.ascontiguousarray(ratio_init, dtype=np.double)
        self.c_obj.setRatioInit(c_ratio_init)
     
    def res(self, CyLinSolver solver):
        cdef vector[double] c_res
    
        self.c_obj.res(solver.c_solver[0], c_res)
        
        return np.array(c_res)
    
    def resGrad(self, CyLinSolver solver):
        cdef vector[double] c_res
        cdef vector[vector[double] ] c_res_grad
    
        self.c_obj.resGrad(solver.c_solver[0], c_res, c_res_grad)
        
        res =  np.array(c_res)
        res_grad = np.zeros([len(c_res_grad), len(c_res_grad[0])], dtype=np.double)
        
        for i in range(c_res_grad.size()):
            res_grad[i, :] = np.array(c_res_grad[i]) 
                        
        return (res, res_grad)    
    
    
    def func(self, CyLinSolver solver):
        cdef double c_obj = 0.0
    
        self.c_obj.func(solver.c_solver[0], c_obj)
        
        return c_obj
        
    def funcGrad(self, CyLinSolver solver):
        cdef double c_obj = 0.0
        cdef vector[double] c_obj_grad
    
        self.c_obj.funcGrad(solver.c_solver[0], c_obj, c_obj_grad)
        
        obj_grad =  np.array(c_obj_grad)
    
        return (c_obj, obj_grad)
    
    def funcHess(self, CyLinSolver solver):
        cdef double c_obj = 0.0
        cdef vector[double] c_obj_grad
        cdef vector[vector[double] ] c_obj_hess
    
        self.c_obj.funcHess(solver.c_solver[0], c_obj, c_obj_grad, c_obj_hess)
        
        obj_grad =  np.array(c_obj_grad)
        obj_hess = np.zeros([len(c_obj_hess), len(c_obj_hess)], dtype=np.double)
        
        for i in range(c_obj_hess.size()):
            obj_hess[i, :] = np.array(c_obj_hess[i]) 
            
        return (c_obj, obj_grad, obj_hess)
    
    def objFuncTerms(self, meas):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
            
        cdef double c_obj = 0.0
        cdef vector[double] c_terms
            
        self.c_obj.objFuncTerms(c_meas, c_obj, c_terms)
        
        return (c_obj, np.array(c_terms))
        
    def objFunc(self, meas):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
            
        cdef double c_obj = 0.0
        
        self.c_obj.objFunc(c_meas, c_obj)
        
        return c_obj
        
    def objFuncGrad(self, meas, meas_grad):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
        cdef vector[vector[double]] c_meas_grad
        
        cdef vector[double] tmp
        for i in range(len(meas)):
            tmp = np.ascontiguousarray(meas_grad[i, :], dtype=np.double)
            c_meas_grad.push_back(tmp)
            
        cdef double c_obj = 0.0
        cdef vector[double] c_obj_grad
        
        self.c_obj.objFuncGrad(c_meas, c_meas_grad, c_obj_grad)
        
        return np.array(c_obj_grad)
    
    
    def projMeas(self, meas):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
            
        cdef vector[double] c_pmeas
            
        self.c_obj.projMeas(c_meas, c_pmeas)
        
        return np.array(c_pmeas)
    
    def projGrad(self, meas, meas_grad):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
        cdef vector[vector[double]] c_meas_grad
        
        cdef vector[double] tmp
        for i in range(len(meas)):
            tmp = np.ascontiguousarray(meas_grad[i, :], dtype=np.double)
            c_meas_grad.push_back(tmp)
            
        cdef double c_obj = 0.0
        cdef vector[vector[double]] c_pgrad
        
        self.c_obj.projGrad(c_meas, c_meas_grad, c_pgrad)
        
        pgrad = np.empty_like(meas_grad)
        
        for i in range(len(pgrad)):
            pgrad[i] = np.array(c_pgrad[i])
        
        return pgrad
    
cdef class CyEqRatioChangeObjFunc:
    cdef EqRatioChangeObjFunc c_obj
    
    def __cinit__(self, Nterms, Ngrad, ratio_init, delta_ratio_target):
        
        cdef int c_Nterms = Nterms
        cdef int c_Ngrad = Ngrad
        cdef vector[double] c_ratio_init = np.ascontiguousarray(ratio_init, dtype=np.double)
        cdef vector[double] c_delta_ratio_target = np.ascontiguousarray(delta_ratio_target, dtype=np.double)
        
        self.c_obj = EqRatioChangeObjFunc(c_Nterms, c_Ngrad, c_ratio_init, c_delta_ratio_target)
        
        
    def setRatioInit(self, ratio_init):
        cdef vector[double] c_ratio_init = np.ascontiguousarray(ratio_init, dtype=np.double)
        self.c_obj.setRatioInit(c_ratio_init)
     
    def objFuncTerms(self, meas):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
            
        cdef double c_obj = 0.0
        cdef vector[double] c_terms
            
        self.c_obj.objFuncTerms(c_meas, c_obj, c_terms)
        
        return (c_obj, np.array(c_terms))
        
    def objFunc(self, meas):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
            
        cdef double c_obj = 0.0
        
        self.c_obj.objFunc(c_meas, c_obj)
        
        return c_obj
        
    def objFuncGrad(self, meas, meas_grad):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
        cdef vector[vector[double]] c_meas_grad
        
        cdef vector[double] tmp
        for i in range(len(meas)):
            tmp = np.ascontiguousarray(meas_grad[i, :], dtype=np.double)
            c_meas_grad.push_back(tmp)
            
        cdef double c_obj = 0.0
        cdef vector[double] c_obj_grad
        
        self.c_obj.objFuncGrad(c_meas, c_meas_grad, c_obj_grad)
        
        return np.array(c_obj_grad)
    
    
    def projMeas(self, meas):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
            
        cdef vector[double] c_pmeas
            
        self.c_obj.projMeas(c_meas, c_pmeas)
        
        return np.array(c_pmeas)
    
    def projGrad(self, meas, meas_grad):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
        cdef vector[vector[double]] c_meas_grad
        
        cdef vector[double] tmp
        for i in range(len(meas)):
            tmp = np.ascontiguousarray(meas_grad[i, :], dtype=np.double)
            c_meas_grad.push_back(tmp)
            
        cdef double c_obj = 0.0
        cdef vector[vector[double]] c_pgrad
        
        self.c_obj.projGrad(c_meas, c_meas_grad, c_pgrad)
        
        pgrad = np.empty_like(meas_grad)
        
        for i in range(len(pgrad)):
            pgrad[i] = np.array(c_pgrad[i])
        
        return pgrad
        
cdef class CyIneqRatioObjFunc:
    cdef IneqRatioObjFunc c_obj
    
    def __cinit__(self, Nterms, Ngrad, ratio_init, delta_ratio_target):
        
        cdef int c_Nterms = Nterms
        cdef int c_Ngrad = Ngrad
        cdef vector[double] c_ratio_init = np.ascontiguousarray(ratio_init, dtype=np.double)
        cdef vector[double] c_delta_ratio_target = np.ascontiguousarray(delta_ratio_target, dtype=np.double)
        
        self.c_obj = IneqRatioObjFunc(c_Nterms, c_Ngrad, c_ratio_init, c_delta_ratio_target)
        
        
    def setRatioInit(self, ratio_init):
        cdef vector[double] c_ratio_init = np.ascontiguousarray(ratio_init, dtype=np.double)
        self.c_obj.setRatioInit(c_ratio_init)
     
    def res(self, CyLinSolver solver):
        cdef vector[double] c_res
    
        self.c_obj.res(solver.c_solver[0], c_res)
        
        return np.array(c_res)
    
    def resGrad(self, CyLinSolver solver):
        cdef vector[double] c_res
        cdef vector[vector[double] ] c_res_grad
    
        self.c_obj.resGrad(solver.c_solver[0], c_res, c_res_grad)
        
        res =  np.array(c_res)
        res_grad = np.zeros([len(c_res_grad), len(c_res_grad[0])], dtype=np.double)
        
        for i in range(c_res_grad.size()):
            res_grad[i, :] = np.array(c_res_grad[i]) 
                        
        return (res, res_grad)    
    
    
    def func(self, CyLinSolver solver):
        cdef double c_obj = 0.0
    
        self.c_obj.func(solver.c_solver[0], c_obj)
        
        return c_obj
        
    def funcGrad(self, CyLinSolver solver):
        cdef double c_obj = 0.0
        cdef vector[double] c_obj_grad
    
        self.c_obj.funcGrad(solver.c_solver[0], c_obj, c_obj_grad)
        
        obj_grad =  np.array(c_obj_grad)
    
        return (c_obj, obj_grad)
    
    def funcHess(self, CyLinSolver solver):
        cdef double c_obj = 0.0
        cdef vector[double] c_obj_grad
        cdef vector[vector[double] ] c_obj_hess
    
        self.c_obj.funcHess(solver.c_solver[0], c_obj, c_obj_grad, c_obj_hess)
        
        obj_grad =  np.array(c_obj_grad)
        obj_hess = np.zeros([len(c_obj_hess), len(c_obj_hess)], dtype=np.double)
        
        for i in range(c_obj_hess.size()):
            obj_hess[i, :] = np.array(c_obj_hess[i]) 
            
        return (c_obj, obj_grad, obj_hess)
    
    def objFuncTerms(self, meas):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
            
        cdef double c_obj = 0.0
        cdef vector[double] c_terms
            
        self.c_obj.objFuncTerms(c_meas, c_obj, c_terms)
        
        return (c_obj, np.array(c_terms))
        
    def objFunc(self, meas):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
            
        cdef double c_obj = 0.0
        
        self.c_obj.objFunc(c_meas, c_obj)
        
        return c_obj
        
    def objFuncGrad(self, meas, meas_grad):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
        cdef vector[vector[double]] c_meas_grad
        
        cdef vector[double] tmp
        for i in range(len(meas)):
            tmp = np.ascontiguousarray(meas_grad[i, :], dtype=np.double)
            c_meas_grad.push_back(tmp)
            
        cdef double c_obj = 0.0
        cdef vector[double] c_obj_grad
        
        self.c_obj.objFuncGrad(c_meas, c_meas_grad, c_obj_grad)
        
        return np.array(c_obj_grad)
    
    
    def projMeas(self, meas):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
            
        cdef vector[double] c_pmeas
            
        self.c_obj.projMeas(c_meas, c_pmeas)
        
        return np.array(c_pmeas)
    
    def projGrad(self, meas, meas_grad):
        cdef vector[double] c_meas = np.ascontiguousarray(meas, dtype=np.double)
        cdef vector[vector[double]] c_meas_grad
        
        cdef vector[double] tmp
        for i in range(len(meas)):
            tmp = np.ascontiguousarray(meas_grad[i, :], dtype=np.double)
            c_meas_grad.push_back(tmp)
            
        cdef double c_obj = 0.0
        cdef vector[vector[double]] c_pgrad
        
        self.c_obj.projGrad(c_meas, c_meas_grad, c_pgrad)
        
        pgrad = np.empty_like(meas_grad)
        
        for i in range(len(pgrad)):
            pgrad[i] = np.array(c_pgrad[i])
        
        return pgrad
        
        
cdef class CyLinSolver:
    cdef LinSolver *c_solver
    
    
    def __cinit__(self, CyNetwork nw, NF, list pert, list meas):
        
        cdef vector[Perturb] c_pert
        cdef vector[Measure] c_meas
        
        for p in pert:
            c_pert.push_back((<CyPerturb>p).c_pert)
        
        for m in meas:
            c_meas.push_back((<CyMeasure>m).c_meas)
        
        self.c_solver = new LinSolver(nw.c_nw, NF, c_pert, c_meas)
      
    def __dealloc__(self):
        del self.c_solver
        
    def setIntStrengths(self, K):
        
        cdef vector[double] c_K = np.ascontiguousarray(K, dtype=np.double)
        
        self.c_solver.setIntStrengths(c_K)
      
    def solveDOF(self):
        cdef vector[vector[double]] c_disp
        cdef vector[vector[double]] c_strain_tensor
        
        self.c_solver.solveDOF(c_disp, c_strain_tensor)
        
        disp = []
        strain_tensor = []
        
        for i in range(c_disp.size()):
            disp.append(np.array(c_disp[i]))
            
        for i in range(c_strain_tensor.size()):
            strain_tensor.append(np.array(c_strain_tensor[i]))
            
        return (disp, strain_tensor)
    
    def solveAll(self):
        
        cdef vector[vector[double]] c_u
        self.c_solver.solveU(c_u)
        
        u = []
        
        for i in range(c_u.size()):
            u.append(np.array(c_u[i]))
                
        return u
    
    def solveMeas(self):
        
        cdef vector[double] c_meas
        self.c_solver.solveM(c_meas)
                
        return np.array(c_meas)
    
    def solveMeasGrad(self):
        
        cdef vector[double] c_meas
        cdef vector[vector[double] ] c_grad
        
        
        self.c_solver.solveMGrad(c_meas, c_grad)
        
        meas = np.array(c_meas)
        grad = np.zeros([c_grad.size(), c_grad[0].size()], np.double)
        
        for j in range(c_grad.size()):
            grad[j, :] = np.array(c_grad[j]) 

        return (meas, grad)
        
    def prepareUpdateList(self, up_list):
        cdef vector[Update] c_up_list
        
        for up in up_list:
            c_up_list.push_back((<CyUpdate>up).c_up)
            
        self.c_solver.prepareUpdateList(c_up_list)
        
    def solveMeasUpdate(self, i):
        cdef vector[vector[double]] c_meas
        cdef c_i = i
        
        cdef double error = self.c_solver.solveMeasUpdate(c_i, c_meas)
        
        meas = []
        
        for i in range(c_meas.size()):
            meas.append(np.array(c_meas[i]))
                
        return (error, meas)
        
    def setUpdate(self, i):  
        cdef vector[vector[double]] c_meas
        cdef c_i = i
        
        cdef double error = self.c_solver.setUpdate(c_i, c_meas)
        
        meas = []
        
        for i in range(c_meas.size()):
            meas.append(np.array(c_meas[i]))
                
        return (error, meas)
    
    def replaceUpdates(self, replace_index, replace_ups):
        cdef vector[Update] c_replace_ups
        cdef vector[int] c_replace_index = np.ascontiguousarray(replace_index, dtype=np.int)
        
        for up in replace_ups:
            c_replace_ups.push_back((<CyUpdate>up).c_up)
            
        self.c_solver.replaceUpdates(c_replace_index, c_replace_ups)
    
    
    def getConditionNum(self):
        cdef double condition = self.c_solver.getConditionNum()
        
        return condition
    
    def solveFeasability(self, CyIneqRatioChangeObjFunc obj_func):
    
        cdef vector[bool] c_feasible
        cdef vector[vector[double]] c_u
        cdef vector[vector[double]] c_con_err
        
        self.c_solver.solveFeasability(obj_func.c_obj, c_feasible, c_u, c_con_err)
        
        u = []
        feasible = []
        con_err = []
            
        for i in range(c_feasible.size()):
            feasible.append(np.array(c_feasible[i]))
            
        for i in range(c_u.size()):
            u.append(np.array(c_u[i]))
            
        for i in range(c_con_err.size()):
            con_err.append(np.array(c_con_err[i]))
                
        return (feasible, u, con_err)
    
    
    def getAugHessian(self):
        
        cdef vector[vector[vector[double] ] ] c_AH
        
        
        self.c_solver.getAugHessian(c_AH)
                
        AH = []
        
        for t in range(c_AH.size()):
            AH.append(np.zeros([c_AH[t].size(), c_AH[t][0].size()], np.double))
            for j in range(c_AH[t].size()):
                AH[t][j, :] = np.array(c_AH[t][j]) 

        return AH
    
    def getAugMeasMat(self):
        
        cdef vector[vector[vector[double] ] ] c_AM
        
        
        self.c_solver.getAugMeasMat(c_AM)
                
        AM = []
        
        for t in range(c_AM.size()):
            AM.append(np.zeros([c_AM[t].size(), c_AM[t][0].size()], np.double))
            for j in range(c_AM[t].size()):
                AM[t][j, :] = np.array(c_AM[t][j]) 

        return AM
    
    def getAugForce(self):
        
        cdef vector[vector[double] ] c_Af
        
        
        self.c_solver.getAugForce(c_Af)
                
        fM = []
        
        for t in range(c_Af.size()):
            fM.append(np.array(c_Af[t]))

        return fM
    
    def getEigenvals(self, bordered=True):
        cdef vector[double] c_evals
        cdef bool c_bordered = bordered
        
        self.c_solver.getEigenvals(c_evals, c_bordered)
        
        return np.array(c_evals)
    
    
    def initEdgeCalc(self):
        cdef vector[vector[double]] c_meas
        
        self.c_solver.initEdgeCalc(c_meas)
        
        meas = []
        
        for i in range(c_meas.size()):
            meas.append(np.array(c_meas[i]))
                
        return meas
    
    def calcEdgeResponse(self):
        cdef vector[vector[double]] c_meas
        
        self.c_solver.calcEdgeResponse(c_meas)
        
        meas = []
        
        for i in range(c_meas.size()):
            meas.append(np.array(c_meas[i]))
                
        return meas
    
    def solveEdgeUpdate(self, i):
        cdef vector[vector[double]] c_meas
        cdef c_i = i
        
        cdef double error = self.c_solver.solveEdgeUpdate(c_i, c_meas)
        
        meas = []
        
        for i in range(c_meas.size()):
            meas.append(np.array(c_meas[i]))
                
        return (error, meas)
    
    def removeEdge(self, i):
        cdef vector[vector[double]] c_meas
        cdef c_i = i
        
        cdef double error = self.c_solver.removeEdge(c_i, c_meas)
        
        meas = []
        
        for i in range(c_meas.size()):
            meas.append(np.array(c_meas[i]))
                
        return (error, meas)
    
cdef class CyNonlinSolver:
    cdef NonlinSolver *c_solver
    
    
    def __cinit__(self, CyNetwork nw, NF, list pert, list meas):
        
        cdef vector[Perturb] c_pert
        cdef vector[Measure] c_meas
        
        for p in pert:
            c_pert.push_back((<CyPerturb>p).c_pert)
            
        for m in meas:
            c_meas.push_back((<CyMeasure>m).c_meas)
        
        self.c_solver = new NonlinSolver(nw.c_nw, NF, c_pert, c_meas)
      
    def __dealloc__(self):
        del self.c_solver
        
    def setIntStrengths(self, K):
        
        cdef vector[double] c_K = np.ascontiguousarray(K, dtype=np.double)
        
        self.c_solver.setIntStrengths(c_K)
        
    def setAmplitude(self, amp):
        cdef double c_amp = amp
        self.c_solver.setAmplitude(c_amp)
        
    def solveDOF(self):
        cdef vector[vector[double]] c_disp
        cdef vector[vector[double]] c_strain_tensor
        
        self.c_solver.solveDOF(c_disp, c_strain_tensor)
        
        disp = []
        strain_tensor = []
        
        for i in range(c_disp.size()):
            disp.append(np.array(c_disp[i]))
            
        for i in range(c_strain_tensor.size()):
            strain_tensor.append(np.array(c_strain_tensor[i]))
            
        return (disp, strain_tensor)
    
    def solveAll(self):
        
        cdef vector[vector[double]] c_u
        self.c_solver.solveAll(c_u)
        
        u = []
        
        for i in range(c_u.size()):
            u.append(np.array(c_u[i]))
                
        return u
    
    def solveMeas(self):
        
        cdef vector[vector[double]] c_meas
        self.c_solver.solveMeas(c_meas)
        
        meas = []
        
        for i in range(c_meas.size()):
            meas.append(np.array(c_meas[i]))
                
        return meas
      
    def solveEnergy(self, u, t):
        cdef int c_t = t
        cdef vector[double] c_u = np.ascontiguousarray(u, dtype=np.double)
        
        cdef double c_energy = 0.0
        
        self.c_solver.solveEnergy(c_t, c_u, c_energy)
        
        # print "energy", c_energy
            
        return c_energy
    
    def solveGrad(self, u, t):
        cdef int c_t = t
        cdef vector[double] c_u = np.ascontiguousarray(u, dtype=np.double)
        
        cdef vector[double] c_grad
        
        self.c_solver.solveGrad(c_t, c_u, c_grad)
        
        

        return np.array(c_grad)
    
    def solveCon(self, u, t):
        cdef int c_t = t
        cdef vector[double] c_u  = np.ascontiguousarray(u, dtype=np.double)
        
        cdef double c_con = 0.0
        
        self.c_solver.solveCon(c_t, c_u, c_con)
            
        # print "con", c_con
            
        return c_con
    
    def solveConGrad(self, u, t):
        cdef int c_t = t
        cdef vector[double] c_u =  np.ascontiguousarray(u, dtype=np.double)
        
        cdef vector[double] c_grad
        
        self.c_solver.solveConGrad(c_t, c_u, c_grad)
        
        return np.array(c_grad)
    