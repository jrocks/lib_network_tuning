import mech_network_solver as mns
import numpy as np
import scipy.optimize as spo
import numpy.linalg as la
import time


class NonlinSolver(object):
    
    def __init__(self, nw, NF, pert, meas):
        self.NF = NF
        self.NNDOF = nw.DIM * nw.NN
        if nw.enable_affine:
            self.NADOF = nw.DIM**2
        else:
            self.NADOF = 0
            
        self.NDOF = self.NNDOF + self.NADOF
        self.meas = meas
        self.pert = pert
        
        cypert = []
        for p in pert:
            cypert.append(p.getCyPert())
            
        cymeas = []
        for m in meas:
            cymeas.append(m.getCyMeas())
            
        self.solver = mns.CyNonlinSolver(nw.getCyNetwork(), NF, cypert, cymeas)
        
        
        u0 = np.zeros(self.NDOF, float)
        
        print "Grad check error:", spo.check_grad(self.solver.solveEnergy, self.solver.solveGrad, u0, 0)
        print "Con grad check error:", spo.check_grad(self.solver.solveCon, self.solver.solveConGrad, u0, 0)
        
        
    def setIntStrengths(self, K):
                
        self.solver.setIntStrengths(K)
        
        
    def solveAll(self):
        
        u = []
        
        tol = np.sqrt(np.finfo(float).eps)
        for t in range(self.NF):
            u0 = np.zeros(self.NDOF, float)
            constraints = {'type': 'eq', 'fun': self.solver.solveCon, 
                                    'jac': self.solver.solveConGrad, 'args': (t,)}
            # constraints = []
        
            t0 = time.time()

            res = spo.minimize(self.solver.solveEnergy, u0, args=(t,), method='SLSQP', callback=None,
                               jac=self.solver.solveGrad, constraints=constraints,
                               options={'ftol': tol, 'maxiter':10000000})
            t1 = time.time()
            print "Time:", t1-t0, "Success:", res.success, res.message
            
            u.append(res.x)
            
            
        return u
      
    def solveDOF(self):
        u = self.solveAll()
        
        disp = []
        strain_tensor = []
        
        for t in range(self.NF):
            disp.append(u[t][:self.NNDOF])
            strain_tensor.append(u[t][self.NNDOF:self.NNDOF+self.NADOF])
            
        return (disp, strain_tensor)
    
    def solveMeas(self):
        
        (disp, strain_tensor) = self.solveDOF()
        
        DIM = nw.DIM
        
        m = []
        for t in range(self.NF):
            m_tmp = np.zeros(meas[t].NOstrain, float)
            for c in range(meas[t].NOstrain):
                ei = meas[t].ostrain_nodesi[c]
                ej = meas[t].ostrain_nodesj[c]
                X0ij = meas[t].ostrain_vec[DIM*c:DIM*c+DIM]
                Xij = X0ij + disp[t][DIM*ej:DIM*ej+DIM] - disp[t][DIM*ei:DIM*ei+DIM]
                
                X0norm = la.norm(X0ij)
                Xnorm = la.norm(Xij)
                m_tmp[c] = Xnorm / X0norm - 1.0
            
            if meas[t].measure_affine_strain:
                m_tmp = np.append(m_tmp, strain_tensor[t])
                
            m.append(m_tmp)
                
        return m