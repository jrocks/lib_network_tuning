#ifndef NONLINSOLVER
#define NONLINSOLVER
    
#include "util.hpp"
#include "network.hpp"
#include "perturb.hpp"
#include "measure.hpp"
#include "stdafx.h"
#include "optimization.h"
    
class NonlinSolver {
    private:
    
        // Network object
        Network nw;
    
        // Number of dof in Hessian (includes fixed constraints on global dofs)
        int NDOF;
        // Number of node dofs
        int NNDOF;
        // Number of affine dofs;
        int NADOF;
        // Number of fixed global dofs
        int NFGDOF;
    
        // Number of independent functions (pert/meas pairs)
        int NF;

        // Number of measurements for each measurement
        std::vector<int> NM;
    
        // Vector of interaction strengths
        XVec K;
        // Matrix of fixed global dof constraints
        XMat G;
        
        // List of perturbations to apply
        std::vector<Perturb> pert;
    
        // Forces
        std::vector<SMat > f;
    
        // List of measurements
        std::vector<Measure> meas;
    
        // Perturbation amplitude (0 to 1)
        double amp;
    
    public:
    
        NonlinSolver() {};
        NonlinSolver(Network &nw, int NF, std::vector<Perturb> &pert, std::vector<Measure> &meas);
    
        // Set interaction strengths
        void setIntStrengths(std::vector<double> &K);
    
        // Solve for deformation given the current set of interaction strengths
        void solveAll(std::vector<std::vector<double> > &u);
        void solveMeas(std::vector<std::vector<double> > &meas);
        void solveMeasGrad(std::vector<std::vector<double> > &meas, std::vector<std::vector<std::vector<double> > > &grad);
        void solveDOF(std::vector<std::vector<double> > &disp, std::vector<std::vector<double> > &strain_tensor);
    
        // Solve for deformation energy and gradient given the current set of interaction strengths
        void solveEnergy(int t, std::vector<double> &u, double &energy);
        void solveGrad(int t, std::vector<double> &u, std::vector<double> &grad);
        void solveCon(int t, std::vector<double> &u, double &con);
        void solveConGrad(int t, std::vector<double> &u, std::vector<double> &grad);
    
        static void FuncJacWrapper(const alglib::real_1d_array &x, alglib::real_1d_array &func, 
                                   alglib::real_2d_array &jac, void *ptr);
    
        void setAmplitude(double amp);
    
    private:
        void setupGlobalConMat();
    
};  

struct NonlinOptParams {
	int t;
	NonlinSolver *solver;
	
};
    
#endif
