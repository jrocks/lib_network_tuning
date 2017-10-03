#ifndef LINSOLVER
#define LINSOLVER
    
#include "util.hpp"
#include "network.hpp"
#include "perturb.hpp"
#include "measure.hpp"
#include "abstract_objective_function.hpp"
#include "stdafx.h"
#include "optimization.h"
#include "abstract_solver.hpp"

    
class Update {
    public:
        
        // Update stretch moduli
        int NSM;
        XiVec sm_bonds;
        XVec stretch_mod;
    
    Update() {}
    Update(int NSM, std::vector<int> &sm_bonds, std::vector<double> &stretch_mod) {
        this->NSM = NSM;

        vectorToEigen(sm_bonds, this->sm_bonds);
        vectorToEigen(stretch_mod, this->stretch_mod);
    }
    
};
    
class LinSolver: public AbstractSolver {
    public:
    
        // Network object
        Network nw;
    
        // Number of dof in Hessian (inclues fixed global DOFs)
        int NDOF;
        // Number of node dofs
        int NNDOF;
        // Number of affine dofs;
        int NADOF;
        // Number of fixed global dofs
        int NFGDOF;
        // Number of dofs in Bordered Hessian
        int NBDOF;
    
        // Number of independent functions (pert/meas pairs)
        int NF;
    
        // Number of constraints for each function
        std::vector<int> NC;
        // Total number constraints across all functions
        int NC_tot;
        // Number of measurements for each function
        std::vector<int> NM;
        // Total number of measurements across all functions
        int NM_tot;
        // Maps measurements from each function to a single array
        std::vector<int> meas_index;

    
        // Equilibrium Matrix
        SMat Q;
        // Vector of interaction strengths
        XVec K;
    
        // Matrix of fixed global dof constraints
        SMat G;
        // True if global degrees of freedom are to be fixed
        bool fix_global;
        
    
    
        // List of perturbations to apply
        std::vector<Perturb> pert;
        // Linear coeffs to perturbs
        std::vector<SMat > C1;
        // Const coeffs to perturbs
        std::vector<XVec > C0;
        // Forces
        std::vector<SMat > f;
    
    
        // List of measurements
        std::vector<Measure> meas;
        // Measurement matrices
        std::vector<SMat >  M;
        
    
        // Hessian
        SMat H;
    
        bool bordered;
    
        // Sparse matrix solver
        Eigen::UmfPackLU<SMat > solver;
        
        // Inverted bordered Hessian times linear coeffs
        std::vector<XMat > HinvC1;
        // Inverted bordered Hessian times force
        std::vector<XVec > Hinvf;
        // Inverted bordered Hessian times measurement matrices
        std::vector<XMat > HinvM;
        // Inverse bordered Hessian
        XMat Hinv;
         // Culmulative change in inverse Hessian due to updates
        XMat dHinv;
        
        
        
    
    
        // List of updates and associated matrices
        std::vector<Update > up_list;
        std::vector<XMat > SM_updates;
       
        
    
        // Maps dxd matrix to compressed symmetric matrix vector
        DiMat sm_index;
    
        Eigen::SelfAdjointEigenSolver<XMat > eigen_solver;
    
    public:
    
        LinSolver() {};
        LinSolver(Network &nw, int NF, std::vector<Perturb> &pert, std::vector<Measure> &meas);
    
        // Set interaction strengths
        void setIntStrengths(std::vector<double> &K);
    
    
    private:
        // Setup various matrices
        void setupEqMat();
        void setupGlobalConMat();
        void setupPertMat();
        void setupMeasMat();
        void setupHessian(SMat &H);
        void setupBorderedHessian(SMat &H);
        void extendBorderedSystem();
    
        // // Perform matrix calculations
        // void calcHessian();
        // void calcInvHessian();
        // void calcPert(bool use_full_inverse);
        // void calcMeas(bool use_full_inverse);
    
    public:
        
        // Internal C++ interface using eigen library
        // Returns all dofs
        void isolveU(std::vector<XVec > &u);
        // Return lagrange multipliers
        void isolveLambda(std::vector<XVec > &lambda);
        // Returns all measurements
        void isolveM(XVec &meas);
        // Returns gradient of each measurement
        void isolveMGrad(XVec &meas, std::vector<XVec > &grad);
        // Returns hessian of each measurement
        void isolveMHess(XVec &meas, std::vector<XVec > &grad, std::vector<XMat > &hess);
    
        // Solve for just the degrees of freedom
        // void solveDOF(std::vector<std::vector<double> > &disp, std::vector<std::vector<double> > &strain_tensor);
    
        void getEigenvals(std::vector<double> &evals, bool bordered);
    
    public:
    
    
        // Prepare list of possible updates
        void prepareUpdateList(std::vector<Update> &up_list);
        // Solve given update i is applied to the Hessian
        double solveMeasUpdate(int i, std::vector<std::vector<double> > &meas);
        // Make update i permanent and replace with a new update (new update should affect the same interactions) 
        double setUpdate(int i, std::vector<std::vector<double> > &meas);
        void replaceUpdates(std::vector<int> &replace_index, std::vector<Update > &replace_update);
        double getConditionNum();
    
    public:
    

        void solveEnergy(int t, std::vector<double> &u, double &energy);
        void solveGrad(int t, std::vector<double> &u, std::vector<double> &grad);
    

}; 

struct LinOptParams {
	int t;
	LinSolver *solver;
	
};
    
#endif
