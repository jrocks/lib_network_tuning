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
        // Hessian
        SMat H;
        // Matrix of fixed global dof constraints
        SMat G;
        // Inverse Hessian
        XMat Hinv;
    
        // Sparse matrix solver
        Eigen::UmfPackLU<SMat > solver;
        Eigen::UmfPackLU<SMat > Bsolver;
        // Eigen::ConjugateGradient<SMat, Eigen::Lower|Eigen::Upper > Bsolver;
    
        // Eigen::BiCGSTAB<SMat> Bsolver;
    
    
    
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
        // Inverted Hessian times linear coeffs
        std::vector<XMat > HinvC1;
        // Inverted Hessian times force
        std::vector<XVec > Hinvf;
    
        // List of measurements
        std::vector<Measure> meas;
    
        // Measurement matrices
        std::vector<SMat > M;
        // Inverted Hessian times measurement matrices
        std::vector<XMat > HinvM;
        
    
        bool need_H;
        bool need_Hinv;
        bool need_HinvPert;
        bool need_HinvM;
    
    
        // List of updates and associated matrices
        std::vector<Update > up_list;
        std::vector<XMat > SM_updates;
        // Culmulative change in inverse Hessian due to updates
        XMat dHinv;
    
    
        // Bordered Hessian
        SMat BH;
        //
        SMat BM;
        SMat Bf;
        // Culmulative change in inverse of bordered Hessian due to updates
        XMat dBHinv;
        // Inverted bordered Hessian times bordered force
        XVec BHinvf;
        // Inverted bordered Hessian times bordered measurement matrix
        XMat BHinvM;
        XMat BHinv;
    
    
        // Maps dxd matrix to compressed symmetric matrix vector
        DiMat sm_index;
    
        Eigen::SelfAdjointEigenSolver<XMat > eigen_solver;
    
    
    
    
    
    
    
    
    
    
    
        // Edge basis calculation
        // Number of remaining edges
        int NEr;
        int NSSS;
        int NSCS;
        // Reduced edge basis to full basis
        std::vector<int> r2f;
        // Full edgebasis to reduced basis
        std::vector<int> f2r;
        // Reduced stiffness vector
        XVec Kr;
        // Reduced Equlibrium matrix
        SMat Qr;
        SMat Mr;
    
        Eigen::SelfAdjointEigenSolver<XMat > edge_basis_solver;
        Eigen::ColPivHouseholderQR<XMat > qr_solver;
    
    
        XMat SSS_basis;
        XMat SCS_basis;
    
        XVec Mext;
        XVec Cs;
    
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
    
        // Perform matrix calculations
        void calcHessian();
        void calcInvHessian();
        void calcPert(bool use_full_inverse);
        void calcMeas(bool use_full_inverse);
    
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
        void solveDOF(std::vector<std::vector<double> > &disp, std::vector<std::vector<double> > &strain_tensor);
    
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
    
        void solveFeasability(AbstractObjFunc &obj_func, std::vector<bool> &feasible,
                                         std::vector<std::vector<double> > &u, std::vector<std::vector<double> > &con_err);
        void solveEnergy(int t, std::vector<double> &u, double &energy);
        void solveGrad(int t, std::vector<double> &u, std::vector<double> &grad);
        static void FeasibilityFuncJacWrapper(const alglib::real_1d_array &x, double &func, 
                                   alglib::real_1d_array &grad, void *ptr);
    
        void getAugHessian(std::vector<std::vector<std::vector<double> > > &AH);
        void getAugMeasMat(std::vector<std::vector<std::vector<double> > > &AM);
        void getAugForce(std::vector<std::vector<double> > &Af);
    
    
    public:
        void initEdgeCalc(std::vector<std::vector<double> > &meas);
        void setupEdgeCalcMats();
        void calcEdgeBasis();
        void calcEdgeResponse(std::vector<std::vector<double> > &meas);
        double solveEdgeUpdate(int i, std::vector<std::vector<double> > &meas);
        double removeEdge(int irem, std::vector<std::vector<double> > &meas);
        
}; 

struct LinOptParams {
	int t;
	LinSolver *solver;
	
};
    
#endif
