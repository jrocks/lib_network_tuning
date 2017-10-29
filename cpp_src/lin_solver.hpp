#ifndef LINSOLVER
#define LINSOLVER
    
#include "util.hpp"
#include "network.hpp"
#include "perturb.hpp"
#include "measure.hpp"
#include "lin_solver_state.hpp"
#include "lin_solver_result.hpp"
    
    
template <int DIM>
class LinSolver {
    public:
    
        // Network object
        Network nw;
    
        // Number of dof in Hessian
        int NDOF;
        // Number of node dofs
        int NNDOF;
        // Number of affine dofs;
        int NADOF;
    
    
        // Equilibrium Matrix
        SMat Q;
        // Vector of interaction strengths
        XVec K;
        // Matrix of fixed global dof constraints
        SMat G;
    
        // Hessian matrix
        Smat H;
    
        // Sparse matrix solver
        Eigen::CholmodSupernodalLLT<SMat > solver;
    
    
        // Number of independent functions (pert/meas pairs)
        int NF;
    
        // Number of constraints for each function
        std::vector<int> NC;
        // Number of measurements for each function
        std::vector<int> NM;
    
    
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
    
    public:
        LinSolver() {};
        LinSolver(Network &nw, int NF, std::vector<Perturb> &pert, std::vector<Measure> &meas);
    
        // Set interaction strengths
        void setK(RXVec K);
    
    
    private:
        // Setup various matrices
        void setupPert();
        void setupMeas();
        void setupEqMat();
        void setupConMat();
        void setupHessian();
    
    public:
        
    
        // Solve using current values of interaction strengths
        int solve(LinSolverResult &result);
        // Solve with a linear update to Hessian
        int solve(LinUpdate &up, LinSolverResult &result);
    
        // Solve and place result into solver state object
        int setSolverState(LinSolverState &state);
        // Update solver state object
        int updateSolverState(LinUpdate &up, LinSolverState &state);
    
        // Solve using information contained specified in solver state object (solver state must have previously been set, otherwise set it up)
        int solve(LinSolverState &state, LinSolverResult &result);
        // Solve using information contained specified in solver state object with update to Hessian
        int solve(LinUpdate &up, LinSolverState &state, LinSolverResult &result);   
        
}; 


class LinUpdate {
    public:
        
        // Update stretch moduli
        int NdK;
        std::vector<int> dK_edges;
        XVec::dK;
    
    Update() {}
    Update(int Ndk, std::vector<int> &dK_edges, RXVec dK) {
        this->Ndk = Ndk;
        this->dK_edges = dK_edges;
        this->dK = dK;
    }
    
};
#endif //LINSOLVER
