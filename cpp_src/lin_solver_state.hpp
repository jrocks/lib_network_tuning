#ifndef LINSOLVERSTATE
#define LINSOLVERSTATE
    
#include "util.hpp"
    
    
class LinSolverState {
    public:
    
        // Updated vector of interaction strengths
        XVec K;
        // Cumulative change in Hessian
        SMat dH;
        // Cumulative change in inverse Hessian
        XMat dHinv;
        // Inverted Hessian times linear coeffs
        std::vector<XMat > HinvC1;
        // Inverted Hessian times force
        std::vector<XVec > Hinvf;
    
        // Inverted Hessian times equilibrium matrix (only store columns as needed)
        XMat HinvQ;
        // Array of booleans indicating which columns HinvQ have been solved for (true indicates it already exists)
        std::vector<bool> have_HinvQ;
    
        
}; 
#endif //LINSOLVERSTATE