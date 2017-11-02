#ifndef LINSOLVERSTATE
#define LINSOLVERSTATE
    
#include "util.hpp"
    
    
class LinSolverState {
    public:
    
        // Inverted Hessian times linear coeffs
        std::vector<XMat > HiC1;
        // Inverted Hessian times force
        std::vector<XMat > Hif;
    
    
        bool hess_update;
        // Updated vector of interaction strengths
        XVec K;
        // Cumulative change in Hessian
        SMat dH;
        // Cumulative change in inverse Hessian
        XMat dHi;
        
        LinSolverState(int NF) {

            hess_update = false;
            
            HiC1.resize(NF);
            Hif.resize(NF);

        
        }; 
}; 
#endif //LINSOLVERSTATE