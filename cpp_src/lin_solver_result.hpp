#ifndef LINSOLVERRESULT
#define LINSOLVERRESULT
    
#include "util.hpp"
    
    
class LinSolverResult {
    public:
    
        int NF;
    
        // Number of dof in Hessian
        int NDOF;
        // Number of node dofs
        int NNDOF;
        // Number of affine dofs;
        int NADOF;
    
        // Number of constraints for each function
        std::vector<int> NC;
        // Number of measurements for each function
        std::vector<int> NM;
    
        
}; 
#endif //LINSOLVERRESULT