#ifndef LINSOLVERRESULT
#define LINSOLVERRESULT
    
#include "util.hpp"
    
    
class LinSolverResult {
    public:

        int NF;
    
        std::vector<XVec > disp;
        std::vector<XVec > strain;
        std::vector<XVec > lamb;
    
        std::vector<XVec > ostrain;
        std::vector<XVec > ostress;
    
        std::vector<XVec > affine_strain;
        std::vector<XVec > affine_stress;
    
        void setNF(int NF) {
            this->NF = NF;
            disp.resize(NF);
            strain.resize(NF);
            lamb.resize(NF);
            
            ostrain.resize(NF);
            ostress.resize(NF);
            
            affine_strain.resize(NF);
            affine_stress.resize(NF);
        }
    
        
}; 

#endif // LINSOLVERRESULT