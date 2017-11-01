#ifndef OBJFUNC
#define OBJFUNC
    
#include "util.hpp"
#include "lin_solver_result.hpp"


class LeastSquaresObjFunc {
    
    public: 
        int NT;
        XVec target;

        bool use_ineq;
        std::vector<int> ineq;

        bool use_offset;
        XVec offset;

        bool use_norm;
        XVec norm;
    
    public:
        
        LeastSquaresObjFunc(int NT, RXVec target) {
            this->NT = NT;
            this->target = target;
            
            use_ineq = false;
            use_offset = false;
            use_norm = false;
        }
    
        void setIneq(std::vector<int> &ineq) {
            this->ineq = ineq;
        }
    
        void setOffset(RXVec offset) {
            this->offset = offset;
        }
    
        void setNorm(RXVec norm) {
            this->norm = norm;
        }
    
        double evalFunc(LinSolverResult &result);
            
        
    
};

double LeastSquaresObjFunc::evalFunc(LinSolverResult &result) {
    
    XVec m(NT);
        
    int index = 0;
    for(int t = 0; t < result.NF; t++) {
        m.segment(index, result.ostrain[t].size()) = result.ostrain[t];
        index += result.ostrain[t].size();
        
        m.segment(index, result.ostress[t].size()) = result.ostress[t];
        index += result.ostress[t].size();
        
        m.segment(index, result.affine_strain[t].size()) = result.affine_strain[t];
        index += result.affine_strain[t].size();
        
        m.segment(index, result.affine_stress[t].size()) = result.affine_stress[t];
        index += result.affine_stress[t].size();
        
        m.segment(index, result.olambda[t].size()) = result.olambda[t];
        index += result.olambda[t].size();
    }
    
    if(use_offset) {
        m -= offset;
    }
    
    if(use_norm) {
        m = m.cwiseQuotient(norm);
    }
    
    m -= target;
    
    if(use_ineq) {
        for(int i = 0; i < NT; i++) {
            if((ineq[i] == 1 && m(i) >= 0.0) || (ineq[i] == -1 && m(i) <= 0.0)) {
                m(i) = 0.0;
            }
        } 
    }
    
    return 0.5 * m.squaredNorm();
    
}

#endif // OBJFUNC
