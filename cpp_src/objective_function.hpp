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
            this->use_ineq = true;
            this->ineq = ineq;
        }
    
        void setOffset(RXVec offset) {
            this->use_offset = true;
            this->offset = offset;
        }
    
        void setNorm(RXVec norm) {
            this->use_norm = true;
            this->norm = norm;
        }
    
        double evalFunc(RXVec m);
            
        
    
};

double LeastSquaresObjFunc::evalFunc(RXVec m) {
    
    XVec res = m;
    
    if(use_offset) {
        res -= offset;
    }
    
    if(use_norm) {
        res = res.cwiseQuotient(norm);
    }
    
    res -= target;
    
    if(use_ineq) {
        for(int i = 0; i < NT; i++) {
            if((ineq[i] == 1 && res(i) >= 0.0) || (ineq[i] == -1 && res(i) <= 0.0)) {
                res(i) = 0.0;
            }
        } 
    }
    
    return 0.5 * res.squaredNorm();
    
}

#endif // OBJFUNC
