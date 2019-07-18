#ifndef OBJFUNC
#define OBJFUNC
    
#include "util.hpp"
#include "lin_solver_result.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
    
        XVec evalRes(RXVec& m);
        double evalFunc(RXVec& m);
        XMat evalResGrad(RXVec& m, RXMat& mgrad);
        XVec evalGrad(RXVec& m, RXMat& mgrad);
        
            
        
    
};

XVec LeastSquaresObjFunc::evalRes(RXVec& m) {
    
    XVec res = m;
    
    if(use_offset) {
        res -= offset;
    }
    
    
    res -= target;
    
    if(use_ineq) {
        for(int i = 0; i < NT; i++) {
            if((ineq[i] == 1 && res(i) >= 0.0) || (ineq[i] == -1 && res(i) <= 0.0)) {
                res(i) = 0.0;
            }
        } 
    }
    
    if(use_norm) {
        res = res.cwiseQuotient(norm);
    }
    
    return res;
    
}

double LeastSquaresObjFunc::evalFunc(RXVec& m) {
    
    XVec res = evalRes(m);
    
    return 0.5 * res.squaredNorm();
    
}

XMat LeastSquaresObjFunc::evalResGrad(RXVec& m, RXMat& mgrad) {
    
    XVec res = m;
    
    if(use_offset) {
        res -= offset;
    }
    
    
    res -= target;
    
    if(use_ineq) {
        for(int i = 0; i < NT; i++) {
            if((ineq[i] == 1 && res(i) >= 0.0) || (ineq[i] == -1 && res(i) <= 0.0)) {
                res(i) = 0.0;
            }
        } 
    }
    
    if(use_norm) {
        res = res.cwiseQuotient(norm);
    }
    
    XMat dresdK = XMat::Zero(mgrad.rows(), mgrad.cols());
    for(int i = 0; i < NT; i++) {
        if(res(i) == 0.0) {
            continue;
        }
        
        dresdK.row(i) = mgrad.row(i);
        
        if(use_norm) {
            dresdK.row(i) /= norm(i);
        }
        
    }
    
    return dresdK;
    
}

XVec LeastSquaresObjFunc::evalGrad(RXVec& m, RXMat& mgrad) {
    
    XVec res = evalRes(m);
    
    XMat dresdK = evalResGrad(m, mgrad);
    
    XVec dfdK = XVec::Zero(mgrad.cols());
    for(int i = 0; i < NT; i++) {
        if(res(i) == 0.0) {
            continue;
        }
        
        dfdK += res(i) * dresdK.row(i);

    }
    
    return dfdK;
    
}


#endif // OBJFUNC
