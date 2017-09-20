#ifndef ABSSOLVER
#define ABSSOLVER
    
#include "util.hpp"
    
class AbstractSolver {
    
    public:
    
        AbstractSolver() {};
        virtual ~AbstractSolver() {};
    
        // External interface using stl library
        // Return all dofs
        void solveU(std::vector<std::vector<double> > &u);
        // Returns all measurements
        void solveM(std::vector<double> &meas);
        // Returns gradient of each measurement
        void solveMGrad(std::vector<double> &meas, std::vector<std::vector<double> > &grad);
    
    
        // Internal C++ interface using eigen library
        // Returns all dofs
        virtual void isolveU(std::vector<XVec > &u) = 0;
        // Returns all measurements
        virtual void isolveM(XVec &meas) = 0;
        // Returns gradient of each measurement
        virtual void isolveMGrad(XVec &meas, std::vector<XVec > &grad) = 0;
        // Returns hessian of each measurement
        virtual void isolveMHess(XVec &meas, std::vector<XVec > &grad, std::vector<XMat > &hess) = 0;
    
}; 
    
#endif //ABSSOLVER
