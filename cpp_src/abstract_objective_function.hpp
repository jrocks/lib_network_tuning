#ifndef ABSOBJFUNC
#define ABSOBJFUNC
    
#include "util.hpp"
#include "abstract_solver.hpp"


class AbstractObjFunc { 
    
    public:
        
        AbstractObjFunc() {};
        virtual ~AbstractObjFunc() {};
    
        virtual void res(AbstractSolver &solver, std::vector<double> &res) = 0;
        virtual void resGrad(AbstractSolver &solver, std::vector<double> &res, 
                             std::vector<std::vector<double> > &res_grad) = 0;
        virtual void func(AbstractSolver &solver, double &obj) = 0;
        virtual void funcGrad(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad) = 0;
        virtual void funcHess(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad, 
                              std::vector<std::vector<double> > &obj_hess) = 0;
    
        virtual void getConstraints(std::vector<double> &C, std::vector<int> &CT) = 0;
};

#endif //ABSOBJFUNC
