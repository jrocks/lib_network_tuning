#ifndef OBJFUNC
#define OBJFUNC
    
#include "util.hpp"
#include "abstract_solver.hpp"
#include "abstract_objective_function.hpp"
#include "lin_solver.hpp"


class AugIneqRatioChangeObjFunc {

    XVec ratio_init;
    XVec delta_ratio_target;
    
    XVec x_reg;
    double mu;
    
    std::vector<XVec > fd;
    
    double a, b, c;
    
    public:
        AugIneqRatioChangeObjFunc() {};
        AugIneqRatioChangeObjFunc(std::vector<double> &delta_ratio_target);
    
        void initialize(LinSolver &solver, std::vector<double> &x_init);
        void setWeights(double a, double b, double c);
        void setRegularize(double mu, std::vector<double> &x_reg);
    
        void res(std::vector<double> &x, LinSolver &solver, std::vector<double> &res);
        void resGrad(std::vector<double> &x, LinSolver &solver, std::vector<double> &res,
                                        std::vector<int> &rows, std::vector<int> &cols, std::vector<double> &vals);
        void func(std::vector<double> &x, LinSolver &solver, double &obj);
        void funcGrad(std::vector<double> &x, LinSolver &solver, double &obj, std::vector<double> &obj_grad);
        void funcHess(std::vector<double> &x, LinSolver &solver,
                                        std::vector<int> &rows, std::vector<int> &cols, std::vector<double> &vals);
    
        // void funcTerms(std::vector<double> &x, LinSolver &solver, std::vector<double> &terms);
        // void funcTermsGrad(std::vector<double> &x, LinSolver &solver, std::vector<double> &terms_grad);
        
    
};
    
class IneqRatioChangeObjFunc: public AbstractObjFunc {

    int Nterms;
    int Ngrad;
    XVec ratio_init;
    XVec delta_ratio_target;
    
    public:
        IneqRatioChangeObjFunc() {};
        IneqRatioChangeObjFunc(int Nterms, int Ngrad, std::vector<double> &ratio_init, std::vector<double> &delta_ratio_target) {
            this->Nterms = Nterms;
            this->Ngrad = Ngrad;
            vectorToEigen(ratio_init, this->ratio_init);
            vectorToEigen(delta_ratio_target, this->delta_ratio_target);
        };
    
        void setRatioInit(std::vector<double> &ratio_init);
    
        void res(AbstractSolver &solver, std::vector<double> &res);
        void resGrad(AbstractSolver &solver, std::vector<double> &res, 
                             std::vector<std::vector<double> > &res_grad);
        void func(AbstractSolver &solver, double &obj);
        void funcGrad(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad);
        void funcHess(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad, 
                              std::vector<std::vector<double> > &obj_hess);
    
        void objFuncTerms(std::vector<double> &meas, double &obj, std::vector<double> &terms);
        void objFunc(std::vector<double> &meas, double &obj);
        void objFuncGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<double> &obj_grad);
    
        void projMeas(std::vector<double> &meas, std::vector<double> &pmeas);
        void projGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<std::vector<double> > &pgrad);
    
        void getConstraints(std::vector<double> &C, std::vector<int> &CT);
        
    
};

class EqRatioChangeObjFunc: public AbstractObjFunc {

    int Nterms;
    int Ngrad;
    XVec ratio_init;
    XVec delta_ratio_target;
    
    public:
        EqRatioChangeObjFunc() {};
        EqRatioChangeObjFunc(int Nterms, int Ngrad, std::vector<double> &ratio_init, std::vector<double> &delta_ratio_target) {
            this->Nterms = Nterms;
            this->Ngrad = Ngrad;
            vectorToEigen(ratio_init, this->ratio_init);
            vectorToEigen(delta_ratio_target, this->delta_ratio_target);
        };
    
        void setRatioInit(std::vector<double> &ratio_init);
    
        void res(AbstractSolver &solver, std::vector<double> &res);
        void resGrad(AbstractSolver &solver, std::vector<double> &res, 
                             std::vector<std::vector<double> > &res_grad);
        void func(AbstractSolver &solver, double &obj);
        void funcGrad(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad);
        void funcHess(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad, 
                              std::vector<std::vector<double> > &obj_hess);
    
        void objFuncTerms(std::vector<double> &meas, double &obj, std::vector<double> &terms);
        void objFunc(std::vector<double> &meas, double &obj);
        void objFuncGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<double> &obj_grad);
    
        void projMeas(std::vector<double> &meas, std::vector<double> &pmeas);
        void projGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<std::vector<double> > &pgrad);
    
        void getConstraints(std::vector<double> &C, std::vector<int> &CT);
        
    
};

class IneqRatioObjFunc: public AbstractObjFunc {

    int Nterms;
    int Ngrad;
    XVec ratio_init;
    XVec delta_ratio_target;
    
    public:
        IneqRatioObjFunc() {};
        IneqRatioObjFunc(int Nterms, int Ngrad, std::vector<double> &ratio_init, std::vector<double> &delta_ratio_target) {
            this->Nterms = Nterms;
            this->Ngrad = Ngrad;
            vectorToEigen(ratio_init, this->ratio_init);
            vectorToEigen(delta_ratio_target, this->delta_ratio_target);
        };
    
        void setRatioInit(std::vector<double> &ratio_init);
    
        void res(AbstractSolver &solver, std::vector<double> &res);
        void resGrad(AbstractSolver &solver, std::vector<double> &res, 
                             std::vector<std::vector<double> > &res_grad);
        void func(AbstractSolver &solver, double &obj);
        void funcGrad(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad);
        void funcHess(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad, 
                              std::vector<std::vector<double> > &obj_hess);
    
        void objFuncTerms(std::vector<double> &meas, double &obj, std::vector<double> &terms);
        void objFunc(std::vector<double> &meas, double &obj);
        void objFuncGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<double> &obj_grad);
    
        void projMeas(std::vector<double> &meas, std::vector<double> &pmeas);
        void projGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<std::vector<double> > &pgrad);
    
        void getConstraints(std::vector<double> &C, std::vector<int> &CT);
        
    
};


#endif //OBJFUNC
