#include "objective_function.hpp"
   
AugIneqRatioChangeObjFunc::AugIneqRatioChangeObjFunc(std::vector<double> &delta_ratio_target) {
    
    vectorToEigen(delta_ratio_target, this->delta_ratio_target);
    
    a = 1.0;
    b = 1.0;
    c = 1.0;
}

void AugIneqRatioChangeObjFunc::setWeights(double a, double b, double c) {
    this->a = a;
    this->b = b;
    this->c = c;
}

void AugIneqRatioChangeObjFunc::setRegularize(double mu, std::vector<double> &x_reg) {
    XVecMap _x_reg(x_reg.data(), x_reg.size());
    this->x_reg = _x_reg;
    
    this->mu = mu;
}

void AugIneqRatioChangeObjFunc::initialize(LinSolver &solver, std::vector<double> &x_init) {    
    int NE = solver.nw.NE;
    int NDOF = solver.NDOF;
    int NC = solver.NC[0];    
    
    fd.resize(DIM);
    for(int d = 0; d < DIM; d++) {
        fd[d] = XVec::Constant(solver.NDOF, -1.0);
        fd[d](d) = (solver.NDOF - 1.0);
    }
    
    solver.isolveM(ratio_init);
    
    x_init.resize(NE+NDOF+NC+DIM*NDOF);
    
    XVecMap _x_init(x_init.data(), x_init.size());
    
    _x_init.segment(0, NE) = solver.K;
        
    std::vector<XVec > u;
    solver.isolveU(u);
    _x_init.segment(NE, NDOF) = u[0];
    
    std::vector<XVec > lambda;
    solver.isolveLambda(lambda);
    _x_init.segment(NE+NDOF, NC) = lambda[0];
    
    for(int d = 0; d < DIM; d++) {
        _x_init.segment(NE+NDOF+NC+d*NDOF, NDOF) = solver.solver.solve(fd[d]);
        if (solver.solver.info() != Eigen::Success) {
            std::cout << "Solving H^{-1}f failed." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }    
}

void AugIneqRatioChangeObjFunc::res(std::vector<double> &x, LinSolver &solver, std::vector<double> &res) {
    int NE = solver.nw.NE;
    int NDOF = solver.NDOF;
    int NC = solver.NC[0];
    int NM = solver.NM[0];
    int NVars = NE + NDOF+NC + DIM*NDOF;
    
    XVecMap _x(x.data(), NVars);
    XVecMap K(x.data(), NE);
    XVecMap u(x.data()+NE, NDOF);
    XVecMap lambda(x.data()+NE+NDOF, NC);
    XVecMap ud(x.data()+NE+NDOF+NC, DIM*NDOF);

    res.resize(NM+NDOF+NC+DIM*NDOF);
    XVecMap _res(res.data(), res.size());
        
    XVec meas = solver.M[0].transpose() * u;
    
   
    _res.segment(0, NM) = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    

    SMat I(NDOF, NDOF);
    std::vector<Trip> I_trip_list;
    for(int i = 0; i < NDOF; i++) {
        I_trip_list.push_back(Trip(i, i, 0.0));
    }
    I.setFromTriplets(I_trip_list.begin(), I_trip_list.end());
    
    SMat H = solver.Q * K.asDiagonal() * solver.Q.transpose() + solver.G + I;
    _res.segment(NM, NDOF) = H * u - solver.C1[0]*lambda;
    _res.segment(NM, NDOF) -= solver.f[0];
    _res.segment(NM+NDOF, NC) = -solver.C1[0].transpose() * u - solver.C0[0];
    
    
    for(int d = 0; d < DIM; d++) {
        _res.segment(NM+NDOF+NC+d*NDOF, NDOF) = H * ud.segment(d*NDOF, NDOF) - fd[d];
    }
    
    _res.segment(0, NM) *= sqrt(a);
    _res.segment(NM, NDOF+NC) *= sqrt(b);
    _res.segment(NM+NDOF+NC, DIM*NDOF) *= sqrt(c);
    
    // _res.segment(NM+NDOF+NC+DIM*NDOF, NVars) = sqrt(mu) * (_x - x_reg);
    
    XMat H_tmp = H;
    double matnorm = 0;
    for(int i = 0; i < NDOF; i++) { 
        double tmp = H_tmp.col(i).lpNorm<1>();
        if(tmp > matnorm) {
            matnorm = tmp;
        }
    }
    XVec u_tmp = ud.segment(0, NDOF);
    double vecnorm = u_tmp.lpNorm<1>();
    double gamma = NDOF * 2.22045e-16 / (1.0 - NDOF * 2.22045e-16);
    std::cout << matnorm << "\t" << vecnorm << "\t" << gamma << "\t" << matnorm * vecnorm * gamma << std::endl;
    
    
}




void AugIneqRatioChangeObjFunc::resGrad(std::vector<double> &x, LinSolver &solver, std::vector<double> &res,
                                        std::vector<int> &rows, std::vector<int> &cols, std::vector<double> &vals) {
    int NE = solver.nw.NE;
    int NDOF = solver.NDOF;
    int NC = solver.NC[0];
    int NM = solver.NM[0];
    int NVars = NE + NDOF+NC + DIM*NDOF;
    // int NRes = NM + NDOF+NC + DIM*NDOF;

    XVecMap _x(x.data(), NVars);
    XVecMap K(x.data(), NE);
    XVecMap u(x.data()+NE, NDOF);
    XVecMap lambda(x.data()+NE+NDOF, NC);
    XVecMap ud(x.data()+NE+NDOF+NC, DIM*NDOF);
    
    
    res.resize(NM+NDOF+NC+DIM*NDOF);
    XVecMap _res(res.data(), res.size());
        
    
    XVec meas = solver.M[0].transpose() * u;
    _res.segment(0, NM) = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    
    
    SMat I(NDOF, NDOF);
    std::vector<Trip> I_trip_list;
    for(int i = 0; i < NDOF; i++) {
        I_trip_list.push_back(Trip(i, i, 0.0));
    }
    I.setFromTriplets(I_trip_list.begin(), I_trip_list.end());
    
    SMat H = solver.Q * K.asDiagonal() * solver.Q.transpose() + solver.G + I;
    _res.segment(NM, NDOF) = H * u - solver.C1[0]*lambda;
    _res.segment(NM, NDOF) -= solver.f[0];
    _res.segment(NM+NDOF, NC) = -solver.C1[0].transpose() * u - solver.C0[0];
    
    for(int d = 0; d < DIM; d++) {
        _res.segment(NM+NDOF+NC+d*NDOF, NDOF) = H * ud.segment(d*NDOF, NDOF) - fd[d];
    }
    
    _res.segment(0, NM) *= sqrt(a);
    _res.segment(NM, NDOF+NC) *= sqrt(b);
    _res.segment(NM+NDOF+NC, DIM*NDOF) *= sqrt(c);
    
    // _res.segment(NM+NDOF+NC+DIM*NDOF, NVars) = sqrt(mu) * (_x - x_reg);
        
    SMat block;
    // Gradient for first NM terms
    for(int i = 0; i < NM; i++) {
        if(_res(i) > 0.0) {
            block = -solver.M[0].transpose().row(i) / delta_ratio_target(i);
            block *= sqrt(a);
            insert_sparse_block(block, i, NE, rows, cols, vals);           
        }
    }
    
    // Gradient for next NDOF terms
    
    // wrt spring constants
    for(int bi = 0; bi < NE; bi++) {
        block = solver.Q.col(bi) * (solver.Q.col(bi).transpose() * u);
        block *= sqrt(b);
        insert_sparse_block(block, NM, bi, rows, cols, vals);          
    }
    // wrt displacements
    block = H;
    block *= sqrt(b);
    insert_sparse_block(block, NM, NE, rows, cols, vals);      
    // wrt langrange multipliers
    block = -solver.C1[0];
    block *= sqrt(b);
    insert_sparse_block(block, NM, NE+NDOF, rows, cols, vals);
    
    // Gradient for next NC terms
    //wrt displacements
    block = -solver.C1[0].transpose();
    block *= sqrt(b);
    insert_sparse_block(block, NM+NDOF, NE, rows, cols, vals);
    
    //Gradient for next DIM*NDOF terms
    // wrt spring constants
    for(int d = 0; d < DIM; d++) {
        for(int bi = 0; bi < NE; bi++) {
            block = solver.Q.col(bi) * (solver.Q.col(bi).transpose() * ud.segment(d*NDOF, NDOF));
            block *= sqrt(c);
            insert_sparse_block(block, NM+NDOF+NC+d*NDOF, bi, rows, cols, vals);
        }
    }
    // wrt displacements
    for(int d = 0; d < DIM; d++) {
        block = H;
        block *= sqrt(c);
        insert_sparse_block(block, NM+NDOF+NC+d*NDOF, NE+NDOF+NC+d*NDOF, rows, cols, vals);
    }
    
    // for(int i = 0; i < NVars; i++) {
    //     rows.push_back(NM+NDOF+NC+DIM*NDOF+i);
    //     cols.push_back(i);
    //     vals.push_back(sqrt(mu));
    // }
}


void AugIneqRatioChangeObjFunc::func(std::vector<double> &x, LinSolver &solver, double &obj) {
    
//     int NE = solver.nw.NE;
//     int NDOF = solver.NDOF;
//     int NC = solver.NC[0];
//     int NM = solver.NM[0];
    
    std::vector<double> _res;
    
    res(x, solver, _res);
    
    XVecMap res_map(_res.data(), _res.size());
    
    // double obj1 = 0.5 * res_map.segment(0, NM).squaredNorm();
    // double obj2 = 0.5 * res_map.segment(NM, NDOF+NC).squaredNorm();
    // double obj3 = 0.5 * res_map.segment(NDOF+NC, DIM*NDOF).squaredNorm();
    obj = 0.5 * res_map.squaredNorm();
    
    // std::cout << obj << "\t" << obj1 << "\t" << obj2 << "\t" << obj3 << std::endl;
    
}

void AugIneqRatioChangeObjFunc::funcGrad(std::vector<double> &x, LinSolver &solver, double &obj, std::vector<double> &obj_grad) {
    
    int NE = solver.nw.NE;
    int NDOF = solver.NDOF;
    int NC = solver.NC[0];  
    int NM = solver.NM[0];
    int NVars = NE + NDOF+NC + DIM*NDOF;
    
    std::vector<double> res;
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
    resGrad(x, solver, res, rows, cols, vals);
    
    
    XVecMap _res(res.data(), res.size());
    obj = 0.5 * _res.squaredNorm();
    
    std::vector<Trip> trip_list;
    
    for(int i = 0; i < int(rows.size()); i++) {
        trip_list.push_back(Trip(rows[i], cols[i], vals[i]));
    }
        
    SMat g1(NM+NDOF+NC+DIM*NDOF+NVars, NVars);
    g1.setFromTriplets(trip_list.begin(), trip_list.end());
    
    SMat g = g1.transpose();
    
    // std::cout << g << std::endl;
    
    
    obj_grad.resize(NVars, 0.0);
    XVecMap _obj_grad(obj_grad.data(), obj_grad.size());
    
    for(int i = 0; i < int(res.size()); i++) {
        
        _obj_grad += _res(i) * g.col(i); 
    }
    
}

void AugIneqRatioChangeObjFunc::funcHess(std::vector<double> &x, LinSolver &solver,
                                        std::vector<int> &rows, std::vector<int> &cols, std::vector<double> &vals) {
    int NE = solver.nw.NE;
    int NDOF = solver.NDOF;
    int NC = solver.NC[0];
    int NM = solver.NM[0];
    int NVars = NE + NDOF+NC + DIM*NDOF;
    int NRes = NM + NDOF+NC + DIM*NDOF;

    XVecMap _x(x.data(), NVars);
    XVecMap K(x.data(), NE);
    XVecMap u(x.data()+NE, NDOF);
    XVecMap lambda(x.data()+NE+NDOF, NC);
    XVecMap ud(x.data()+NE+NDOF+NC, DIM*NDOF);
    
    XVec meas = solver.M[0].transpose() * u;
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    
    
    SMat H = solver.Q * K.asDiagonal() * solver.Q.transpose() + solver.G;
    
    SMat block, blockT;
    
    XVec qiu = XVec::Zero(NE);
    std::vector<XVec > qiud(DIM, XVec::Zero(NE));
    for(int bi = 0; bi < NE; bi++) {
        qiu(bi) = (solver.Q.col(bi).transpose() * u)(0);
        for(int d = 0; d < DIM; d++) {
            qiud[d](bi) = (solver.Q.col(bi).transpose() * ud.segment(d*NDOF, NDOF))(0);
        }
    }
    
    XMat kkblock = XMat::Zero(NE, NE);
    for(int bi = 0; bi < NE; bi++) {
        double qiqi = SMat(solver.Q.col(bi).transpose() * solver.Q.col(bi)).coeffRef(0, 0);
        
        kkblock(bi, bi) += qiu(bi) * qiqi * qiu(bi);
        for(int d = 0; d < DIM; d++) {
            kkblock(bi, bi) += qiud[d](bi) * qiqi * qiud[d](bi);
        }
        
        for(int bj = bi+1; bj < NE; bj++) {
            double qiqj = SMat(solver.Q.col(bi).transpose() * solver.Q.col(bj)).coeffRef(0, 0);
        
            kkblock(bi, bj) += qiu(bi) * qiqj * qiu(bj);
            kkblock(bj, bi) += qiu(bj) * qiqj * qiu(bi);
            for(int d = 0; d < DIM; d++) {
                kkblock(bi, bj) += qiud[d](bi) * qiqj * qiud[d](bj);
                kkblock(bj, bi) += qiud[d](bj) * qiqj * qiud[d](bi);
            }
        }
        
        SMat Hqi = H * solver.Q.col(bi);
        XVec Hu = H * u - solver.C1[0] * lambda;
        Hu -= solver.f[0];
        
        block = Hqi.transpose() * qiu(bi) 
            + (Hu.transpose() * solver.Q.col(bi)) * solver.Q.col(bi).transpose();
        insert_sparse_block(block, bi, NE, rows, cols, vals);
        blockT = block.transpose();
        insert_sparse_block(blockT, NE, bi, rows, cols, vals);
        
        block = -qiu(bi) * solver.Q.col(bi).transpose() * solver.C1[0];
        insert_sparse_block(block, bi, NE+NDOF, rows, cols, vals);
        blockT = block.transpose();
        insert_sparse_block(blockT, NE+NDOF, bi, rows, cols, vals);
        
        for(int d = 0; d < DIM; d++) {
            XVec Hud = H * ud.segment(d*NDOF, NDOF) - fd[d];
            block = Hqi.transpose() * qiud[d](bi)
                + (Hud.transpose() * solver.Q.col(bi)) * solver.Q.col(bi).transpose();
            insert_sparse_block(block, bi, NE+NDOF+NC+d*NDOF, rows, cols, vals);
            blockT = block.transpose();
            insert_sparse_block(blockT, NE+NDOF+NC+d*NDOF, bi, rows, cols, vals);  
        }
    }
    
    for(int bi = 0; bi < NE; bi++) {
        for(int bj = 0; bj < NE; bj++) {
            rows.push_back(bi);
            cols.push_back(bj);
            vals.push_back(kkblock(bi, bj));
        }
    }
    
    block = H * H + solver.C1[0] * solver.C1[0].transpose();
    for(int i = 0; i < NM; i++) {
        if(_res(i) > 0.0) {
            block += (solver.M[0].col(i) * solver.M[0].col(i).transpose()) / (delta_ratio_target(i)*delta_ratio_target(i));
        }
    }
    insert_sparse_block(block, NE, NE, rows, cols, vals); 
    
    block = -H * solver.C1[0];
    insert_sparse_block(block, NE, NE+NDOF, rows, cols, vals);
    blockT = block.transpose();
    insert_sparse_block(blockT, NE+NDOF, NE, rows, cols, vals);
    
    block = solver.C1[0].transpose() * solver.C1[0];
    insert_sparse_block(block, NE+NDOF, NE+NDOF, rows, cols, vals);

    for(int d = 0; d < DIM; d++) {
        block = H * H;
        insert_sparse_block(block, NE+NDOF+NC+d*NDOF, NE+NDOF+NC+d*NDOF, rows, cols, vals);
    }
    
}

// void AugIneqRatioChangeObjFunc::funcTerms(std::vector<double> &x, LinSolver &solver, std::vector<double> &terms) {
       
//     int NE = solver.nw.NE;
//     int NDOF = solver.NDOF;
//     int NC = solver.NC[0];
//     int NM = solver.NM[0];
    
//     std::vector<double> _res;
    
//     res(x, solver, _res);
    
//     XVecMap res_map(_res.data(), _res.size());
    
//     terms.resize()
    
//     // double obj1 = 0.5 * res_map.segment(0, NM).squaredNorm();
//     // double obj2 = 0.5 * res_map.segment(NM, NDOF+NC).squaredNorm();
//     // double obj3 = 0.5 * res_map.segment(NDOF+NC, DIM*NDOF).squaredNorm();
//     obj = 0.5 * res_map.squaredNorm();
// }

// void AugIneqRatioChangeObjFunc::funcTermsGrad(std::vector<double> &x, LinSolver &solver, std::vector<double> &terms_grad) {
    
// }


// void AugIneqRatioChangeObjFunc::func(AbstractSolver &solver, double &obj) {
    
//     XVec meas;
//     solver.isolveM(meas);
    
//     XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
//     obj = 0.5 * _res.squaredNorm();
    
// }

// void AugIneqRatioChangeObjFunc::funcGrad(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad) {
    
//     XVec meas;
//     std::vector<XVec > grad;
//     solver.isolveMGrad(meas, grad);
    
//     XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
//     obj = 0.5 * _res.squaredNorm();
        
//     XVec _obj_grad = XVec::Zero(Ngrad);
    
//     for(int i = 0; i < meas.size(); i++) {
//         if(_res(i) > 0.0) {
//             _obj_grad += -_res(i) / delta_ratio_target(i) * grad[i];
//         }
//     }
    
//     eigenToVector(_obj_grad, obj_grad);
// }
    
    

void IneqRatioChangeObjFunc::setRatioInit(std::vector<double> &ratio_init) {
    vectorToEigen(ratio_init, this->ratio_init);
}


void IneqRatioChangeObjFunc::res(AbstractSolver &solver, std::vector<double> &res) {
    XVec meas;
    solver.isolveM(meas);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    // Could use a map to avoid an extra copy
    eigenToVector(_res, res);
}

void IneqRatioChangeObjFunc::resGrad(AbstractSolver &solver, std::vector<double> &res, 
                     std::vector<std::vector<double> > &res_grad) {
    
    XVec meas;
    std::vector<XVec > grad;
    solver.isolveMGrad(meas, grad);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    
    res_grad.resize(Nterms);
    eigenToVector(_res, res);
    
    for(int i = 0; i < meas.size(); i++) {
        if(_res(i) > 0.0) {
            XVec _res_grad = -grad[i] / delta_ratio_target(i);
            eigenToVector(_res_grad, res_grad[i]);
        } else {
            res_grad[i].resize(grad[i].size(), 0.0);
        }
    }
}

void IneqRatioChangeObjFunc::func(AbstractSolver &solver, double &obj) {
    
    XVec meas;
    solver.isolveM(meas);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    obj = 0.5 * _res.squaredNorm();
    
}

void IneqRatioChangeObjFunc::funcGrad(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad) {
    
    XVec meas;
    std::vector<XVec > grad;
    solver.isolveMGrad(meas, grad);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    obj = 0.5 * _res.squaredNorm();
        
    XVec _obj_grad = XVec::Zero(Ngrad);
    
    for(int i = 0; i < meas.size(); i++) {
        if(_res(i) > 0.0) {
            _obj_grad += -_res(i) / delta_ratio_target(i) * grad[i];
        }
    }
    
    eigenToVector(_obj_grad, obj_grad);
}

void IneqRatioChangeObjFunc::funcHess(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad, 
                      std::vector<std::vector<double> > &obj_hess) {
    
        
    XVec meas;
    std::vector<XVec > grad;
    std::vector<XMat > hess;
    solver.isolveMHess(meas, grad, hess);
        
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    obj = 0.5 * _res.squaredNorm();
        
    XVec _obj_grad = XVec::Zero(Ngrad);
    XMat _obj_hess = XMat::Zero(Ngrad, Ngrad);
    
    for(int i = 0; i < meas.size(); i++) {
        if(_res(i) > 0.0) {
            XVec _res_grad = -grad[i] / delta_ratio_target(i);
            _obj_grad += _res(i) * _res_grad;
            
            XMat _res_hess = -hess[i] / delta_ratio_target(i);
            
            _obj_hess += _res_grad * _res_grad.transpose() + _res(i) * _res_hess;
        }
    }
    eigenToVector(_obj_grad, obj_grad);
    eigenMatToVector(_obj_hess, obj_hess);
    
}

    
void IneqRatioChangeObjFunc::projMeas(std::vector<double> &meas, std::vector<double> &pmeas) {

    XVecMap _meas(meas.data(), meas.size());
    XVec delta_ratio;
    
    if(relative && change) {
        delta_ratio = (_meas - ratio_init).cwiseQuotient(ratio_init);
    } else if(!relative && change) {
        delta_ratio = _meas - ratio_init;
    } else if(relative & !change) {
        delta_ratio = _meas.cwiseQuotient(ratio_init);
    } else if(!relative & !change) {
        delta_ratio = _meas;
    }
        
    XVec _pmeas = XVec::Zero(Nterms);
    
    for(int i = 0; i < Nterms; i++) {
        // double penalty = delta_ratio(i) / delta_ratio_target(i) - 1.0;
        // if(penalty < 0.0) {
        //     _pmeas[i] = penalty;
        // }
        
        
        double penalty = delta_ratio(i) - delta_ratio_target(i);
        if((ineq(i) > 0 && penalty < 0.0) || (ineq(i) < 0 && penalty > 0.0)) {
            _pmeas[i] = penalty;
        }
    }
        
    eigenToVector(_pmeas, pmeas);
    
}


void IneqRatioChangeObjFunc::objFuncTerms(std::vector<double> &meas, double &obj, std::vector<double> &terms) {
      
    std::vector<double> pmeas;
    projMeas(meas, pmeas);
    
    
    XVec _terms = XVec::Zero(Nterms);
    XVec _pmeas;
    vectorToEigen(pmeas, _pmeas);
    
    _terms = _pmeas.array().pow(2.0).matrix();
    
    obj = 0.5 * _terms.sum();
    
    eigenToVector(_terms, terms);
    
}

void IneqRatioChangeObjFunc::objFunc(std::vector<double> &meas, double &obj) {
    
    std::vector<double> terms;
    objFuncTerms(meas, obj, terms);
    
}

void IneqRatioChangeObjFunc::projGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<std::vector<double> > &pgrad) {
    
    pgrad.resize(Nterms);
    
    XVecMap _meas(meas.data(), meas.size());
    
    // XVec delta_ratio = _meas - ratio_init;
    
    
    XVec delta_ratio = (_meas - ratio_init).cwiseQuotient(ratio_init);

    
    for(int i = 0; i < Nterms; i++) {
        
        XVecMap g_row(meas_grad[i].data(), meas_grad[i].size());
        
        XVec grad = XVec::Zero(Ngrad);
        double penalty = delta_ratio(i) / delta_ratio_target(i) - 1.0;
        if(penalty < 0.0) {
            grad = g_row / ratio_init(i) / delta_ratio_target(i);
        }
        
        eigenToVector(grad, pgrad[i]);
    }
    
}


void IneqRatioChangeObjFunc::objFuncGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<double> &obj_grad) {
    
    
    
    XVec grad = XVec::Zero(Ngrad);
    
    XVecMap _meas(meas.data(), meas.size());
    
    // XVec delta_ratio = _meas - ratio_init;
    
    XVec delta_ratio = (_meas - ratio_init).cwiseQuotient(ratio_init);

    
    for(int i = 0; i < Nterms; i++) {
        
        XVecMap g_row(meas_grad[i].data(), meas_grad[i].size());
        
        double penalty = delta_ratio(i) / delta_ratio_target(i) - 1.0;
        if(penalty < 0.0) {
            grad += penalty * g_row / ratio_init(i) / delta_ratio_target(i);
        }
    }
    
    eigenToVector(grad, obj_grad);
    
}

void IneqRatioChangeObjFunc::getConstraints(std::vector<double> &C, std::vector<int> &CT) {
    
    XVec _C = delta_ratio_target + ratio_init;
    eigenToVector(_C, C);
    
    XiVec _CT = XiVec::Zero(Nterms);
    for(int i = 0; i < Nterms; i++) {
        if(delta_ratio_target(i) > 0.0) {
            _CT(i) = 1;
        } else {
            _CT(i) = -1;
        }
    }
    
    eigenToVector(_CT, CT);
    
    
};

void EqRatioChangeObjFunc::setRatioInit(std::vector<double> &ratio_init) {
    vectorToEigen(ratio_init, this->ratio_init);
}


void EqRatioChangeObjFunc::res(AbstractSolver &solver, std::vector<double> &res) {
    XVec meas;
    solver.isolveM(meas);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    // Could use a map to avoid an extra copy
    eigenToVector(_res, res);
}

void EqRatioChangeObjFunc::resGrad(AbstractSolver &solver, std::vector<double> &res, 
                     std::vector<std::vector<double> > &res_grad) {
    
    XVec meas;
    std::vector<XVec > grad;
    solver.isolveMGrad(meas, grad);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    
    res_grad.resize(Nterms);
    
    for(int i = 0; i < meas.size(); i++) {
        if(_res(i) > 0.0) {
            XVec _res_grad = -grad[i] / delta_ratio_target(i);
            eigenToVector(_res_grad, res_grad[i]);
        } else {
            res_grad[i].resize(grad[i].size(), 0.0);
        }
    }
}

void EqRatioChangeObjFunc::func(AbstractSolver &solver, double &obj) {
    
    XVec meas;
    solver.isolveM(meas);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).min(0.0).matrix();
    obj = 0.5 * _res.squaredNorm();
}

void EqRatioChangeObjFunc::funcGrad(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad) {
    
    XVec meas;
    std::vector<XVec > grad;
    solver.isolveMGrad(meas, grad);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).min(0.0).matrix();
    obj = 0.5 * _res.squaredNorm();
        
    XVec _obj_grad = XVec::Zero(Ngrad);
    
    for(int i = 0; i < meas.size(); i++) {
        if(_res(i) > 0.0) {
            _obj_grad += -_res(i) / delta_ratio_target(i) * grad[i];
        }
    }
    
    eigenToVector(_obj_grad, obj_grad);
}

void EqRatioChangeObjFunc::funcHess(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad, 
                      std::vector<std::vector<double> > &obj_hess) {
    
    XVec meas;
    std::vector<XVec > grad;
    std::vector<XMat > hess;
    solver.isolveMHess(meas, grad, hess);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).min(0.0).matrix();
    obj = 0.5 * _res.squaredNorm();
        
    XVec _obj_grad = XVec::Zero(Ngrad);
    XMat _obj_hess = XMat::Zero(Ngrad, Ngrad);
    
    for(int i = 0; i < meas.size(); i++) {
        if(_res(i) > 0.0) {
            XVec _res_grad = -grad[i] / delta_ratio_target(i);
            _obj_grad += _res(i) * _res_grad;
            
            XMat _res_hess = -hess[i] / delta_ratio_target(i);
            
            _obj_hess += _res_grad * _res_grad.transpose() + _res(i) * _res_hess;
        }
    }
    eigenToVector(_obj_grad, obj_grad);
    eigenMatToVector(_obj_hess, obj_hess);
    
}
    
void EqRatioChangeObjFunc::projMeas(std::vector<double> &meas, std::vector<double> &pmeas) {
        

    XVecMap _meas(meas.data(), meas.size());
            
    XVec delta_ratio;
    
    
    if(relative && change) {
        delta_ratio = (_meas - ratio_init).cwiseQuotient(ratio_init);
    } else if(!relative && change) {
        delta_ratio = _meas - ratio_init;
    } else if(relative & !change) {
        delta_ratio = _meas.cwiseQuotient(ratio_init);
    } else if(!relative & !change) {
        delta_ratio = _meas;
    }
    
    XVec penalty = delta_ratio - delta_ratio_target;
    
    XVec _pmeas = (penalty.array().abs() < accuracy).select(0.0, penalty.array());
    
    
//     XVec _pmeas = XVec::Zero(Nterms);
    
//     for(int i = 0; i < Nterms; i++) {
        
//         double penalty = delta_ratio(i) - delta_ratio_target(i);
//         _pmeas[i] = penalty;
//     }
        
    eigenToVector(_pmeas, pmeas);
    
}


void EqRatioChangeObjFunc::objFuncTerms(std::vector<double> &meas, double &obj, std::vector<double> &terms) {
      
    std::vector<double> pmeas;
    projMeas(meas, pmeas);
    
    
    XVec _terms = XVec::Zero(Nterms);
    XVec _pmeas;
    vectorToEigen(pmeas, _pmeas);
    
    _terms = _pmeas.array().pow(2.0).matrix();
    
    obj = 0.5 * _terms.sum();
    
    eigenToVector(_terms, terms);
    
}

void EqRatioChangeObjFunc::objFunc(std::vector<double> &meas, double &obj) {
    
    std::vector<double> terms;
    objFuncTerms(meas, obj, terms);
    
}

void EqRatioChangeObjFunc::projGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<std::vector<double> > &pgrad) {
    
    pgrad.resize(Nterms);
    
    XVecMap _meas(meas.data(), meas.size());
    
    XVec delta_ratio = _meas - ratio_init;
    
    for(int i = 0; i < Nterms; i++) {
        
        XVecMap g_row(meas_grad[i].data(), meas_grad[i].size());
        
        XVec grad = XVec::Zero(Ngrad);
        grad = g_row / delta_ratio_target(i);
        
        eigenToVector(grad, pgrad[i]);
    }
    
}


void EqRatioChangeObjFunc::objFuncGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<double> &obj_grad) {
    
    
//     std::vector<std::vector<double> > pgrad;
//     projGrad(meas, meas_grad, pgrad);
    
    
//     XVec grad = XVec::Zero(Ngrad);
    
//     XVecMap _meas(meas.data(), meas.size());
    
//     XVec delta_ratio = _meas - ratio_init;
    
//     for(int i = 0; i < Nterms; i++) {
        
//         XVecMap pg_row(pgrad[i].data(), pgrad[i].size());
        
//         double penalty = delta_ratio(i) / delta_ratio_target(i) - 1.0;
//         if(penalty < 0.0) {
//             grad += penalty * pg_row;
//         }
//     }
    
//     eigenToVector(grad, obj_grad);
    
    XVec grad = XVec::Zero(Ngrad);
    
    XVecMap _meas(meas.data(), meas.size());
    
    XVec delta_ratio = _meas - ratio_init;
    
    for(int i = 0; i < Nterms; i++) {
        
        XVecMap g_row(meas_grad[i].data(), meas_grad[i].size());
        double penalty = delta_ratio(i) / delta_ratio_target(i) - 1.0;

        grad += penalty * g_row / delta_ratio_target(i);
    }
    
    eigenToVector(grad, obj_grad);
    
}

void EqRatioChangeObjFunc::getConstraints(std::vector<double> &C, std::vector<int> &CT) {
    
    XVec _C = delta_ratio_target + ratio_init;
    eigenToVector(_C, C);
    
    XiVec _CT = XiVec::Zero(Nterms);
    
    eigenToVector(_CT, CT);
    
    
};


////////////////////////////////////////////////////////////////////////////////////////////////////


void IneqRatioObjFunc::setRatioInit(std::vector<double> &ratio_init) {
    // vectorToEigen(ratio_init, this->ratio_init);
    
    this->ratio_init = XVec::Zero(ratio_init.size());
}


void IneqRatioObjFunc::res(AbstractSolver &solver, std::vector<double> &res) {
    XVec meas;
    solver.isolveM(meas);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    // Could use a map to avoid an extra copy
    eigenToVector(_res, res);
}

void IneqRatioObjFunc::resGrad(AbstractSolver &solver, std::vector<double> &res, 
                     std::vector<std::vector<double> > &res_grad) {
    
    XVec meas;
    std::vector<XVec > grad;
    solver.isolveMGrad(meas, grad);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    
    res_grad.resize(Nterms);
    eigenToVector(_res, res);
    
    for(int i = 0; i < meas.size(); i++) {
        if(_res(i) > 0.0) {
            XVec _res_grad = -grad[i] / delta_ratio_target(i);
            eigenToVector(_res_grad, res_grad[i]);
        } else {
            res_grad[i].resize(grad[i].size(), 0.0);
        }
    }
}

void IneqRatioObjFunc::func(AbstractSolver &solver, double &obj) {
    
    XVec meas;
    solver.isolveM(meas);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    obj = 0.5 * _res.squaredNorm();
    
}

void IneqRatioObjFunc::funcGrad(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad) {
    
    XVec meas;
    std::vector<XVec > grad;
    solver.isolveMGrad(meas, grad);
    
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    obj = 0.5 * _res.squaredNorm();
        
    XVec _obj_grad = XVec::Zero(Ngrad);
    
    for(int i = 0; i < meas.size(); i++) {
        if(_res(i) > 0.0) {
            _obj_grad += -_res(i) / delta_ratio_target(i) * grad[i];
        }
    }
    
    eigenToVector(_obj_grad, obj_grad);
}

void IneqRatioObjFunc::funcHess(AbstractSolver &solver, double &obj, std::vector<double> &obj_grad, 
                      std::vector<std::vector<double> > &obj_hess) {
    
        
    XVec meas;
    std::vector<XVec > grad;
    std::vector<XMat > hess;
    solver.isolveMHess(meas, grad, hess);
        
    XVec _res = (1.0 - (meas - ratio_init).array() / delta_ratio_target.array()).max(0.0).matrix();
    obj = 0.5 * _res.squaredNorm();
        
    XVec _obj_grad = XVec::Zero(Ngrad);
    XMat _obj_hess = XMat::Zero(Ngrad, Ngrad);
    
    for(int i = 0; i < meas.size(); i++) {
        if(_res(i) > 0.0) {
            XVec _res_grad = -grad[i] / delta_ratio_target(i);
            _obj_grad += _res(i) * _res_grad;
            
            XMat _res_hess = -hess[i] / delta_ratio_target(i);
            
            _obj_hess += _res_grad * _res_grad.transpose() + _res(i) * _res_hess;
        }
    }
    eigenToVector(_obj_grad, obj_grad);
    eigenMatToVector(_obj_hess, obj_hess);
    
}

    
void IneqRatioObjFunc::projMeas(std::vector<double> &meas, std::vector<double> &pmeas) {

    XVecMap _meas(meas.data(), meas.size());
            
    // XVec delta_ratio = (_meas - ratio_init).cwiseQuotient(_meas.cwiseAbs());
    XVec delta_ratio = _meas - ratio_init;
        
    XVec _pmeas = XVec::Zero(Nterms);
    
    for(int i = 0; i < Nterms; i++) {
        double penalty = delta_ratio(i) / delta_ratio_target(i) - 1.0;
        if(penalty < 0.0) {
            _pmeas[i] = penalty;
        }
    }
        
    eigenToVector(_pmeas, pmeas);
    
}


void IneqRatioObjFunc::objFuncTerms(std::vector<double> &meas, double &obj, std::vector<double> &terms) {
      
    std::vector<double> pmeas;
    projMeas(meas, pmeas);
    
    
    XVec _terms = XVec::Zero(Nterms);
    XVec _pmeas;
    vectorToEigen(pmeas, _pmeas);
    
    _terms = _pmeas.array().pow(2.0).matrix();
    
    obj = 0.5 * _terms.sum();
    
    eigenToVector(_terms, terms);
    
}

void IneqRatioObjFunc::objFunc(std::vector<double> &meas, double &obj) {
    
    std::vector<double> terms;
    objFuncTerms(meas, obj, terms);
    
}

void IneqRatioObjFunc::projGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<std::vector<double> > &pgrad) {
    
    pgrad.resize(Nterms);
    
    XVecMap _meas(meas.data(), meas.size());
    
    XVec delta_ratio = _meas - ratio_init;
    
    for(int i = 0; i < Nterms; i++) {
        
        XVecMap g_row(meas_grad[i].data(), meas_grad[i].size());
        
        XVec grad = XVec::Zero(Ngrad);
        double penalty = delta_ratio(i) / delta_ratio_target(i) - 1.0;
        if(penalty < 0.0) {
            grad = g_row / delta_ratio_target(i);
        }
        
        eigenToVector(grad, pgrad[i]);
    }
    
}


void IneqRatioObjFunc::objFuncGrad(std::vector<double> &meas, std::vector<std::vector<double> > &meas_grad,
                                           std::vector<double> &obj_grad) {
    
    
//     std::vector<std::vector<double> > pgrad;
//     projGrad(meas, meas_grad, pgrad);
    
    
//     XVec grad = XVec::Zero(Ngrad);
    
//     XVecMap _meas(meas.data(), meas.size());
    
//     XVec delta_ratio = _meas - ratio_init;
    
//     for(int i = 0; i < Nterms; i++) {
        
//         XVecMap pg_row(pgrad[i].data(), pgrad[i].size());
        
//         double penalty = delta_ratio(i) / delta_ratio_target(i) - 1.0;
//         if(penalty < 0.0) {
//             grad += penalty * pg_row;
//         }
//     }
    
//     eigenToVector(grad, obj_grad);
    
    XVec grad = XVec::Zero(Ngrad);
    
    XVecMap _meas(meas.data(), meas.size());
    
    XVec delta_ratio = _meas - ratio_init;
    
    for(int i = 0; i < Nterms; i++) {
        
        XVecMap g_row(meas_grad[i].data(), meas_grad[i].size());
        
        double penalty = delta_ratio(i) / delta_ratio_target(i) - 1.0;
        if(penalty < 0.0) {
            grad += penalty * g_row / delta_ratio_target(i);
        }
    }
    
    eigenToVector(grad, obj_grad);
    
}

void IneqRatioObjFunc::getConstraints(std::vector<double> &C, std::vector<int> &CT) {
    
    XVec _C = delta_ratio_target + ratio_init;
    eigenToVector(_C, C);
    
    XiVec _CT = XiVec::Zero(Nterms);
    for(int i = 0; i < Nterms; i++) {
        if(delta_ratio_target(i) > 0.0) {
            _CT(i) = 1;
        } else {
            _CT(i) = -1;
        }
    }
    
    eigenToVector(_CT, CT);
    
    
};


