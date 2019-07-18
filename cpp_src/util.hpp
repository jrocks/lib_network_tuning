#ifndef UTIL
#define UTIL

#define BILLION 1000000000L
#define PI 3.14159265358979323846
#define INF 1e10    
#define MEPS 2.22045e-16
#define SQRTMEPS 1.49011745e-8

#include <vector>
#include <iostream>
#include <numeric>
#include <unordered_map>
    
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
// #include <Eigen/CholmodSupport>
#include <Eigen/UmfPackSupport>
    
//Eigen typedefs
    
// Dimension dependent types
#define DEIGEN(DIM) \
typedef Eigen::Matrix<double, DIM, 1> DVec; \
typedef Eigen::Matrix<double, DIM, DIM> DMat;
    
// Dimension dependent compressed symmetric types
#define DSYMEIGEN(DIM) \
typedef Eigen::Matrix<double, DIM*(DIM+1)/2, 1> SDVec; \
typedef Eigen::Matrix<double, DIM*(DIM+1)/2, DIM*(DIM+1)/2> SDMat;

    
// Variable length types
typedef Eigen::VectorXd XVec;
typedef Eigen::MatrixXd XMat;

// Sparse types
typedef Eigen::Triplet<double> Trip;
typedef Eigen::SparseMatrix<double> SMat;
typedef Eigen::SparseVector<double> SVec;

// Reference types
typedef const Eigen::Ref<const XVec > RXVec;
typedef const Eigen::Ref<const XMat > RXMat;

typedef Eigen::Ref<SVec > RSVec;
typedef Eigen::Ref<SMat > RSMat;


// Map types
typedef Eigen::Map<XVec > XVecMap;
typedef Eigen::Map<XMat > XMatMap;

typedef Eigen::Map<const XVec > XVecConstMap;
typedef Eigen::Map<const XMat > XMatConstMap;






// Eigen typedefs
// typedef Eigen::Matrix<double, DIM, 1> DVec;
// typedef Eigen::Matrix<double, DIM, DIM> DMat;
// typedef Eigen::Matrix<int, DIM, DIM> DiMat;

// typedef Eigen::Matrix<double, DIM*DIM, 1> D2Vec;
// typedef Eigen::Matrix<double, DIM*DIM, DIM*DIM> D2Mat;



// typedef Eigen::Matrix<double, 3, 1> Vec3d;



// typedef Eigen::Map<XVec > MXVec;
// typedef Eigen::Map<XMat > MXMat;
// typedef Eigen::Map<DMat > MDMat;




#endif // UTIL
