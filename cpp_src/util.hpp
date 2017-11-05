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
typedef Eigen::Ref<XVec > RXVec;
typedef Eigen::Ref<XMat > RXMat;

typedef Eigen::Ref<SVec > RSVec;
typedef Eigen::Ref<SMat > RSMat;









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





// inline void vectorToEigen(std::vector<double> &v, DVec &e) {
//     XVecMap map(v.data(), DIM);
//     e = map;
// }

// inline void vectorToEigen(std::vector<double> &v, XVec &e) {
//     XVecMap map(v.data(), v.size());
//     e = map;
// }

// inline void vectorToEigen(std::vector<int> &v, XiVec &e) {
//     XiVecMap map(v.data(), v.size());
//     e = map;
// }

// inline void eigenToVector(XVec &e, std::vector<double> &v) {
//     v.reserve(e.size());
//     v.assign(e.data(), e.data()+e.size());
// }

// inline void eigenToVector(XiVec &e, std::vector<int> &v) {
//     v.reserve(e.size());
//     v.assign(e.data(), e.data()+e.size());
// }

// inline void eigenMatToVector(XMat &e, std::vector<std::vector<double> > &v) {
        
//     v.resize(e.rows());
//     for(int i = 0; i < e.rows(); i++) {
//         XVec tmp = e.row(i);
//         eigenToVector(tmp, v[i]);
//     }  
// }

// inline void insert_sparse_block(SMat &block, int row_offset, int col_offset, std::vector<int> &rows, 
//                                 std::vector<int> &cols, std::vector<double> &vals) {
//     for (int k = 0; k < block.outerSize(); k++) {
//         for (SMat::InnerIterator it(block,k); it; ++it) {
//             rows.push_back(row_offset+it.row());
//             cols.push_back(col_offset+it.col());
//             vals.push_back(it.value());
//         }
//     } 
// }

#endif // UTIL
