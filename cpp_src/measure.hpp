#ifndef MEASURE
#define MEASURE
    
#include "util.hpp"
    
class Measure {
    public:
    
        int NOstrain;
        XiVec ostrain_nodesi;
        XiVec ostrain_nodesj;
        XVec ostrain_vec;
        XiVec ostrain_bonds;

        int NOstress;
        XiVec ostress_bonds;
    
        
        // Affine deformation response
        bool measure_affine_strain;
        bool measure_affine_stress; 
    
        int NLambda;
        XiVec lambda_vars;

        Measure() {
            NOstrain = 0;
            measure_affine_strain = false;
            NOstress = 0;
            measure_affine_stress = false;
        };
    
        Measure(int NOstrain, std::vector<int> &ostrain_nodesi, std::vector<int> &ostrain_nodesj, 
                std::vector<int> &ostrain_bonds,
                std::vector<double> &ostrain_vec,
                int NOstress, std::vector<int> &ostress_bonds,
                bool measure_affine_strain, bool measure_affine_stress,
               int NLambda, std::vector<int> lambda_vars) {
            
            this->NOstrain = NOstrain;
            this->measure_affine_strain = measure_affine_strain;

            this->NOstress = NOstress;
            this->measure_affine_stress = measure_affine_stress;

            vectorToEigen(ostrain_nodesi, this->ostrain_nodesi);
            vectorToEigen(ostrain_nodesj, this->ostrain_nodesj);
            vectorToEigen(ostrain_bonds, this->ostrain_bonds);
            vectorToEigen(ostrain_vec, this->ostrain_vec);
            
            vectorToEigen(ostress_bonds, this->ostress_bonds);
            
            this->NLambda = NLambda;
            vectorToEigen(lambda_vars, this->lambda_vars);
            
            
        };
    
};
    
#endif
