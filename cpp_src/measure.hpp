#ifndef MEASURE
#define MEASURE
    
#include "util.hpp"
   
template <int DIM>
class Measure {
    public:
        static const int dim;
    
        bool measure_disp;
        bool measure_strain;
        bool measure_lamb;
    
        int N_ostrain;
        std::vector<int> ostrain_nodesi;
        std::vector<int> ostrain_nodesj;
        XVec ostrain_vec;
        // Whether to interpret strains as extensions
        bool is_extension;

        int N_ostress;
        std::vector<int> ostress_edges;
        // Whether to interpret stresses as tensions (stress = tension * l0)
        bool is_tension;
    
        // Affine deformation response
        bool measure_affine_strain;
        bool measure_affine_stress; 

        Measure() {
            measure_disp = false;
            measure_strain = false;
            measure_lamb = false;
            N_ostrain = 0;
            is_extension = false;
            N_ostress = 0;
            is_tension = false;
            measure_affine_strain = false;
            measure_affine_stress = false;
        };
    
        void setOutputDOF(bool measure_disp, bool measure_strain, bool measure_lamb) {
            this->measure_disp = measure_disp;
            this->measure_strain = measure_strain;
            this->measure_lamb = measure_lamb;
        }
    
        void setOutputStrain(int N_ostrain, std::vector<int> &ostrain_nodesi, std::vector<int> &ostrain_nodesj, RXVec ostrain_vec, bool is_extension) {
            this->N_ostrain = N_ostrain;
            this->ostrain_nodesi = ostrain_nodesi;
            this->ostrain_nodesj = ostrain_nodesj;
            this->ostrain_vec = ostrain_vec;
            this->is_extension = is_extension;
        };
    
        void setOutputStress(int N_ostress, std::vector<int> &ostress_edges, bool is_tension) {
            this->N_ostress = N_ostress;
            this->ostress_edges = ostress_edges;
            this->is_tension = is_tension;
        };
    
        void setOutputAffineStrain(bool measure_affine_strain) {
            this->measure_affine_strain = measure_affine_strain;
        }
    
        void setOutputAffineStress(bool measure_affine_stress) {
            this->measure_affine_stress = measure_affine_stress;
        }   
    
};

template <int DIM>
const int Measure<DIM>::dim = DIM;
    
#endif // MEASURE
