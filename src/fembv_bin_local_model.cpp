#include "fembv_bin_local_model.hpp"

namespace fembvpp {

FEMBVBin_local_model::FEMBVBin_local_model(int n_features)
{
   if (n_features < 1) {
      throw std::runtime_error(
         "number of features must be at least one");
   }

   Lambda = std::vector<double>(n_features, 0);
   predictor_indices = std::vector<int>(n_features);
   std::iota(std::begin(predictor_indices), std::end(predictor_indices), 0);
}

FEMBVBin_local_model::FEMBVBin_local_model(
   const std::vector<int>& predictor_indices_)
   : predictor_indices(predictor_indices_)
{
   const std::size_t n_parameters = predictor_indices_.size();
   Lambda = std::vector<double>(n_parameters, 0);
}

FEMBVBin_local_model::FEMBVBin_local_model(
   const std::vector<int>& predictor_indices_,
   const std::vector<double>& Lambda_)
   : predictor_indices(predictor_indices_)
   , Lambda(Lambda_)
{
   if (predictor_indices_.size() != Lambda_.size()) {
      throw std::runtime_error(
         "number of predictor indices does not match "
         "number of parameters");
   }
}

void FEMBVBin_local_model::set_parameters(const std::vector<double>& Lambda_)
{
   if (Lambda_.size() != Lambda.size()) {
      throw std::runtime_error(
         "number of new parameters does not match number "
         "of old parameters");
   }
   Lambda = Lambda_;
}

void FEMBVBin_local_model::set_parameter(int i, double l)
{
   const int n_parameters = Lambda.size();
   if (i < 0 || i >= n_parameters) {
      throw std::runtime_error(
         "parameter index out of bounds");
   }
   Lambda[i] = l;
}

} // namespace fembvpp
