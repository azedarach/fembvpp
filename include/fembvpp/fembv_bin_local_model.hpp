#ifndef FEMBVPP_FEMBV_BIN_LOCAL_MODEL_HPP_INCLUDED
#define FEMBVPP_FEMBV_BIN_LOCAL_MODEL_HPP_INCLUDED

#include <cmath>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fembvpp {

class FEMBVBin_local_model {
public:
   double epsilon{0};

   FEMBVBin_local_model() = default;
   explicit FEMBVBin_local_model(int);
   explicit FEMBVBin_local_model(const std::vector<int>&);
   FEMBVBin_local_model(const std::vector<int>&,
                        const std::vector<double>&);
   ~FEMBVBin_local_model() = default;
   FEMBVBin_local_model(const FEMBVBin_local_model&) = default;
   FEMBVBin_local_model(FEMBVBin_local_model&&) = default;
   FEMBVBin_local_model& operator=(const FEMBVBin_local_model&) = default;
   FEMBVBin_local_model& operator=(FEMBVBin_local_model&&) = default;

   int get_n_parameters() const { return Lambda.size(); }

   const std::vector<double>& get_parameters() const { return Lambda; }
   void set_parameters(const std::vector<double>&);
   void set_parameter(int, double);

   template <class PredictorsVector>
   double loss(double, const PredictorsVector&) const;
   template <class PredictorsVector>
   double loss_gradient(int, double, const PredictorsVector&) const;
   double regularization() const {
      return epsilon * std::accumulate(std::begin(Lambda), std::end(Lambda), 0.0);
   }
   double regularization_gradient(int /* i */) const {
      return epsilon;
   }

private:
   std::vector<int> predictor_indices{};
   std::vector<double> Lambda{};
};

template <class PredictorsVector>
double FEMBVBin_local_model::loss(double y, const PredictorsVector& X) const
{
   using std::log;

   double lp = 0;
   for (auto j : predictor_indices) {
      lp += Lambda[j] * X(j);
   }
   return y * log(lp) + (1 - y) * log(1 - lp);
}

template <class PredictorsVector>
double FEMBVBin_local_model::loss_gradient(int i, double y, const PredictorsVector& X) const
{
   using std::log;

   const int n_parameters = Lambda.size();
   if (i >= n_parameters) {
      throw std::runtime_error(
         "parameter index out of bounds");
   }

   double lp;
   for (auto j : predictor_indices) {
      lp += Lambda[j] * X(j);
   }

   const int idx = predictor_indices[i];
   return y * X(idx) / lp - (1 - y) * X(idx) / (1 - lp);
}

} // namespace fembvpp

#endif
