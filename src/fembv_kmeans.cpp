#include "fembv_kmeans.hpp"

#include <cmath>

namespace fembvpp {

namespace detail {

bool check_convergence(double old_cost, double new_cost, double tolerance)
{
   using std::abs;

   const double cost_delta = abs(old_cost - new_cost);

   const double min_cost = abs(old_cost) > abs(new_cost) ? new_cost : old_cost;
   const double max_cost = abs(old_cost) > abs(new_cost) ? old_cost : new_cost;

   const double rel_cost = 1 - abs(min_cost / max_cost);

   return cost_delta < tolerance || rel_cost < tolerance;
}

} // namespace detail

FEMBVKMeans::FEMBVKMeans(int n_components_, double max_tv_norm_)
   : n_components(n_components_)
   , max_tv_norm(max_tv_norm_)
{
   if (n_components_ < 1) {
      throw std::runtime_error("number of components must be at least one");
   }
}

} // namespace fembvpp
