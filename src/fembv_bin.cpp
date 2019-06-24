#include "fembv_bin.hpp"

#include <cmath>
#include <iterator>
#include <numeric>

namespace fembvpp {

namespace detail {

bool check_fembv_bin_convergence(double old_cost, double new_cost, double tolerance)
{
   using std::abs;

   const double cost_delta = abs(old_cost - new_cost);

   const double min_cost = abs(old_cost) > abs(new_cost) ? new_cost : old_cost;
   const double max_cost = abs(old_cost) > abs(new_cost) ? old_cost : new_cost;

   const double rel_cost = 1 - abs(min_cost / max_cost);

   return cost_delta < tolerance || rel_cost < tolerance;
}

double fembv_bin_regularization(const std::vector<double>& Lambda)
{
   return std::accumulate(std::begin(Lambda), std::end(Lambda), 0);
}

double fembv_bin_local_constraint(
   const std::vector<double>& x, std::vector<double>& dx, void* /* params */)
{
   if (!dx.empty()) {
      for (auto& dxi : dx) {
         dxi = 1;
      }
   }
   return std::accumulate(std::begin(x), std::end(x), 0.0) - 1.0;
}

} // namespace detail

} // namespace fembvpp
