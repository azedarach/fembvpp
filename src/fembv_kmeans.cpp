#include "fembv_kmeans.hpp"

namespace fembvpp {

FEMBVKMeans::FEMBVKMeans(int n_components_, double max_tv_norm_)
   : n_components(n_components_)
   , max_tv_norm(max_tv_norm_)
{
   if (n_components_ < 1) {
      throw std::runtime_error("number of components must be at least one");
   }
}

} // namespace fembvpp
