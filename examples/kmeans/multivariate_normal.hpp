#ifndef FEMBVPP_MULTIVARIATE_NORMAL_HPP_INCLUDED
#define FEMBVPP_MULTIVARIATE_NORMAL_HPP_INCLUDED

#include <Eigen/Core>

#include <random>

class Multivariate_normal_distribution {
public:
   Multivariate_normal_distribution(
      const Eigen::VectorXd&,
      const Eigen::MatrixXd&);

   template <class Generator>
   Eigen::VectorXd operator()(Generator&) const;

private:
   Eigen::VectorXd mean;
   Eigen::MatrixXd covariance;
   Eigen::MatrixXd transform;
};

template <class Generator>
Eigen::VectorXd Multivariate_normal_distribution::operator()(
   Generator& generator)
{
   std::normal_distribution<> dist(0., 1.);

   const int n_dims = mean.size();
   Eigen::VectorXd z(n_dims);
   for (int i = 0; i < n_dims; ++i) {
      z(i) = dist(generator);
   }

   return mean + transform * z;
}

#endif
