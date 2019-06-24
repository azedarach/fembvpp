#include "multivariate_normal.hpp"

#include <Eigen/Eigenvalues>

#include <stdexcept>

namespace fembvpp {

Multivariate_normal_distribution::Multivariate_normal_distribution(
   const Eigen::VectorXd& mean_,
   const Eigen::MatrixXd& covariance_)
   : mean(mean_)
   , covariance(covariance_)
{
   int n_dims = mean_.size();
   if (covariance.rows() != n_dims || covariance.cols() != n_dims) {
      throw std::runtime_error(
         "invalid size of covariance matrix");
   }

   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      covariance_);
   Eigen::VectorXd evals(solver.eigenvalues());

   if ((evals.array() < 0).any()) {
      throw std::runtime_error(
         "covariance matrix is not positive semi-definite");
   }

   transform = solver.eigenvectors() * evals.cwiseSqrt().asDiagonal();
}

} // namespace fembvpp
