#ifndef FEMBVPP_FEMBV_KMEANS_HPP_INCLUDED
#define FEMBVPP_FEMBV_KMEANS_HPP_INCLUDED

/**
 * @file fembv_kmeans.hpp
 * @brief contains definition of FEM-BV-k-means class
 */

#include "random_matrix.hpp"

#include <iostream>
#include <vector>

namespace fembvpp {

namespace detail {

template <class DataMatrix, class AffiliationsMatrix, class ParametersMatrix,
          class Generator>
void random_initialization(
   const DataMatrix& X, AffiliationsMatrix& Gamma, ParametersMatrix& Theta,
   Generator& generator)
{
   const int n_samples = X.cols();
   const int n_components = Theta.cols();

   std::uniform_int_distribution<> dist(0, n_samples - 1);
   std::vector<int> selected_samples;
   int pos = 0;
   while (pos < n_components) {
      const int idx = dist(generator);
      if (std::find(
             std::begin(selected_samples),
             std::end(selected_samples), idx) == std::end(selected_samples)) {
         selected_samples.push_back(idx);
         Theta.col(pos) = X.col(idx);
         pos++;
      }
   }

   random_left_stochastic_matrix(Gamma, generator);
}

template <class AffiliationsMatrix, class DistanceMatrix>
double fembv_kmeans_cost(
   const AffiliationsMatrix& Gamma, const DistanceMatrix& G)
{
   return (Gamma.transpose() * G).trace();
}

template <class DataMatrix, class ParametersMatrix, class DistanceMatrix>
void fill_distance_matrix(const DataMatrix& X, const ParametersMatrix& Theta,
                          DistanceMatrix& G)
{
   const int n_samples = X.cols();
   const int n_components = Theta.cols();

   for (int j = 0; j < n_samples; ++j) {
      for (int i = 0; i < n_components; ++i) {
         const auto dist = (X.col(i) - Theta.col(j)).norm();
         G(i, j) = dist * dist;
      }
   }
}

template <class DataMatrix, class AffiliationsMatrix, class ParametersMatrix>
bool update_kmeans_parameters(
   const DataMatrix& X, const AffiliationsMatrix& Gamma,
   ParametersMatrix& Theta)
{
   const int n_features = X.rows();
   const int n_samples = Gamma.cols();
   const int n_components = Gamma.rows();

   std::vector<double> normalizations(n_components, 0);
   for (int j = 0; j < n_samples; ++j) {
      for (int i = 0; i < n_components; ++i) {
         normalizations[i] += Gamma(i, j);
      }
   }

   Theta = Gamma.transpose() * X;
   for (int j = 0; j < n_components; ++j) {
      for (int i = 0; i < n_features; ++i) {
         Theta(i, j) /= normalizations[j];
      }
   }

   return true;
}

bool check_convergence(double old_cost, double new_cost, double tolerance)
{
   using std::abs;

   const double cost_delta = abs(old_cost - new_cost);

   const double min_cost = abs(old_cost) > abs(new_cost) ? new_cost : old_cost;
   const double max_cost = abs(old_cost) > abs(new_cost) ? old_cost : new_cost;

   const double rel_cost = 1 - abs(min_cost / max_cost);

   return cost_delta < tolerance || rel_cost < tolerance;
}

template <class DataMatrix, class AffiliationsMatrix, class ParametersMatrix,
          class AffiliationsSolver, class DistanceMatrix>
std::tuple<bool, bool, std::size_t, double>
fembv_kmeans_subspace(
   const DataMatrix& X, AffiliationsMatrix& Gamma, ParametersMatrix& Theta,
   AffiliationsSolver& gamma_solver, DistanceMatrix& G,
   std::size_t max_iterations, double tolerance,
   bool update_parameters, int verbosity)
{
   auto initial_cost = kmeans_cost(Gamma, G);

   auto old_cost = initial_cost;
   bool parameters_success = true;
   bool affiliations_success = true;
   for (std::size_t n_iter = 0; n_iter < max_iterations; ++n_iter) {
      if (update_parameters) {
         parameters_success = update_parameters(X, Gamma, Theta);
      }

      fill_distance_matrix(X, Theta, G);

      const auto gamma_status = gamma_solver.update_affiliations(G);
      solver.get_affiliations(Gamma);

      const auto new_cost = kmeans_cost(Gamma, G);

      if (verbosity > 0) {
         std::cout << "Iteration " << n_iter << '\n';
         std::cout << "Current cost: " << old_cost << '\n';
         std::cout << "Updated cost: " << new_cost << '\n';
         std::cout << "Cost increment: " << new_cost - old_cost << '\n';
         std::cout << "Cost ratio: " << new_cost / initial_cost << '\n';
      }

      const auto converged = check_convergence(old_cost, new_cost, tolerance);

      old_cost = new_cost;

      if (converged) {
         if (verbosity > 0) {
            std::cout << "Converged at iteration " << n_iter + 1 << '\n';
         }
         break;
      }
   }

   if (verbosity > 0 && n_iter == max_iterations && !converged) {
      std::cout << "Warning: maximum iterations reached\n";
   }

   return std::make_tuple(parameters_success, affiliations_success,
                          n_iter, old_cost);
}

} // namespace detail

template <class DataMatrix>
std::tuple<bool, std::size_t, double>
fembv_kmeans(const DataMatrix& X, AffiliationsMatrix& Gamma, )
{

}

} // namespace fembvpp

#endif
