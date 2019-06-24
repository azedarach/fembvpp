#ifndef FEMBVPP_FEMBV_KMEANS_HPP_INCLUDED
#define FEMBVPP_FEMBV_KMEANS_HPP_INCLUDED

/**
 * @file fembv_kmeans.hpp
 * @brief contains definition of FEM-BV-k-means class
 */

#include "clpsimplex_affiliations_solver.hpp"
#include "random_matrix.hpp"

#include <Eigen/Core>

#include <iostream>
#include <vector>

namespace fembvpp {

namespace detail {

bool check_convergence(double, double, double);

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

   for (int t = 0; t < n_samples; ++t) {
      for (int i = 0; i < n_components; ++i) {
         const auto dist = (X.col(t) - Theta.col(i)).norm();
         G(i, t) = dist * dist;
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

   Theta = X * Gamma.transpose();
   for (int j = 0; j < n_components; ++j) {
      for (int i = 0; i < n_features; ++i) {
         Theta(i, j) /= normalizations[j];
      }
   }

   return true;
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
   auto initial_cost = fembv_kmeans_cost(Gamma, G);

   auto old_cost = initial_cost;
   bool parameters_success = true;
   bool affiliations_success = true;
   std::size_t n_iter = 0;
   bool converged = false;
   while (n_iter < max_iterations) {
      if (update_parameters) {
         parameters_success = update_kmeans_parameters(X, Gamma, Theta);
      }

      fill_distance_matrix(X, Theta, G);

      const auto gamma_status = gamma_solver.update_affiliations(G);
      if (static_cast<int>(gamma_status) == 0) {
         affiliations_success = true;
      } else {
         affiliations_success = false;
      }
      gamma_solver.get_affiliations(Gamma);

      const auto new_cost = fembv_kmeans_cost(Gamma, G);

      if (verbosity > 0) {
         std::cout << "Iteration " << n_iter << '\n';
         std::cout << "Current cost: " << old_cost << '\n';
         std::cout << "Updated cost: " << new_cost << '\n';
         std::cout << "Cost increment: " << new_cost - old_cost << '\n';
         std::cout << "Cost ratio: " << new_cost / initial_cost << '\n';
      }

      converged = check_convergence(old_cost, new_cost, tolerance);

      old_cost = new_cost;

      if (converged) {
         if (verbosity > 0) {
            std::cout << "Converged at iteration " << n_iter + 1 << '\n';
         }
         break;
      }

      n_iter++;
   }

   if (verbosity > 0 && n_iter == max_iterations && !converged) {
      std::cout << "Warning: maximum iterations reached\n";
   }

   return std::make_tuple(parameters_success, affiliations_success,
                          n_iter, old_cost);
}

} // namespace detail

struct FEMBVKMeans_parameters {
   double max_tv_norm{-1};
   std::size_t max_iterations{1000};
   double tolerance{1e-8};
   bool update_parameters{true};
   int verbosity{0};
};

template <class DataMatrix, class AffiliationsMatrix, class ParametersMatrix,
          class DistanceMatrix, class BasisMatrix>
std::tuple<bool, std::size_t, double>
fembv_kmeans(
   const DataMatrix& X, AffiliationsMatrix& Gamma, ParametersMatrix& Theta,
   DistanceMatrix& G, const BasisMatrix& V, const FEMBVKMeans_parameters& parameters)
{
   const int n_features = X.rows();
   const int n_samples = X.cols();
   const int n_components = Gamma.rows();

   if (Gamma.cols() != n_samples) {
      throw std::runtime_error(
         "number of affiliation samples does not match "
         "number of data samples");
   }

   if (Theta.rows() != n_features) {
      throw std::runtime_error(
         "number of parameter dimensions does not match "
         "data dimension");
   }

   if (G.rows() != n_components) {
      throw std::runtime_error(
         "number of distance matrix rows does not match "
         "number of components");
   }
   if (G.cols() != n_samples) {
      throw std::runtime_error(
         "number of distance matrix columns does not match "
         "number of samples");
   }

   if (V.cols() != n_samples) {
      throw std::runtime_error(
         "number of basis element columns does not match "
         "number of samples");
   }

   detail::fill_distance_matrix(X, Theta, G);

   ClpSimplex_affiliations_solver gamma_solver(G, V, parameters.max_tv_norm);
   gamma_solver.set_max_iterations(parameters.max_iterations);
   gamma_solver.set_verbosity(parameters.verbosity);

   const auto result = detail::fembv_kmeans_subspace(
      X, Gamma, Theta, gamma_solver, G,
      parameters.max_iterations, parameters.tolerance,
      parameters.update_parameters, parameters.verbosity);

   const bool parameters_success = std::get<0>(result);
   const bool affiliations_success = std::get<1>(result);
   const std::size_t n_iter = std::get<2>(result);
   const double cost = std::get<3>(result);

   if (!parameters_success && parameters.verbosity > 0) {
      std::cout << "Warning: failed to find optimal parameters\n";
   }
   if (!affiliations_success && parameters.verbosity > 0) {
      std::cout << "Warning: failed to find optimal affiliations\n";
   }

   if (n_iter == parameters.max_iterations && parameters.verbosity > 0) {
      std::cout << "Warning: maximum number of iterations reached\n";
   }

   const bool success = parameters_success && affiliations_success;

   return std::make_tuple(success, n_iter, cost);
}

class FEMBVKMeans {
public:

   FEMBVKMeans(int, double);
   ~FEMBVKMeans() = default;

   template <class DataMatrix, class Generator>
   bool fit(const DataMatrix&, Generator&);
   template <class DataMatrix>
   Eigen::MatrixXd transform(const DataMatrix&) const;

   void set_max_iterations(std::size_t i) { max_iterations = i; }
   void set_tolerance(double t) { tolerance = t; }
   void set_verbosity(int v) { verbosity = v; }

   double get_cost() const { return cost; }
   std::size_t get_n_iter() const { return n_iter; }
   const Eigen::MatrixXd& get_parameters() const { return parameters; }
   const Eigen::MatrixXd& get_affiliations() const { return affiliations; }

private:
   int n_components{2};
   double max_tv_norm{-1};
   std::size_t max_iterations{1000};
   double tolerance{1e-6};
   int verbosity{0};
   double cost{-1};
   std::size_t n_iter{0};
   Eigen::MatrixXd parameters{};
   Eigen::MatrixXd affiliations{};
};

template <class DataMatrix, class Generator>
bool FEMBVKMeans::fit(const DataMatrix& X, Generator& generator)
{
   const int n_samples = X.cols();
   const int n_features = X.rows();

   FEMBVKMeans_parameters kmeans_parameters;
   kmeans_parameters.max_tv_norm = max_tv_norm;
   kmeans_parameters.max_iterations = max_iterations;
   kmeans_parameters.tolerance = tolerance;
   kmeans_parameters.update_parameters = true;
   kmeans_parameters.verbosity = verbosity;

   // restrict to triangular basis functions
   Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_samples, n_samples));
   Eigen::MatrixXd G(n_components, n_samples);

   affiliations = Eigen::MatrixXd(n_components, n_samples);
   parameters = Eigen::MatrixXd(n_features, n_components);
   detail::random_initialization(X, affiliations, parameters, generator);

   const auto result = fembv_kmeans(
      X, affiliations, parameters, G, V, 
      kmeans_parameters);

   n_iter = std::get<1>(result);
   cost = std::get<2>(result);

   return std::get<0>(result);
}

template <class DataMatrix>
Eigen::MatrixXd FEMBVKMeans::transform(const DataMatrix& X) const
{
   const int n_samples = X.cols();

   FEMBVKMeans_parameters kmeans_parameters;
   kmeans_parameters.max_tv_norm = max_tv_norm;
   kmeans_parameters.max_iterations = max_iterations;
   kmeans_parameters.tolerance = tolerance;
   kmeans_parameters.update_parameters = false;
   kmeans_parameters.verbosity = verbosity;

   Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_samples, n_samples));
   Eigen::MatrixXd G(n_components, n_samples);
   Eigen::MatrixXd local_affiliations(affiliations);

   const auto result = fembv_kmeans(
      X, local_affiliations, parameters, G, V,
      kmeans_parameters);

   const bool success = std::get<0>(result);
   if (!success) {
      throw std::runtime_error(
         "calculating representation of data failed");
   }

   return local_affiliations;
}

} // namespace fembvpp

#endif
