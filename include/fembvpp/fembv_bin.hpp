#ifndef FEMBVPP_FEMBV_BIN_HPP_INCLUDED
#define FEMBVPP_FEMBV_BIN_HPP_INCLUDED

/**
 * @file fembv_bin.hpp
 * @brief contains definition of FEM-BV-BIN class
 */

#include "clpsimplex_affiliations_solver.hpp"
#include "fembv_bin_local_model.hpp"
#include "fembv_bin_local_model_ipopt_solver.hpp"
#include "random_matrix.hpp"

#include <Eigen/Core>

#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

namespace fembvpp {

namespace detail {

bool check_fembv_bin_convergence(double, double, double);
double fembv_bin_local_model_constraint(
   const std::vector<double>&, std::vector<double>&, void*);

template <class AffiliationsMatrix, class Generator>
void fembv_bin_random_initialization(
   AffiliationsMatrix& Gamma, std::vector<FEMBVBin_local_model>& models,
   Generator& generator)
{
   const std::size_t n_components = models.size();
   std::uniform_real_distribution<> dist(0., 1.);

   for (std::size_t i = 0; i < n_components; ++i) {
      const int n_parameters = models[i].get_n_parameters();
      std::vector<double> Lambda(n_parameters);
      double sum = 0;
      for (int j = 0; j < n_parameters; ++j) {
         Lambda[j] = dist(generator);
         sum += Lambda[j];
      }

      for (int j = 0; j < n_parameters; ++j) {
         Lambda[j] /= (2 * sum);
      }

      models[i].set_parameters(Lambda);
   }

   random_left_stochastic_matrix(Gamma, generator);
}

template <class AffiliationsMatrix, class DistanceMatrix>
double fembv_bin_cost(
   const AffiliationsMatrix& Gamma, const std::vector<FEMBVBin_local_model>& models,
   const DistanceMatrix& G)
{
   double cost = (Gamma.transpose() * G).trace();
   for (const auto& m: models) {
      cost += m.regularization();
   }
   return cost;
}

template <class OutcomesVector, class PredictorsMatrix, class DistanceMatrix>
void fill_fembv_bin_distance_matrix(
   const OutcomesVector& Y, const PredictorsMatrix& X,
   const std::vector<FEMBVBin_local_model>& models, DistanceMatrix& G)
{
   const int n_samples = X.cols();
   const int n_components = models.size();

   for (int t = 0; t < n_samples; ++t) {
      for (int i = 0; i < n_components; ++i) {
         G(i, t) = -models[i].loss(Y(t), X.col(t));
      }
   }
}

template <class OutcomesVector, class PredictorsMatrix, class AffiliationsMatrix,
          class ParametersSolver>
bool update_fembv_bin_parameters(
   const OutcomesVector& Y, const PredictorsMatrix& X,
   const AffiliationsMatrix& Gamma,
   std::vector<FEMBVBin_local_model>& models,
   ParametersSolver& solver)
{
   const int n_components = Gamma.rows();

   bool success = true;
   for (int i = 0; i < n_components; ++i) {
      success = success && solver.update_local_model(
         Y, X, Gamma.row(i), models[i]);
   }

   return success;
}

template <class OutcomesVector, class PredictorsMatrix, class AffiliationsMatrix,
          class AffiliationsSolver, class ParametersSolver, class DistanceMatrix>
std::tuple<bool, bool, std::size_t, double>
fembv_bin_subspace(
   const OutcomesVector& Y, const PredictorsMatrix& X, AffiliationsMatrix& Gamma,
   std::vector<FEMBVBin_local_model>& models, AffiliationsSolver& gamma_solver,
   ParametersSolver& theta_solver, DistanceMatrix& G, std::size_t max_iterations,
   double tolerance, bool update_parameters, int verbosity)
{
   auto initial_cost = fembv_bin_cost(Gamma, models, G);

   auto old_cost = initial_cost;
   bool parameters_success = true;
   bool affiliations_success = true;
   std::size_t n_iter = 0;
   bool converged = false;
   while (n_iter < max_iterations) {
      if (update_parameters) {
         parameters_success = update_fembv_bin_parameters(
            Y, X, Gamma, models, theta_solver);
      }

      fill_fembv_bin_distance_matrix(Y, X, models, G);

      const auto gamma_status = gamma_solver.update_affiliations(G);
      if (static_cast<int>(gamma_status) == 0) {
         affiliations_success = true;
      } else {
         affiliations_success = false;
      }
      
      gamma_solver.get_affiliations(Gamma);

      const auto new_cost = fembv_bin_cost(Gamma, models, G);

      if (verbosity > 0) {
         std::cout << "Iteration " << n_iter + 1 << '\n';
         std::cout << "Current cost: " << old_cost << '\n';
         std::cout << "Updated cost: " << new_cost << '\n';
         std::cout << "Cost increment: " << new_cost - old_cost << '\n';
         std::cout << "Cost ratio: " << new_cost / initial_cost << '\n';
      }

      converged = check_fembv_bin_convergence(old_cost, new_cost, tolerance);

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

struct FEMBVBin_parameters {
   double max_tv_norm{-1};
   std::size_t max_iterations{1000};
   int max_affiliations_iterations{10000};
   double tolerance{1e-8};
   double parameters_tolerance{1e-6};
   int max_parameters_iterations{10000};
   Ipopt_initial_guess parameters_initialization{Ipopt_initial_guess::Uniform};
   bool update_parameters{true};
   int verbosity{0};
   int random_seed{0};
};

template <class OutcomesVector, class PredictorsMatrix,
          class AffiliationsMatrix, class DistanceMatrix,
          class BasisMatrix>
std::tuple<bool, std::size_t, double>
fembv_bin(
   const OutcomesVector& Y, const PredictorsMatrix& X, AffiliationsMatrix& Gamma,
   std::vector<FEMBVBin_local_model>& models, DistanceMatrix& G,
   const BasisMatrix& V, const FEMBVBin_parameters& parameters)
{
   const int n_samples = X.cols();
   const int n_components = models.size();

   if (Gamma.cols() != n_samples) {
      throw std::runtime_error(
         "number of affiliation samples does not match "
         "number of data samples");
   }

   if (Gamma.rows() != n_components) {
      throw std::runtime_error(
         "number of affiliation series does not match "
         "number of components");
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

   detail::fill_fembv_bin_distance_matrix(Y, X, models, G);

   FEMBVBin_local_model_ipopt_solver theta_solver(parameters.random_seed);
   theta_solver.set_tolerance(parameters.parameters_tolerance);
   theta_solver.set_max_iterations(parameters.max_parameters_iterations);
   theta_solver.set_verbosity(parameters.verbosity);
   theta_solver.set_initialization_method(parameters.parameters_initialization);
   theta_solver.initialize();

   ClpSimplex_affiliations_solver gamma_solver(G, V, parameters.max_tv_norm);
   gamma_solver.set_max_iterations(parameters.max_affiliations_iterations);
   gamma_solver.set_verbosity(parameters.verbosity);

   const auto result = detail::fembv_bin_subspace(
      Y, X, Gamma, models, gamma_solver, theta_solver, G,
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

class FEMBVBin {
public:

   FEMBVBin(int, double);
   ~FEMBVBin() = default;

   template <class OutcomesVector, class PredictorsMatrix, class Generator>
   bool fit(const OutcomesVector&, const PredictorsMatrix&, Generator&);
   template <class OutcomesVector, class PredictorsMatrix>
   Eigen::MatrixXd transform(const OutcomesVector&, const PredictorsMatrix&) const;

   void set_regularization(double e) { epsilon = e; }

   void set_max_iterations(std::size_t i) { max_iterations = i; }
   void set_tolerance(double t) { tolerance = t; }
   void set_verbosity(int v) { verbosity = v; }
   void set_random_seed(int s) { random_seed = s; }
   void set_parameters_initialization(Ipopt_initial_guess i) {
      parameters_initialization = i;
   }

   double get_cost() const { return cost; }
   double get_log_likelihood_bound() const { return log_likelihood_bound; }
   std::size_t get_total_n_parameters() const { return n_parameters; }
   std::size_t get_n_iter() const { return n_iter; }
   const std::vector<FEMBVBin_local_model>& get_parameters() const { return models; }
   const Eigen::MatrixXd& get_affiliations() const { return affiliations; }

private:
   int n_components{2};
   double max_tv_norm{-1};
   double epsilon{0};
   std::size_t max_iterations{1000};
   std::size_t max_affiliations_iterations{10000};
   double tolerance{1e-6};
   double parameters_tolerance{1e-6};
   std::size_t max_parameters_iterations{10000};
   int verbosity{0};
   int random_seed{0};
   Ipopt_initial_guess parameters_initialization{Ipopt_initial_guess::Uniform};
   double cost{-1};
   double log_likelihood_bound{-1};
   std::size_t n_parameters{0};
   std::size_t n_iter{0};
   std::vector<FEMBVBin_local_model> models{};
   Eigen::MatrixXd affiliations{};

   template <class OutcomesVector, class PredictorsMatrix>
   double calculate_log_likelihood_bound(
      const OutcomesVector&, const PredictorsMatrix&) const;
};

template <class OutcomesVector, class PredictorsMatrix, class Generator>
bool FEMBVBin::fit(const OutcomesVector& Y, const PredictorsMatrix& X, Generator& generator)
{
   const int n_samples = X.cols();
   const int n_features = X.rows();

   FEMBVBin_parameters fembv_bin_parameters;
   fembv_bin_parameters.max_tv_norm = max_tv_norm;
   fembv_bin_parameters.max_iterations = max_iterations;
   fembv_bin_parameters.max_affiliations_iterations = max_affiliations_iterations;
   fembv_bin_parameters.parameters_tolerance = parameters_tolerance;
   fembv_bin_parameters.max_parameters_iterations = max_parameters_iterations;
   fembv_bin_parameters.parameters_initialization = parameters_initialization;
   fembv_bin_parameters.tolerance = tolerance;
   fembv_bin_parameters.update_parameters = true;
   fembv_bin_parameters.verbosity = verbosity;

   // restrict to triangular basis functions
   Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_samples, n_samples));
   Eigen::MatrixXd G(n_components, n_samples);

   affiliations = Eigen::MatrixXd(n_components, n_samples);
   models = std::vector<FEMBVBin_local_model>(n_components);
   for (int i = 0; i < n_components; ++i) {
      models[i] = FEMBVBin_local_model(n_features);
      models[i].epsilon = epsilon;
   }
   detail::fembv_bin_random_initialization(affiliations, models, generator);

   const auto result = fembv_bin(
      Y, X, affiliations, models, G, V, fembv_bin_parameters);

   n_iter = std::get<1>(result);
   // note: count does not take into account equality constraints
   n_parameters = n_components * n_samples;
   for (int i = 0; i < n_components; ++i) {
      n_parameters += models[i].get_n_parameters();
   }
   cost = std::get<2>(result);

   log_likelihood_bound = calculate_log_likelihood_bound(Y, X);

   return std::get<0>(result);
}

template <class OutcomesVector, class PredictorsMatrix>
Eigen::MatrixXd FEMBVBin::transform(const OutcomesVector& Y, const PredictorsMatrix& X) const
{
   const int n_samples = X.cols();

   FEMBVBin_parameters fembv_bin_parameters;
   fembv_bin_parameters.max_tv_norm = max_tv_norm;
   fembv_bin_parameters.max_iterations = max_iterations;
   fembv_bin_parameters.tolerance = tolerance;
   fembv_bin_parameters.parameters_tolerance = parameters_tolerance;
   fembv_bin_parameters.max_parameters_iterations = max_parameters_iterations;
   fembv_bin_parameters.parameters_initialization = parameters_initialization;
   fembv_bin_parameters.update_parameters = false;
   fembv_bin_parameters.verbosity = verbosity;

   Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_samples, n_samples));
   Eigen::MatrixXd G(n_components, n_samples);
   Eigen::MatrixXd local_affiliations(affiliations);

   const auto result = fembv_bin(
      Y, X, local_affiliations, models, G, V, fembv_bin_parameters);

   const bool success = std::get<0>(result);
   if (!success) {
      throw std::runtime_error(
         "calculating representation of data failed");
   }

   return local_affiliations;
}

template <class OutcomesVector, class PredictorsMatrix>
double FEMBVBin::calculate_log_likelihood_bound(
   const OutcomesVector& Y, const PredictorsMatrix& X) const
{
   using std::log;

   const int n_samples = Y.size();

   double bound = 0;
   for (int t = 0; t < n_samples; ++t) {
      for (int i = 0; i < n_components; ++i) {
         bound += affiliations(i, t) * models[i].loss(Y(t), X.col(t));
      }
   }

   return bound;
}

} // namespace fembvpp

#endif
