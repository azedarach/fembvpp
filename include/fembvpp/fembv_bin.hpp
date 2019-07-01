#ifndef FEMBVPP_FEMBV_BIN_HPP_INCLUDED
#define FEMBVPP_FEMBV_BIN_HPP_INCLUDED

/**
 * @file fembv_bin.hpp
 * @brief contains definition of FEM-BV-BIN class
 */

#include "clpsimplex_affiliations_solver.hpp"
#include "fembv_bin_local_model.hpp"
#include "random_matrix.hpp"

#include <Eigen/Core>
#include <nlopt.hpp>

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

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
double fembv_bin_local_likelihood(
   const OutcomesVector& Y, const PredictorsMatrix& X, const WeightsVector& weights,
   const FEMBVBin_local_model& model)
{
   using std::log;

   const int n_samples = Y.size();

   double loss = 0;
   for (int t = 0; t < n_samples; ++t) {
      loss += weights(t) * model.loss(Y(t), X.col(t));
   }

   return loss - model.regularization();
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
std::vector<double> fembv_bin_local_likelihood_grad(
   const OutcomesVector& Y, const PredictorsMatrix& X, const WeightsVector& weights,
   const FEMBVBin_local_model& model)
{
   const int n_samples = Y.size();
   const int n_parameters = model.get_n_parameters();

   std::vector<double> grad(n_parameters, 0);
   for (int t = 0; t < n_samples; ++t) {
      for (int i = 0; i < n_parameters; ++i) {
         grad[i] += weights(t) * model.loss_gradient(i, Y(t), X.col(t));
      }
   }

   for (int i = 0; i < n_parameters; ++i) {
      grad[i] -= model.regularization_gradient(i);
   }

   return grad;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
struct FEMBVBin_cost_parameters {
   const OutcomesVector& Y;
   const PredictorsMatrix& X;
   const WeightsVector& weights;
   FEMBVBin_local_model& model;

   FEMBVBin_cost_parameters(const OutcomesVector& Y_, const PredictorsMatrix& X_,
                            const WeightsVector& weights_, FEMBVBin_local_model& model_)
      : Y(Y_), X(X_), weights(weights_), model(model_)
      {}
};

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
double fembv_bin_local_model_cost(
   const std::vector<double>& x, std::vector<double>& dx, void* params)
{
   using Parameters_type = FEMBVBin_cost_parameters<
      OutcomesVector, PredictorsMatrix, WeightsVector>;
   
   Parameters_type* p = static_cast<Parameters_type*>(params);

   const auto Y = p->Y;
   const auto X = p->X;
   const auto weights = p->weights;

   FEMBVBin_local_model& model = p->model;
   model.set_parameters(x);

   if (!dx.empty()) {
      dx = fembv_bin_local_likelihood_grad(Y, X, weights, model);
      for (auto& dxi : dx) {
         dxi *= -1;
      }
   }

   return -fembv_bin_local_likelihood(Y, X, weights, model);
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

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool update_local_fembv_bin_model(
   const OutcomesVector& Y, const PredictorsMatrix& X,
   const WeightsVector& weights, FEMBVBin_local_model& model)
{
   using Parameters_type = FEMBVBin_cost_parameters<
      OutcomesVector, PredictorsMatrix, WeightsVector>;

   nlopt::algorithm algorithm = nlopt::LD_SLSQP;

   const int n_parameters = model.get_n_parameters();
   nlopt::opt optimizer(algorithm, n_parameters);

   Parameters_type params(Y, X, weights, model);

   optimizer.set_min_objective(
      fembv_bin_local_model_cost<OutcomesVector, PredictorsMatrix, WeightsVector>,
      &params);

   optimizer.set_lower_bounds(0);
   optimizer.set_upper_bounds(1);
   optimizer.add_inequality_constraint(
      fembv_bin_local_model_constraint, nullptr, 1e-8);

   optimizer.set_ftol_rel(1.e-6);
   optimizer.set_ftol_abs(1.e-6);

   std::vector<double> x(model.get_parameters());
   double loss_value = 0;
   const auto result = optimizer.optimize(x, loss_value);

   if (result < 0) {
      std::cerr << "optimization failed (exit code: " << result << ")\n";
      throw std::runtime_error("optimization failed");
   }

   model.set_parameters(x);

   return true;
}

template <class OutcomesVector, class PredictorsMatrix, class AffiliationsMatrix>
bool update_fembv_bin_parameters(
   const OutcomesVector& Y, const PredictorsMatrix& X,
   const AffiliationsMatrix& Gamma,
   std::vector<FEMBVBin_local_model>& models)
{
   const int n_components = Gamma.rows();

   bool success = true;
   for (int i = 0; i < n_components; ++i) {
      success = success && update_local_fembv_bin_model(
         Y, X, Gamma.row(i), models[i]);
   }

   return success;
}

template <class OutcomesVector, class PredictorsMatrix, class AffiliationsMatrix,
          class AffiliationsSolver, class DistanceMatrix>
std::tuple<bool, bool, std::size_t, double>
fembv_bin_subspace(
   const OutcomesVector& Y, const PredictorsMatrix& X, AffiliationsMatrix& Gamma,
   std::vector<FEMBVBin_local_model>& models, AffiliationsSolver& gamma_solver,
   DistanceMatrix& G, std::size_t max_iterations, double tolerance, bool update_parameters,
   int verbosity)
{
   auto initial_cost = fembv_bin_cost(Gamma, models, G);

   auto old_cost = initial_cost;
   bool parameters_success = true;
   bool affiliations_success = true;
   std::size_t n_iter = 0;
   bool converged = false;
   while (n_iter < max_iterations) {
      if (update_parameters) {
         parameters_success = update_fembv_bin_parameters(Y, X, Gamma, models);
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
   bool update_parameters{true};
   int verbosity{0};
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

   ClpSimplex_affiliations_solver gamma_solver(G, V, parameters.max_tv_norm);
   gamma_solver.set_max_iterations(parameters.max_affiliations_iterations);
   gamma_solver.set_verbosity(parameters.verbosity);

   const auto result = detail::fembv_bin_subspace(
      Y, X, Gamma, models, gamma_solver, G,
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
   int verbosity{0};
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
