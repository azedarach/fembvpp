#ifndef FEMBVPP_FEMBV_BIN_LOCAL_MODEL_IPOPT_SOLVER_HPP_INCLUDED
#define FEMBVPP_FEMBV_BIN_LOCAL_MODEL_IPOPT_SOLVER_HPP_INCLUDED

#include "fembv_bin_local_model.hpp"

#include <IpIpoptApplication.hpp>
#include <IpTNLP.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace fembvpp {

enum class Ipopt_initial_guess : int { Uniform, Random, Current };

namespace detail {

std::string ipopt_status_to_string(Ipopt::SolverReturn);

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
class FEMBVBin_local_nlp : public Ipopt::TNLP {
public:
   using Index = Ipopt::Index;
   using Number = Ipopt::Number;

   FEMBVBin_local_nlp(
      const OutcomesVector&, const PredictorsMatrix&,
      const WeightsVector&, FEMBVBin_local_model&, int,
      Ipopt_initial_guess, int);
   FEMBVBin_local_nlp(
      const OutcomesVector&, const PredictorsMatrix&,
      const WeightsVector&, FEMBVBin_local_model&, int,
      Ipopt_initial_guess);
   FEMBVBin_local_nlp(
      const OutcomesVector&, const PredictorsMatrix&,
      const WeightsVector&, FEMBVBin_local_model&, int);
   FEMBVBin_local_nlp(
      const OutcomesVector&, const PredictorsMatrix&,
      const WeightsVector&, FEMBVBin_local_model&);
   virtual ~FEMBVBin_local_nlp() = default;
   FEMBVBin_local_nlp(const FEMBVBin_local_nlp&) = delete;
   FEMBVBin_local_nlp operator=(const FEMBVBin_local_nlp&) = delete;

   void set_initialization_method(Ipopt_initial_guess i) { initialization = i; }
   void set_verbosity(int v) { verbosity = v; }

   virtual bool get_nlp_info(Index&, Index&, Index&, Index&,
                             Ipopt::TNLP::IndexStyleEnum&) override;
   virtual bool get_bounds_info(Index, Number*, Number*, Index,
                                Number*, Number*) override;
   virtual bool get_starting_point(Index, bool, Number*, bool, Number*,
                                   Number*, Index, bool,Number*) override;
   virtual bool eval_f(Index, const Number*, bool, Number&) override;
   virtual bool eval_grad_f(Index, const Number*, bool, Number*) override;
   virtual bool eval_g(Index, const Number*, bool, Index, Number*) override;
   virtual bool eval_jac_g(Index, const Number*, bool, Index,
                           Index, Index*, Index*, Number*) override;
   virtual bool eval_h(Index, const Number*, bool, Number,
                       Index, const Number*, bool, Index,
                       Index*, Index*, Number*) override;
   virtual void finalize_solution(
      Ipopt::SolverReturn, Index, const Number*, const Number*,
      const Number*, Index, const Number*,
      const Number*, Number, const Ipopt::IpoptData*,
      Ipopt::IpoptCalculatedQuantities*) override;

   double get_objective_value() const { return objective_value; }
   const std::vector<double>& get_parameters() const { return model.get_parameters(); }

private:
   std::mt19937 generator{};
   int verbosity{0};
   double objective_value{-1};
   Ipopt_initial_guess initialization{Ipopt_initial_guess::Uniform};
   const OutcomesVector& Y;
   const PredictorsMatrix& X;
   const WeightsVector& weights;
   FEMBVBin_local_model& model;

   double calculate_predictor_sum(int, const Number*, int) const;
   bool update_succeeded(Ipopt::SolverReturn) const;
};

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::FEMBVBin_local_nlp(
   const OutcomesVector& Y_, const PredictorsMatrix& X_,
   const WeightsVector& weights_, FEMBVBin_local_model& model_,
   int verbosity_, Ipopt_initial_guess initialization_, int seed_)
   : generator(seed_)
   , verbosity(verbosity_)
   , initialization(initialization_)
   , Y(Y_)
   , X(X_)
   , weights(weights_)
   , model(model_)
{
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::FEMBVBin_local_nlp(
   const OutcomesVector& Y_, const PredictorsMatrix& X_,
   const WeightsVector& weights_, FEMBVBin_local_model& model_,
   int verbosity_, Ipopt_initial_guess initialization_)
   : verbosity(verbosity_)
   , initialization(initialization_)
   , Y(Y_)
   , X(X_)
   , weights(weights_)
   , model(model_)
{
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::FEMBVBin_local_nlp(
   const OutcomesVector& Y_, const PredictorsMatrix& X_,
   const WeightsVector& weights_, FEMBVBin_local_model& model_,
   int verbosity_)
   : verbosity(verbosity_)
   , Y(Y_)
   , X(X_)
   , weights(weights_)
   , model(model_)
{
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::FEMBVBin_local_nlp(
   const OutcomesVector& Y_, const PredictorsMatrix& X_,
   const WeightsVector& weights_, FEMBVBin_local_model& model_)
   : Y(Y_)
   , X(X_)
   , weights(weights_)
   , model(model_)
{
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
double FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::calculate_predictor_sum(
   int n, const Number* x, int t) const
{
   const auto& predictor_indices = model.get_predictor_indices();
   double lp = 0.0;
   for (Index i = 0; i < n; ++i) {
      lp += x[i] * X(predictor_indices[i], t);
   }
   return lp;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::get_nlp_info(
   Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag,
   Ipopt::TNLP::IndexStyleEnum& index_style)
{
   // number of parameters
   n = model.get_n_parameters();

   // number of constraints other than variable bounds
   m = 1;

   // number of non-zero entries in the constraint Jacobian
   nnz_jac_g = n;

   // number of distinct non-zero entries in the Hessian for the Lagrangian,
   // noting that it is symmetric
   nnz_h_lag = n * (n + 1) / 2;

   // C-style indexing
   index_style = TNLP::C_STYLE;

   return true;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::get_bounds_info(
   Index n, Number* x_L, Number* x_U, Index /* m */, Number* g_L, Number* g_U)
{
   // parameters are bounded between 0 and 1
   for (Index i = 0; i < n; ++i) {
      x_L[i] = 0.0;
      x_U[i] = 1.0;
   }

   // constraints are bounded between 0 and 1
   g_L[0] = 0.0;
   g_U[0] = 1.0;

   return true;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::get_starting_point(
   Index n, bool /* init_x */, Number* x, bool init_z, Number* /* z_L */, Number* /* z_U */,
   Index /* m */, bool init_lambda, Number* /* lambda */)
{
   if (init_z || init_lambda) {
      throw std::runtime_error("initialization of dual variables not implemented");
   }

   switch (initialization) {
      case Ipopt_initial_guess::Uniform: {
         for (Index i = 0; i < n; ++i) {
            x[i] = 1.0 / (n + 1);
         }
         break;
      }
      case Ipopt_initial_guess::Current: {
         const auto& parameters = model.get_parameters();
         for (Index i = 0; i < n; ++i) {
            x[i] = parameters[i];
         } 
      }
      case Ipopt_initial_guess::Random: {
         std::uniform_real_distribution<> dist(0., 1.);
         double sum = 0.;
         for (Index i = 0; i < n; ++i) {
            x[i] = dist(generator);
            sum += x[i];
         }

         const double target_sum = dist(generator);
         for (Index i = 0; i < n; ++i) {
            x[i] *= target_sum / sum;
         }
      }
   }

   if (verbosity > 0) {
      std::cout << "Initial x = ";
      for (Index i = 0; i < n; ++i) {
         std::cout << x[i] << ' ';
      }
      std::cout << '\n';
   }

   return true;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::eval_f(
   Index n, const Number* x, bool /* new_x */, Number& f)
{
   using std::abs;
   using std::log;

   const auto& predictor_indices = model.get_predictor_indices();

   double log_like = 0;
   const int n_samples = Y.size();
   for (int t = 0; t < n_samples; ++t) {
      const double lp = calculate_predictor_sum(n, x, t);

      if (lp == 0 && Y(t) == 0) {
         continue;
      } else if (lp == 1 && Y(t) == 1) {
         continue;
      } else if (lp <= 0.0 || lp >= 1.0) {
         return false;
      } else {
         log_like -= weights(t) * (Y(t) * log(lp) + (1 - Y(t)) * log(1 - lp));
      }
   }

   double penalty = 0;
   for (Index i = 0; i < n; ++i) {
      penalty += abs(x[i]);
   }

   f = log_like + model.epsilon * penalty;

   return true;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::eval_grad_f(
   Index n, const Number* x, bool /* new_x */, Number* grad_f)
{
   for (int i = 0; i < n; ++i) {
      grad_f[i] = 0.0;
   }

   const auto& predictor_indices = model.get_predictor_indices();
   const int n_samples = Y.size();
   for (int t = 0; t < n_samples; ++t) {
      const double lp = calculate_predictor_sum(n, x, t);

      if (lp <= 0.0 || lp >= 1.0) {
         return false;
      }

      for (Index i = 0; i < n; ++i) {
         grad_f[i] -= weights(t) * (Y(t) * X(predictor_indices[i], t) / lp
            - (1 - Y(t)) * X(predictor_indices[i], t) / (1 - lp));
      }
   }

   for (Index i = 0; i < n; ++i) {
      grad_f[i] += model.epsilon;
   }

  return true;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::eval_g(
   Index n, const Number* x, bool /* new_x */, Index /* m */, Number* g)
{
   g[0] = 0.0;
   for (Index i = 0; i < n; ++i) {
      g[0] += x[i];
   }
   return true;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::eval_jac_g(
   Index n, const Number* /* x */, bool /* new_x */, Index /* m */,
   Index nele_jac, Index* i_row, Index* j_col, Number* values)
{
   if (values == NULL) {
      // dense Jacobian
      Index idx = 0;
      for (Index i = 0; i < n; ++i) {
         i_row[idx] = 0;
         j_col[idx] = i;
         ++idx;
      }
   } else {
      for (Index i = 0; i < nele_jac; ++i) {
         values[i] = 1.0;
      }
   }

   return true;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::eval_h(
   Index n, const Number* x, bool /* new_x */, Number obj_factor,
   Index /* m */, const Number* /* lambda */, bool /* new_lambda */, Index /* nele_hess */,
   Index* i_row, Index* j_col, Number* values)
{
   if (values == NULL) {
      Index idx = 0;
      for (Index i = 0; i < n; ++i) {
         for (Index j = 0; j <= i; ++j) {
            i_row[idx] = i;
            j_col[idx] = j;
            ++idx;
         }
      }
   } else {
      Index idx = 0;
      for (Index i = 0; i < n; ++i) {
         for (Index j = 0; j <= i; ++j) {
            values[idx] = 0.0;
            ++idx;
         }
      }

      const auto& predictor_indices = model.get_predictor_indices();
      const int n_samples = Y.size();
      for (int t = 0; t < n_samples; ++t) {
         double lp = calculate_predictor_sum(n, x, t);

         Index idx = 0;
         for (Index i = 0; i < n; ++i) {
            for (Index j = 0; j <= i; ++j) {
               const auto xi = X(predictor_indices[i], t);
               const auto xj = X(predictor_indices[j], t);
               const auto yt = Y(t);

               values[idx] += obj_factor * weights(t) *
                  (yt * xi * xj / (lp * lp)
                  + (1 - yt) * xi * xj / ((1 - lp) * (1 - lp)));
               ++idx;
            }
         }
      }
   }

   return true;
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::update_succeeded(
    Ipopt::SolverReturn status) const
{
   if (status == Ipopt::SUCCESS || status == Ipopt::STOP_AT_ACCEPTABLE_POINT) {
      return true;
   } else {
      return false;
   }
}

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
void FEMBVBin_local_nlp<OutcomesVector, PredictorsMatrix, WeightsVector>::finalize_solution(
   Ipopt::SolverReturn status, Index n, const Number* x, const Number* z_L,
   const Number* z_U, Index m, const Number* g,
   const Number* lambda, Number obj_value,
   const Ipopt::IpoptData* /* ip_data */,
   Ipopt::IpoptCalculatedQuantities* /* ip_cq */)
{
   if (update_succeeded(status)) {
      for (Index i = 0; i < n; ++i) {
      model.set_parameter(i, x[i]);
      }

      objective_value = obj_value;
   }

   if (verbosity > 0) {
      std::cout << "Finalizing solution for local model update\n";
      std::cout << "Solver status: " << ipopt_status_to_string(status) << '\n';
      std::cout << "Solution x = ";
      for (Index i = 0; i < n; ++i) {
         std::cout << x[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "Dual z_L = ";
      for (Index i = 0; i < n; ++i) {
         std::cout << z_L[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "Dual z_U = ";
      for (Index i = 0; i < n; ++i) {
         std::cout << z_U[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "Constraint values g = ";
      for (Index i = 0; i < m; ++i) {
         std::cout << g[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "Constraint lambda = ";
      for (Index i = 0; i < m; ++i) {
         std::cout << lambda[i] << ' ';
      }
      std::cout << '\n';
      std::cout << "Objective value: " << obj_value << '\n';
   }
}

} // namespace detail

class FEMBVBin_local_model_ipopt_solver {
public:
   FEMBVBin_local_model_ipopt_solver();
   explicit FEMBVBin_local_model_ipopt_solver(int);

   void set_initialization_method(Ipopt_initial_guess i) { initialization = i; }
   void set_max_iterations(int it);
   void set_n_trials(int n) { n_trials = n; }
   void set_tolerance(double t);
   void set_verbosity(int v) { verbosity = v; }

   void initialize();

   template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
   bool update_local_model(const OutcomesVector&, const PredictorsMatrix&,
                           const WeightsVector&, FEMBVBin_local_model&);
private:
   Ipopt_initial_guess initialization{Ipopt_initial_guess::Uniform};
   int verbosity{0};
   int n_trials{10};
   std::mt19937 generator{};
   Ipopt::SmartPtr<Ipopt::IpoptApplication> ip_solver{};
};

template <class OutcomesVector, class PredictorsMatrix, class WeightsVector>
bool FEMBVBin_local_model_ipopt_solver::update_local_model(
   const OutcomesVector& Y, const PredictorsMatrix& X,
   const WeightsVector& weights, FEMBVBin_local_model& model)
{
   using NLP_type = detail::FEMBVBin_local_nlp<OutcomesVector,
                                               PredictorsMatrix,
                                               WeightsVector>;
   std::uniform_int_distribution<> dist(0);
   const int seed = dist(generator);

   Ipopt::SmartPtr<Ipopt::TNLP> nlp = new NLP_type(
      Y, X, weights, model, verbosity, initialization, seed);

   if (initialization == Ipopt_initial_guess::Random) {
      bool success = false;
      double min_objective = std::numeric_limits<double>::max();
      std::vector<double> best_parameters(model.get_parameters());

      for (int i = 0; i < n_trials; ++i) {
         const Ipopt::ApplicationReturnStatus status = ip_solver->OptimizeTNLP(nlp);

         if (status == Ipopt::Solve_Succeeded) {
            NLP_type* result = static_cast<NLP_type*>(Ipopt::GetRawPtr(nlp));
            const auto cost = result->get_objective_value();
            if (!success || cost < min_objective) {
               min_objective = cost;
               best_parameters = result->get_parameters();
               success = true;
            }
         }
      }

      model.set_parameters(best_parameters);

      if (success) {
         return success;
      }
   }

   const Ipopt::ApplicationReturnStatus status = ip_solver->OptimizeTNLP(nlp);

   return status == Ipopt::Solve_Succeeded;
}

} // namespace fembvpp

#endif
