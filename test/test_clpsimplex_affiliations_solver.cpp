#include "catch/catch.hpp"

#include "clpsimplex_affiliations_solver.hpp"

#include <Eigen/Core>

using namespace fembvpp;

TEST_CASE("test basic construction", "[ClpSimplex_affiliations_solver]")
{
   SECTION("maximum TV norm set correctly")
   {
      const int n_components = 5;
      const int n_elements = 10;
      const int n_samples = 20;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      const double max_tv_norm = 1.;

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_max_tv_norm() == max_tv_norm);
   }

   SECTION("number of components set correctly")
   {
      const int n_components = 6;
      const int n_elements = 10;
      const int n_samples = 30;
      const double max_tv_norm = 0;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_components() == n_components);
   }

   SECTION("number of elements set correctly")
   {
      const int n_components = 6;
      const int n_elements = 10;
      const int n_samples = 30;
      const double max_tv_norm = 0;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_elements() == n_elements);
   }

   SECTION("number of samples set correctly")
   {
      const int n_components = 6;
      const int n_elements = 10;
      const int n_samples = 30;
      const double max_tv_norm = 0;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_samples() == n_samples);
   }
}

TEST_CASE("test correct number of variables added", "[ClpSimplex_affiliations_solver]")
{
   SECTION("stores correct number of primary variables with no TV norm constraint")
   {
      const int n_components = 12;
      const int n_elements = 40;
      const int n_samples = 100;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_primary_variables() == n_components * n_elements);
   }

   SECTION("stores correct number of primary variables with TV norm constraint")
   {
      const int n_components = 2;
      const int n_elements = 30;
      const int n_samples = 200;
      const double max_tv_norm = 10;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_primary_variables() == n_components * n_elements);
   }

   SECTION("does not add auxiliary variables when no norm constraint imposed")
   {
      const int n_components = 10;
      const int n_elements = 10;
      const int n_samples = 400;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_auxiliary_variables() == 0);
   }

   SECTION("stores correct number of auxiliary variables when TV norm constraint imposed")
   {
      const int n_components = 15;
      const int n_elements = 20;
      const int n_samples = 300;
      const double max_tv_norm = 12;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_auxiliary_variables() == n_components * n_elements);
   }

   SECTION("stores correct number of variables when no norm constraint imposed")
   {
      const int n_components = 4;
      const int n_elements = 100;
      const int n_samples = 100;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_total_variables() == n_components * n_elements);
   }

   SECTION("stores correct number of variables when norm constraint imposed")
   {
      const int n_components = 3;
      const int n_elements = 4;
      const int n_samples = 300;
      const double max_tv_norm = 23;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_total_variables() == 2 * n_components * n_elements);
   }
}

TEST_CASE("test correct number of constraints added", "[ClpSimplex_affiliations_solver]")
{
   SECTION("stores correct total number of constraints with no TV norm constraint")
   {
      const int n_components = 12;
      const int n_elements = 40;
      const int n_samples = 100;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      // ensure no trivial bounds constraints
      const Eigen::MatrixXd V(Eigen::MatrixXd::Ones(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto expected_n_constraints = (n_components + 1) * n_samples;
      CHECK(solver.get_n_constraints() == expected_n_constraints);
   }

   SECTION("identifies trivial bounds when no norm constraint imposed")
   {
      const int n_components = 4;
      const int n_elements = 40;
      const int n_samples = 40;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      // all positivity constraints in this case reduce to trivial
      // bounds, so the number of constraints is just equal to the
      // number of stochastic constraints (i.e., n_samples)
      const auto expected_n_constraints = n_samples;
      CHECK(solver.get_n_constraints() == expected_n_constraints);
   }

   SECTION("stores correct total number of constraints when norm constraint imposed")
   {
      const int n_components = 4;
      const int n_elements = 5;
      const int n_samples = 200;
      const double max_tv_norm = 2;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      // ensure no trivial bounds constraints
      const Eigen::MatrixXd V(Eigen::MatrixXd::Ones(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto expected_n_constraints =
         (n_components + 1) * n_samples + 3 * n_components * (n_samples - 1)
         + n_components;
      CHECK(solver.get_n_constraints() == expected_n_constraints);
   }

   SECTION("identifies trivial bounds with norm constraint imposed")
   {
      const int n_components = 10;
      const int n_elements = 100;
      const int n_samples = 100;
      const double max_tv_norm = 10;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      // all positivity constraints on the affiliations and auxiliary
      // variables are now just simple bounds, with the non-trivial
      // constraints being the stochasticity constraints at each
      // time point (n_samples constraints), the auxiliary norm
      // constraints (2 * n_components * (n_samples - 1) constraints)
      // and the TV norm constraints (n_components constraints).
      const auto expected_n_constraints =
         n_samples + 2 * n_components * (n_samples - 1) + n_components;
      CHECK(solver.get_n_constraints() == expected_n_constraints);
   }
}

TEST_CASE("test objective coefficients stored correctly")
{
   SECTION("stores correct number of objective coefficients with no norm constraint")
   {
      const int n_components = 5;
      const int n_elements = 24;
      const int n_samples = 24;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Ones(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Ones(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const int expected_n_coeffs = n_components * n_elements;

      const std::vector<double> coeffs(solver.get_objective_coefficients());

      CHECK(coeffs.size() == expected_n_coeffs);
   }

   SECTION("stores correct number of objective coefficients with norm constraint")
   {
      const int n_components = 10;
      const int n_elements = 3;
      const int n_samples = 54;
      const double max_tv_norm = 5.5;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Ones(n_components, n_samples));
      const Eigen::MatrixXd V(Eigen::MatrixXd::Ones(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const int expected_n_coeffs = n_components * n_elements;

      const std::vector<double> coeffs(solver.get_objective_coefficients());

      CHECK(coeffs.size() == expected_n_coeffs);
   }

   SECTION("returns correct value for objective coefficients with no norm constraint")
   {
      const double tol = 1e-12;
      const int n_components = 3;
      const int n_elements = 3;
      const int n_samples = 12;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      const Eigen::MatrixXd V(Eigen::MatrixXd::Random(n_elements, n_samples).cwiseAbs());

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const Eigen::MatrixXd expected_objective_matrix = G * V.transpose();

      std::vector<double> expected_objective_vector(n_components * n_elements);
      for (int j = 0; j < n_elements; ++j) {
         for (int i = 0; i < n_components; ++i) {
            expected_objective_vector[i + j * n_components] =
               expected_objective_matrix(i, j);
         }
      }

      std::vector<double> objective_vector(solver.get_objective_coefficients());
      for (int i = 0; i < n_components * n_elements; ++i) {
         CHECK(std::abs(expected_objective_vector[i] - objective_vector[i]) < tol);
      }

      Eigen::MatrixXd objective_matrix(n_components, n_elements);
      solver.get_objective_coefficients(objective_matrix);

      const double max_diff = (objective_matrix - expected_objective_matrix).cwiseAbs().maxCoeff();

      CHECK(max_diff < tol);
   }

   SECTION("returns correct value for objective coefficients with norm constraint")
   {
      const double tol = 1e-12;
      const int n_components = 4;
      const int n_elements = 10;
      const int n_samples = 50;
      const double max_tv_norm = 23.;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      const Eigen::MatrixXd V(Eigen::MatrixXd::Random(n_elements, n_samples).cwiseAbs());

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const Eigen::MatrixXd expected_objective_matrix = G * V.transpose();

      std::vector<double> expected_objective_vector(n_components * n_elements);
      for (int j = 0; j < n_elements; ++j) {
         for (int i = 0; i < n_components; ++i) {
            expected_objective_vector[i + j * n_components] =
               expected_objective_matrix(i, j);
         }
      }

      std::vector<double> objective_vector(solver.get_objective_coefficients());
      for (int i = 0; i < n_components * n_elements; ++i) {
         CHECK(std::abs(expected_objective_vector[i] - objective_vector[i]) < tol);
      }

      Eigen::MatrixXd objective_matrix(n_components, n_elements);
      solver.get_objective_coefficients(objective_matrix);

      const double max_diff = (objective_matrix - expected_objective_matrix).cwiseAbs().maxCoeff();

      CHECK(max_diff < tol);
   }
}

TEST_CASE("test objective coefficients updated correctly")
{
   SECTION("returns correct updated values with no norm constraint")
   {
      const double tol = 1e-12;
      const int n_components = 7;
      const int n_elements = 25;
      const int n_samples = 25;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      const Eigen::MatrixXd V(Eigen::MatrixXd::Random(n_elements, n_samples).cwiseAbs());

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const Eigen::MatrixXd expected_objective_matrix = G * V.transpose();

      Eigen::MatrixXd initial_objective_matrix(n_components, n_elements);
      solver.get_objective_coefficients(initial_objective_matrix);

      double max_diff = (initial_objective_matrix - expected_objective_matrix).cwiseAbs().maxCoeff();
      REQUIRE(max_diff < tol);

      const Eigen::MatrixXd G2(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      max_diff = (G2 - G).cwiseAbs().maxCoeff();
      // require matrices to differ in at least one element
      REQUIRE(max_diff > tol);

      solver.update_affiliations(G2);

      const Eigen::MatrixXd expected_new_matrix = G2 * V.transpose();
      Eigen::MatrixXd new_objective_matrix(n_components, n_elements);
      solver.get_objective_coefficients(new_objective_matrix);

      max_diff = (new_objective_matrix - expected_new_matrix).cwiseAbs().maxCoeff();

      CHECK(max_diff < tol);
   }

   SECTION("returns correct updated values with norm constraint")
   {
      const double tol = 1e-12;
      const int n_components = 2;
      const int n_elements = 505;
      const int n_samples = 505;
      const double max_tv_norm = 4.;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      const Eigen::MatrixXd V(Eigen::MatrixXd::Random(n_elements, n_samples).cwiseAbs());

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const Eigen::MatrixXd expected_objective_matrix = G * V.transpose();

      Eigen::MatrixXd initial_objective_matrix(n_components, n_elements);
      solver.get_objective_coefficients(initial_objective_matrix);

      double max_diff = (initial_objective_matrix - expected_objective_matrix).cwiseAbs().maxCoeff();
      REQUIRE(max_diff < tol);

      const Eigen::MatrixXd G2(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      max_diff = (G2 - G).cwiseAbs().maxCoeff();
      // require matrices to differ in at least one element
      REQUIRE(max_diff > tol);

      solver.update_affiliations(G2);

      const Eigen::MatrixXd expected_new_matrix = G2 * V.transpose();
      Eigen::MatrixXd new_objective_matrix(n_components, n_elements);
      solver.get_objective_coefficients(new_objective_matrix);

      max_diff = (new_objective_matrix - expected_new_matrix).cwiseAbs().maxCoeff();

      CHECK(max_diff < tol);
   }
}