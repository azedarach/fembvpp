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

TEST_CASE("test objective coefficients stored correctly", "[ClpSimplex_affiliations_solver]")
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

TEST_CASE("test objective coefficients updated correctly", "[ClpSimplex_affiliations_solver]")
{
   SECTION("returns correct updated values with no norm constraint")
   {
      const double tol = 1e-12;
      const int n_components = 7;
      const int n_elements = 25;
      const int n_samples = 25;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

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
      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

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

TEST_CASE("returns trivial solution when only one component", "[ClpSimplex_affiliations_solver]")
{
   SECTION("returns all ones for affiliations when no norm constraint imposed")
   {
      const double tol = 1e-15;
      const int n_components = 1;
      const int n_elements = 100;
      const int n_samples = 100;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      solver.update_affiliations(G);

      const Eigen::MatrixXd expected_Gamma(Eigen::MatrixXd::Ones(n_components, n_samples));
      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      const double max_diff = (Gamma - expected_Gamma).cwiseAbs().maxCoeff();
      CHECK(max_diff < tol);
   }

   SECTION("returns all ones for affiliations when norm constraint imposed")
   {
      const double tol = 1e-15;
      const int n_components = 1;
      const int n_elements = 5;
      const int n_samples = 500;
      const double max_tv_norm = 10;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      solver.update_affiliations(G);

      const Eigen::MatrixXd expected_Gamma(Eigen::MatrixXd::Ones(n_components, n_samples));
      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      const double max_diff = (Gamma - expected_Gamma).cwiseAbs().maxCoeff();
      CHECK(max_diff < tol);
   }
}

TEST_CASE("returns expected solution for test problems", "[ClpSimplex_affiliations_solver]")
{
   SECTION("returns expected solution for case of non-overlapping perfect models")
   {
      const double tol = 1e-15;
      const int n_components = 3;
      const int n_elements = 10;
      const int n_samples = 10;
      const double max_tv_norm = -1;

      // distance matrix for 3 local models in which
      // each model exactly fits at non-overlapping times
      Eigen::MatrixXd G(n_components, n_samples);
      G << 0.5, 0.45, 0.65, 0, 0, 0, 0, 1.2, 0.1, 2.4,
      0, 0, 0, 1.2, 10.2, 0.1, 5.6, 2.4, 5.3, 2.9,
      0.1, 4.5, 2.3, 2.3, 9.8, 3.5, 1.8, 0, 0, 0;

      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      // in this case, optimal solution is deterministic
      // affiliation corresponding to perfect model at each
      // time
      Eigen::MatrixXd expected_Gamma(n_components, n_samples);
      expected_Gamma << 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
      1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 1, 1;

      Eigen::MatrixXd obtained_Gamma(n_components, n_samples);
      solver.get_affiliations(obtained_Gamma);

      const double max_diff = (obtained_Gamma - expected_Gamma).cwiseAbs().maxCoeff();
      CHECK(max_diff < tol);
   }

   SECTION("returns expected solution for case of non-overlapping perfect models with no switching")
   {
      const double tol = 1e-15;
      const int n_components = 2;
      const int n_elements = 10;
      const int n_samples = 10;
      const double max_tv_norm = 0;

      Eigen::MatrixXd G(n_components, n_samples);
      G << 0.5, 0.2, 0.6, 0.1, 0.2, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1.2, 10, 2.3, 4.3, 5.0;

      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      // here no switching is allowed, so the optimal solution is
      // to stay in the model with the minimum loss
      Eigen::MatrixXd expected_Gamma(Eigen::MatrixXd::Zero(n_components, n_samples));
      expected_Gamma.row(0) = Eigen::VectorXd::Ones(n_samples);

      Eigen::MatrixXd obtained_Gamma(n_components, n_samples);
      solver.get_affiliations(obtained_Gamma);

      const double max_diff = (obtained_Gamma - expected_Gamma).cwiseAbs().maxCoeff();
      CHECK(max_diff < tol);
   }
}

TEST_CASE("test solution satisfies constraints", "[ClpSimplex_affiliations_solver]")
{
   SECTION("test affiliations are non-negative when no norm constraint imposed")
   {
      const int n_components = 4;
      const int n_elements = 10;
      const int n_samples = 100;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());

      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));
      for (int i = 0; i < n_elements; ++i) {
         V.block(i, 10 * i, 1, 10) = Eigen::VectorXd::Ones(10);
      }

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      CHECK((Gamma.array() >= 0).all());
   }

   SECTION("test affiliations are stochastic when no norm constraint imposed")
   {
      const double tol = 1e-12;
      const int n_components = 5;
      const int n_elements = 300;
      const int n_samples = 300;
      const double max_tv_norm = -1;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());

      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      const Eigen::RowVectorXd col_sums = Gamma.colwise().sum();
      const double max_diff = (col_sums - Eigen::RowVectorXd::Ones(n_samples)).cwiseAbs().maxCoeff();
      CHECK(max_diff < tol);
   }

   SECTION("test affiliations are non-negative when no switching imposed")
   {
      const int n_components = 2;
      const int n_elements = 4;
      const int n_samples = 200;
      const double max_tv_norm = 0;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());

      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));
      for (int i = 0; i < n_elements; ++i) {
         V.block(i, 50 * i, 1, 50) = Eigen::VectorXd::Ones(50);
      }

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      CHECK((Gamma.array() >= 0).all());
   }

   SECTION("test affiliations are stochastic when no switching imposed")
   {
      const double tol = 1e-12;
      const int n_components = 4;
      const int n_elements = 6;
      const int n_samples = 60;
      const double max_tv_norm = 0;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());

      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));
      for (int i = 0; i < n_elements; ++i) {
         V.block(i, 10 * i, 1, 10) = Eigen::VectorXd::Ones(10);
      }

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      const Eigen::RowVectorXd col_sums = Gamma.colwise().sum();
      const double max_diff = (col_sums - Eigen::RowVectorXd::Ones(n_samples)).cwiseAbs().maxCoeff();
      CHECK(max_diff < tol);
   }

   SECTION("test affiliations respect norm constraint when no switching imposed")
   {
      const double tol = 1e-12;
      const int n_components = 5;
      const int n_elements = 50;
      const int n_samples = 50;
      const double max_tv_norm = 0;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      Eigen::VectorXd norms(Eigen::VectorXd::Zero(n_components));
      for (int t = 0; t < n_samples - 1; ++t) {
         norms += (Gamma.col(t + 1) - Gamma.col(t)).cwiseAbs();
      }

      CHECK((norms.array() <= max_tv_norm + tol).all());
   }

   SECTION("test affiliations are non-negative when norm constraint imposed")
   {
      const int n_components = 2;
      const int n_elements = 500;
      const int n_samples = 500;
      const double max_tv_norm = 10;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());
      const Eigen::MatrixXd V(Eigen::MatrixXd::Identity(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      CHECK((Gamma.array() >= 0).all());
   }

   SECTION("test affiliations are stochastic when norm constraint imposed")
   {
      const double tol = 1e-12;
      const int n_components = 10;
      const int n_elements = 25;
      const int n_samples = 50;
      const double max_tv_norm = 20;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());

      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));
      for (int i = 0; i < n_elements; ++i) {
         V.block(i, 2 * i, 1, 2) = Eigen::VectorXd::Ones(2);
      }

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      const Eigen::RowVectorXd col_sums = Gamma.colwise().sum();
      const double max_diff = (col_sums - Eigen::RowVectorXd::Ones(n_samples)).cwiseAbs().maxCoeff();
      CHECK(max_diff < tol);
   }

   SECTION("test affiliations satisfy norm constraint when norm constraint imposed")
   {
      const double tol = 1e-12;
      const int n_components = 4;
      const int n_elements = 30;
      const int n_samples = 90;
      const double max_tv_norm = 5;

      const Eigen::MatrixXd G(Eigen::MatrixXd::Random(n_components, n_samples).cwiseAbs());

      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));
      for (int i = 0; i < n_elements; ++i) {
         V.block(i, 3 * i, 1, 3) = Eigen::VectorXd::Ones(3);
      }

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      const auto status = solver.update_affiliations(G);

      REQUIRE(status == ClpSimplex_affiliations_solver::Status::SUCCESS);

      Eigen::MatrixXd Gamma(n_components, n_samples);
      solver.get_affiliations(Gamma);

      Eigen::VectorXd norms(Eigen::VectorXd::Zero(n_components));
      for (int t = 0; t < n_samples - 1; ++t) {
         norms += (Gamma.col(t + 1) - Gamma.col(t)).cwiseAbs();
      }

      CHECK((norms.array() <= max_tv_norm + tol).all());
   }
}
