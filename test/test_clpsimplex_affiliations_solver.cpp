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

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

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

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_components() == n_components);
   }

   SECTION("number of elements set correctly")
   {
      const int n_components = 6;
      const int n_elements = 10;
      const int n_samples = 30;
      const double max_tv_norm = 0;

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_elements() == n_elements);
   }

   SECTION("number of samples set correctly")
   {
      const int n_components = 6;
      const int n_elements = 10;
      const int n_samples = 30;
      const double max_tv_norm = 0;

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_samples() == n_samples);
   }
}

TEST_CASE("test number of variables correctly added", "[ClpSimplex_affiliations_solver]")
{
   SECTION("stores correct number of primary variables with no TV norm constraint")
   {
      const int n_components = 12;
      const int n_elements = 40;
      const int n_samples = 100;
      const double max_tv_norm = -1;

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_primary_variables() == n_components * n_elements);
   }

   SECTION("stores correct number of primary variables with TV norm constraint")
   {
      const int n_components = 2;
      const int n_elements = 30;
      const int n_samples = 200;
      const double max_tv_norm = 10;

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_primary_variables() == n_components * n_elements);
   }

   SECTION("does not add auxiliary variables when no norm constraint imposed")
   {
      const int n_components = 10;
      const int n_elements = 10;
      const int n_samples = 400;
      const double max_tv_norm = -1;

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_auxiliary_variables() == 0);
   }

   SECTION("stores correct number of auxiliary variables when TV norm constraint imposed")
   {
      const int n_components = 15;
      const int n_elements = 20;
      const int n_samples = 300;
      const double max_tv_norm = 12;

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_auxiliary_variables() == n_components * n_elements);
   }

   SECTION("stores correct number of variables when no norm constraint imposed")
   {
      const int n_components = 4;
      const int n_elements = 100;
      const int n_samples = 100;
      const double max_tv_norm = -1;

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_total_variables() == n_components * n_elements);
   }

   SECTION("stores correct number of variables when norm constraint imposed")
   {
      const int n_components = 3;
      const int n_elements = 4;
      const int n_samples = 300;
      const double max_tv_norm = 23;

      Eigen::MatrixXd G(Eigen::MatrixXd::Zero(n_components, n_samples));
      Eigen::MatrixXd V(Eigen::MatrixXd::Zero(n_elements, n_samples));

      ClpSimplex_affiliations_solver solver(G, V, max_tv_norm);

      CHECK(solver.get_n_total_variables() == 2 * n_components * n_elements);
   }
}