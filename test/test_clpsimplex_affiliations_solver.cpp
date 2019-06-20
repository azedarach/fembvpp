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