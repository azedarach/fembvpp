#include "catch/catch.hpp"

#include "random_matrix.hpp"

#include <Eigen/Core>

#include <random>

using namespace fembvpp;

TEST_CASE("test making left stochastic matrix", "[random_matrix]")
{
   SECTION("column sums of result equal one")
   {
      const double tol = 1e-12;

      const int n_rows = 10;
      const int n_cols = 40;
      Eigen::MatrixXd A(Eigen::MatrixXd::Random(n_rows, n_cols));

      const Eigen::RowVectorXd ones_vector(Eigen::RowVectorXd::Ones(n_cols));

      const Eigen::RowVectorXd initial_sums = A.colwise().sum();
      const double initial_diff = (initial_sums - ones_vector).cwiseAbs().maxCoeff();

      REQUIRE(initial_diff > tol);

      make_left_stochastic(A);

      const Eigen::RowVectorXd final_sums = A.colwise().sum();
      const double final_diff = (final_sums - ones_vector).cwiseAbs().maxCoeff();

      CHECK(final_diff < tol);
   }
}

TEST_CASE("test making right stochastic matrix", "[random_matrix]")
{
   SECTION("row sums of result equal one")
   {
      const double tol = 1e-12;

      const int n_rows = 14;
      const int n_cols = 20;
      Eigen::MatrixXd A(Eigen::MatrixXd::Random(n_rows, n_cols));

      const Eigen::VectorXd ones_vector(Eigen::VectorXd::Ones(n_rows));

      const Eigen::VectorXd initial_sums = A.rowwise().sum();
      const double initial_diff = (initial_sums - ones_vector).cwiseAbs().maxCoeff();

      REQUIRE(initial_diff > tol);

      make_right_stochastic(A);

      const Eigen::VectorXd final_sums = A.rowwise().sum();
      const double final_diff = (final_sums - ones_vector).cwiseAbs().maxCoeff();

      CHECK(final_diff < tol);
   }
}

TEST_CASE("test generating random left stochastic matrix", "[random_matrix]")
{
   const int seed = 0;
   std::mt19937 generator(seed);

   SECTION("columns sums of generated matrix equal one")
   {
      const double tol = 1e-12;

      const int n_rows = 16;
      const int n_cols = 16;
      Eigen::MatrixXd A(Eigen::MatrixXd::Zero(n_rows, n_cols));

      random_left_stochastic_matrix(A, generator);

      const Eigen::RowVectorXd ones_vector(Eigen::RowVectorXd::Ones(n_cols));
      const Eigen::RowVectorXd sums = A.colwise().sum();
      const double diff = (sums - ones_vector).cwiseAbs().maxCoeff();

      CHECK(diff < tol);
   }
}

TEST_CASE("test generating random right stochastic matrix", "[random_matrix]")
{
   const int seed = 0;
   std::mt19937 generator(seed);

   SECTION("row sums of generated matrix equal one")
   {
      const double tol = 1e-12;

      const int n_rows = 45;
      const int n_cols = 30;
      Eigen::MatrixXd A(Eigen::MatrixXd::Zero(n_rows, n_cols));

      random_right_stochastic_matrix(A, generator);

      const Eigen::VectorXd ones_vector(Eigen::VectorXd::Ones(n_rows));
      const Eigen::VectorXd sums = A.rowwise().sum();
      const double diff = (sums - ones_vector).cwiseAbs().maxCoeff();

      CHECK(diff < tol);
   }
}
