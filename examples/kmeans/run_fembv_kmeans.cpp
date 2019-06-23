/**
 * @file run_fembv_kmeans.cpp
 * @brief example demonstrating FEM-BV-k-means algorithm
 */

#include "multivariate_normal.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace fembvpp;

struct KMeans_options {
   int n_switches{3};
   int n_clusters{3};
   std::string data_output_file{""};
   std::string true_affiliations_output_file{""};
   int verbosity{0};
};

void print_usage()
{
   std::cout <<
      "Usage: run_fembv_kmeans [OPTION]\n\n"
      "Run FEM-BV-k-means on example data.\n\n"
      "Example: run_fembv_kmeans -c 3 -s 3\n\n"
      "Options:\n"
      "  -c, --n-clusters    number of clusters\n"
      "  -h, --help          print this help message\n"
      "  -s, --n-switches    number of switches\n"
}

template <class Generator>
void generate_data(int n_switches, int n_clusters,
                   const std::vector<Eigen::VectorXd>& means,
                   const std::vector<Eigen::MatrixXd>& covariances,
                   Eigen::MatrixXd& X,
                   Eigen::MatrixXd& Gamma,
                   Generator& generator)
{
   const int n_samples = X.cols();
   const int n_components = Gamma.rows();
   const int run_length = n_samples / (n_switches + 1);

   Gamma = Eigen::MatriXd::Zero(n_components, n_samples);

   int cluster = 0;
   std::vector<int> cluster_assignments;
   for (int i = 0; i < n_switches; ++i) {
      for (int t = i * run_length; t < (i + 1) * run_length; ++t) {
         cluster_assignments.push_back(cluster);
      }
      cluster = (cluster + 1) % n_clusters;
   }
   for (int t = n_switches * run_length;  t < n_samples; ++t) {
      cluster_assignments.push_back(cluster);
   }

   std::vector<Multivariate_normal_distribution> distributions(n_clusters);
   for (int i = 0; i < n_clusters; ++i) {
      distributions[i] = Multivariate_normal_distributions(
         means[i], covariances[i]);
   }

   for (int t = 0; t < n_samples; ++t) {
      Gamma(cluster_assignments[t], t) = 1;
      X.col(t) = distributions[cluster_assignments[t]](generator);
   }
}

void write_header_line(
   std::ofstream& ofs, const std::vector<std::string>& fields)
{
   const std::size_t n_fields = fields.size();

   std::string header = "# ";
   for (std::size_t f = 0; f < n_fields; ++f) {
      header = header + fields[f];
      if (f != n_fields - 1) {
         header = header + ',';
      }
   }
   header = header + '\n';

   ofs.write(header);
}

void write_data_lines(std::ofstream& ofs, const Eigen::MatrixXd& data)
{
   const int n_fields = data.rows();
   const int n_samples = data.cols();

   for (int t = 0; t < n_samples; ++t) {
      std::stringstream sstr;
      for (int i = 0; i < n_fields; ++i) {
         sstr << std::scientific << std::setw(18) << data(i, t);
         if (i != n_fields - 1) {
            sstr << ',';
         }
      }
      sstr << '\n';
      ofs.write(sstr.str());
   }
}

void write_csv(const std::string& output_file, const Eigen::MatrixXd& data,
               const std::vector<std::string>& fields)
{
   const int n_fields = fields.size();
   if (data.rows() != n_fields) {
      throw std::runtime_error(
         "number of data rows does not match number of fields");
   }

   std::ofstream ofs(output_file);

   if (!ofs.is_open()) {
      throw std::runtime_error(
         "failed to open datafile for writing");
   }

   write_header_lines(ofs, fields);
   write_data_lines(ofs, data);
}

void write_data(const std::string& output_file, const Eigen::MatrixXd& data)
{
   const int n_fields = data.rows();
   std::vector<std::string> fields(n_fields);
   for (int i = 0; i < n_fields; ++i ) {
      fields[i] = "x" + std::to_string(i);
   }
   write_csv(output_file, data, fields);
}

void write_affiliations(
   const std::string& output_file, const Eigen::MatrixXd& Gamma)
{
   const int n_fields = Gamma.rows();
   std::vector<std::string> fields(n_fields);
   for (int i = 0; i < n_fields; ++i) {
      fields[i] = "Gamma" + std::to_string(i);
   }
   write_csv(output_file, Gamma, fields);
}

int main(int argc, const char* arg[])
{
   const int seed = 0;
   std::mt19937 generator(seed);

   const auto options = parse_cmd_line_args(argc, argv);
}
