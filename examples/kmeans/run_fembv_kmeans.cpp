/**
 * @file run_fembv_kmeans.cpp
 * @brief example demonstrating FEM-BV-k-means algorithm
 */

#include "fembv_kmeans.hpp"
#include "multivariate_normal.hpp"

#include <Eigen/Core>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace fembvpp;

struct KMeans_options {
   int n_components{2};
   int n_samples{6000};
   int n_switches{2};
   int n_init{100};
   double max_tv_norm{2};
   std::string data_output_file{""};
   std::string true_affiliations_output_file{""};
   std::string parameters_output_file{""};
   std::string affiliations_output_file{""};
   bool verbose{false};
};

void print_usage()
{
   std::cout <<
      "Usage: run_fembv_kmeans [OPTION]\n\n"
      "Run FEM-BV-k-means on example data.\n\n"
      "Example: run_fembv_kmeans -l 6000 -s 2\n\n"
      "Options:\n"
      "  -a, --affiliations-output-file=FILE        name of file to write FEM-BV\n"
      "                                             affiliations to\n"
      "  -c, --max-tv-norm=MAX_TV_NORM              TV norm upper bound\n"
      "  -d, --data-output-file=FILE                name of file to\n"
      "                                             write data to\n"
      "  -h, --help                                 print this help message\n"
      "  -i, --n-init=N_INIT                        number of initializations\n"
      "  -k, --n-components=N_COMPONENTS            number of FEM-BV components\n"
      "  -l, --n-samples=N_SAMPLES                  length of time-series\n"
      "  -p, --parameters-output-file=FILE          name of file to write FEM-BV\n"
      "                                             parameters to\n"
      "  -s, --n-switches=N_SWITCHES                number of switches\n"
      "  -t, --true-affiliations-output-file=FILE   name of file to\n"
      "                                             write true affiliations to\n"
      "  -v, --verbose                              produce verbose output\n"
      << std::endl;
}

bool starts_with(const std::string& option, const std::string& prefix)
{
   return !option.compare(0, prefix.size(), prefix);
}

std::string get_option_value(const std::string& option,
                             const std::string& sep = "=")
{
   std::string value{""};
   const auto prefix_end = option.find(sep);

   if (prefix_end != std::string::npos) {
      value = option.substr(prefix_end + 1);
   }

   return value;
}

KMeans_options parse_cmd_line_args(int argc, const char* argv[])
{
   KMeans_options options;

   int i = 1;
   while (i < argc) {
      const std::string opt(argv[i++]);

      if (opt == "-h" || opt == "--help") {
         print_usage();
         exit(EXIT_SUCCESS);
      }

      if (opt == "-c") {
         if (i == argc) {
            throw std::runtime_error(
               "'-c' given but norm bound not provided");
         }
         const std::string max_tv_norm(argv[i++]);
         if (starts_with(max_tv_norm, "-")) {
            throw std::runtime_error(
               "-c' given but norm bound not provided");
         }
         options.max_tv_norm = std::stod(max_tv_norm);
         continue;
      }

      if (starts_with(opt, "--max-tv-norm=")) {
         const std::string max_tv_norm = get_option_value(opt);
         if (max_tv_norm.empty()) {
            throw std::runtime_error(
               "--max_tv_norm=' given but norm bound not provided");
         }
         options.max_tv_norm = std::stod(max_tv_norm);
         continue;
      }

      if (opt == "-i") {
         if (i == argc) {
            throw std::runtime_error(
               "'-i' given but number of repetitions not provided");
         }
         const std::string n_init(argv[i++]);
         if (starts_with(n_init, "-")) {
            throw std::runtime_error(
               "-i' given but number of repetitions not provided");
         }
         options.n_init = std::stoi(n_init);
         continue;
      }

      if (starts_with(opt, "--n-init=")) {
         const std::string n_init = get_option_value(opt);
         if (n_init.empty()) {
            throw std::runtime_error(
               "--n-init=' given but number of repetitions not provided");
         }
         options.n_init = std::stoi(n_init);
         continue;
      }

      if (opt == "-k") {
         if (i == argc) {
            throw std::runtime_error(
               "'-k' given but number of components not provided");
         }
         const std::string n_components(argv[i++]);
         if (starts_with(n_components, "-")) {
            throw std::runtime_error(
               "-k' given but number of components not provided");
         }
         options.n_components = std::stoi(n_components);
         continue;
      }

      if (starts_with(opt, "--n-components=")) {
         const std::string n_components = get_option_value(opt);
         if (n_components.empty()) {
            throw std::runtime_error(
               "--n-components=' given but number of components not provided");
         }
         options.n_components = std::stoi(n_components);
         continue;
      }

      if (opt == "-s") {
         if (i == argc) {
            throw std::runtime_error(
               "'-s' given but number of switches not provided");
         }
         const std::string n_switches(argv[i++]);
         if (starts_with(n_switches, "-")) {
            throw std::runtime_error(
               "-s' given but number of switches not provided");
         }
         options.n_switches = std::stoi(n_switches);
         continue;
      }

      if (starts_with(opt, "--n-switches=")) {
         const std::string n_switches = get_option_value(opt);
         if (n_switches.empty()) {
            throw std::runtime_error(
               "--n-switches=' given but number of switches not provided");
         }
         options.n_switches = std::stoi(n_switches);
         continue;
      }

      if (opt == "-l") {
         if (i == argc) {
            throw std::runtime_error(
               "'-l' given but number of samples not provided");
         }
         const std::string n_samples(argv[i++]);
         if (starts_with(n_samples, "-")) {
            throw std::runtime_error(
               "-l' given but number of samples not provided");
         }
         options.n_samples = std::stoi(n_samples);
         continue;
      }

      if (starts_with(opt, "--n-samples=")) {
         const std::string n_samples = get_option_value(opt);
         if (n_samples.empty()) {
            throw std::runtime_error(
               "--n-samples=' given but number of samples not provided");
         }
         options.n_samples = std::stoi(n_samples);
         continue;
      }

      if (opt == "-d") {
         if (i == argc) {
            throw std::runtime_error(
               "'-d' given but no output file name provided");
         }
         const std::string filename(argv[i++]);
         if (starts_with(filename, "-") && filename != "-") {
            throw std::runtime_error(
               "'-d' given but no output file name provided");
         }
         options.data_output_file = filename;
         continue;
      }

      if (starts_with(opt, "--data-output-file=")) {
         const std::string filename = get_option_value(opt);
         if (filename.empty()) {
            throw std::runtime_error(
               "'--data-output-file=' given but no output file name provided");
         }
         options.data_output_file = filename;
         continue;
      }

      if (opt == "-t") {
         if (i == argc) {
            throw std::runtime_error(
               "'-t' given but no output file name provided");
         }
         const std::string filename(argv[i++]);
         if (starts_with(filename, "-") && filename != "-") {
            throw std::runtime_error(
               "'-t' given but no output file name provided");
         }
         options.true_affiliations_output_file = filename;
         continue;
      }

      if (starts_with(opt, "--true-affiliations-output-file=")) {
         const std::string filename = get_option_value(opt);
         if (filename.empty()) {
            throw std::runtime_error(
               "'--true-affiliations-output-file=' given but "
               "no output file name provided");
         }
         options.true_affiliations_output_file = filename;
         continue;
      }

      if (opt == "-a") {
         if (i == argc) {
            throw std::runtime_error(
               "'-a' given but no output file name provided");
         }
         const std::string filename(argv[i++]);
         if (starts_with(filename, "-") && filename != "-") {
            throw std::runtime_error(
               "'-a' given but no output file name provided");
         }
         options.affiliations_output_file = filename;
         continue;
      }

      if (starts_with(opt, "--affiliations-output-file=")) {
         const std::string filename = get_option_value(opt);
         if (filename.empty()) {
            throw std::runtime_error(
               "'--affiliations-output-file=' given but "
               "no output file name provided");
         }
         options.affiliations_output_file = filename;
         continue;
      }

      if (opt == "-p") {
         if (i == argc) {
            throw std::runtime_error(
               "'-p' given but no output file name provided");
         }
         const std::string filename(argv[i++]);
         if (starts_with(filename, "-") && filename != "-") {
            throw std::runtime_error(
               "'-p' given but no output file name provided");
         }
         options.parameters_output_file = filename;
         continue;
      }

      if (starts_with(opt, "--parameters-output-file=")) {
         const std::string filename = get_option_value(opt);
         if (filename.empty()) {
            throw std::runtime_error(
               "'--parameters-output-file=' given but "
               "no output file name provided");
         }
         options.parameters_output_file = filename;
         continue;
      }

      if (opt == "-v" || opt == "--verbose") {
         options.verbose = true;
         continue;
      }

      throw std::runtime_error(
         "unrecognized command line argument '" + opt + "'");
   }

   return options;
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

   Gamma = Eigen::MatrixXd::Zero(n_components, n_samples);

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

   std::vector<Multivariate_normal_distribution> distributions;
   for (int i = 0; i < n_clusters; ++i) {
      distributions.push_back(Multivariate_normal_distribution(
         means[i], covariances[i]));
   }

   for (int t = 0; t < n_samples; ++t) {
      Gamma(cluster_assignments[t], t) = 1;
      X.col(t) = distributions[cluster_assignments[t]](generator);
   }
}

template <class Generator>
std::tuple<bool, Eigen::MatrixXd, Eigen::MatrixXd>
run_fembv_kmeans(const Eigen::MatrixXd& X, int n_components,
                 double max_tv_norm, int n_init, bool verbose,
                 Generator& generator)
{
   const int n_features = X.rows();
   const int n_samples = X.cols();

   double min_cost = -1;
   Eigen::MatrixXd best_parameters(
      Eigen::MatrixXd::Zero(n_features, n_components));
   Eigen::MatrixXd best_affiliations(
      Eigen::MatrixXd::Zero(n_components, n_samples));
   bool has_best_fit = false;
   for (int i = 0; i < n_init; ++i) {
      FEMBVKMeans model(n_components, max_tv_norm);

      if (verbose) {
         model.set_verbosity(1);
      }

      const bool success = model.fit(X, generator);
      if (success) {
         has_best_fit = true;
      }

      const auto cost = model.get_cost();

      if (success && (cost < min_cost || i == 0)) {
         best_parameters = model.get_parameters();
         best_affiliations = model.get_affiliations();
      }
   }

   return std::make_tuple(has_best_fit, best_parameters, best_affiliations);
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

   ofs << header;
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
      ofs << sstr.str();
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

   write_header_line(ofs, fields);
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

void write_fembv_parameters(
   const std::string& output_file, const Eigen::MatrixXd& Theta)
{
   const int n_fields = Theta.rows();
   std::vector<std::string> fields(n_fields);
   for (int i = 0; i < n_fields; ++i) {
      fields[i] = "x" + std::to_string(i);
   }
   write_csv(output_file, Theta, fields);
}

int main(int argc, const char* argv[])
{
   const int seed = 0;
   std::mt19937 generator(seed);

   const int n_dims = 2;
   const int n_clusters = 3;

   Eigen::VectorXd mu_1(2);
   Eigen::VectorXd mu_2(2);
   Eigen::VectorXd mu_3(2);
   Eigen::MatrixXd sigma_1(2, 2);
   Eigen::MatrixXd sigma_2(2, 2);
   Eigen::MatrixXd sigma_3(2, 2);

   mu_1 << 0, 0.5;
   sigma_1 << 0.001, 0,
      0, 0.02;

   mu_2 << 0, -0.5;
   sigma_2 << 0.001, 0,
      0, 0.02;

   mu_3 << 0.25, 0;
   sigma_3 << 0.002, 0,
      0, 0.3;

   std::vector<Eigen::VectorXd> means;
   means.push_back(mu_1);
   means.push_back(mu_2);
   means.push_back(mu_3);

   std::vector<Eigen::MatrixXd> covariances;
   covariances.push_back(sigma_1);
   covariances.push_back(sigma_2);
   covariances.push_back(sigma_3);

   const auto options = parse_cmd_line_args(argc, argv);

   if (options.n_switches < 0) {
      std::cerr << "Error: number of switches must be non-negative\n";
      exit(EXIT_FAILURE);
   }
   
   if (options.n_samples < 0) {
      std::cerr << "Error: number of samples must be non-negative\n";
      exit(EXIT_FAILURE);
   }

   if (options.n_init < 1) {
      std::cerr << "Error: number of repetitions must be at least one\n";
      exit(EXIT_FAILURE);
   }

   Eigen::MatrixXd data(n_dims, options.n_samples);
   Eigen::MatrixXd true_affiliations(n_clusters, options.n_samples);

   generate_data(options.n_switches, n_clusters, means, covariances,
                 data, true_affiliations, generator);

   if (!options.data_output_file.empty()) {
      write_data(options.data_output_file, data);
   }

   if (!options.true_affiliations_output_file.empty()) {
      write_affiliations(options.true_affiliations_output_file,
                         true_affiliations);
   }

   const auto fembv_result = run_fembv_kmeans(
      data, options.n_components, options.max_tv_norm,
      options.n_init, options.verbose, generator);

   const bool success = std::get<0>(fembv_result);
   if (!success) {
      std::cerr << "Error: failed to fit FEM-BV model\n";
      exit(EXIT_FAILURE);
   }

   const Eigen::MatrixXd parameters(std::get<1>(fembv_result));
   const Eigen::MatrixXd affiliations(std::get<2>(fembv_result));

   if (!options.parameters_output_file.empty()) {
      write_fembv_parameters(options.parameters_output_file, parameters);
   }

   if (!options.affiliations_output_file.empty()) {
      write_affiliations(options.affiliations_output_file, affiliations);
   }

   return 0;
}
