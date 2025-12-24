/*
 * Minimal Dual Simplex Solver CLI
 * Reads MPS format LP files and solves them using the dual-simplex algorithm
 * with presolve and postsolve.
 */

#include <iostream>
#include <string>
#include <stdexcept>

#include <mps_parser/parser.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/user_problem.hpp>
#include <utilities/logger.hpp>

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mps_file_path> [--no-presolve] [--log-file <path>]" << std::endl;
    std::cerr << "\nOptions:" << std::endl;
    std::cerr << "  --no-presolve      Disable presolve/postsolve" << std::endl;
    std::cerr << "  --log-file <path>  Write log to file" << std::endl;
    std::cerr << "  --help             Show this help message" << std::endl;
    return 1;
  }

  std::string mps_file;
  bool enable_presolve = true;
  std::string log_file = "";

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout << "Minimal Dual Simplex Solver\n";
      std::cout << "Usage: " << argv[0] << " <mps_file_path> [OPTIONS]\n\n";
      std::cout << "Options:\n";
      std::cout << "  --no-presolve      Disable presolve/postsolve\n";
      std::cout << "  --log-file <path>  Write log to file\n";
      std::cout << "  --help             Show this help message\n";
      return 0;
    } else if (arg == "--no-presolve") {
      enable_presolve = false;
    } else if (arg == "--log-file" && i + 1 < argc) {
      log_file = argv[++i];
    } else if (mps_file.empty() && arg[0] != '-') {
      mps_file = arg;
    }
  }

  if (mps_file.empty()) {
    std::cerr << "Error: No MPS file specified\n";
    return 1;
  }

  try {
    // Initialize logger
    cuopt::init_logger_t logger(log_file, true);
    CUOPT_LOG_INFO("Minimal Dual Simplex Solver");
    CUOPT_LOG_INFO("MPS file: %s", mps_file.c_str());
    CUOPT_LOG_INFO("Presolve: %s", enable_presolve ? "enabled" : "disabled");

    // Parse MPS file
    CUOPT_LOG_INFO("Parsing MPS file...");
    auto mps_data = cuopt::mps_parser::parse_mps<int, double>(mps_file, false);
    
    CUOPT_LOG_INFO("Problem size: %zu variables, %zu constraints, %zu non-zeros",
                   mps_data.get_objective_coefficients().size(),
                   mps_data.get_constraint_bounds().size(),
                   mps_data.get_constraint_matrix_values().size());

    // Convert to dual_simplex user_problem format
    using i_t = int;
    using f_t = double;
    
    cuopt::linear_programming::dual_simplex::user_problem_t<i_t, f_t> problem(nullptr);
    
    // Set problem data
    problem.num_variables = mps_data.get_objective_coefficients().size();
    problem.num_constraints = mps_data.get_constraint_bounds().size();
    problem.maximize = mps_data.get_sense();
    
    problem.A_values = mps_data.get_constraint_matrix_values();
    problem.A_indices = mps_data.get_constraint_matrix_indices();
    problem.A_offsets = mps_data.get_constraint_matrix_offsets();
    problem.objective = mps_data.get_objective_coefficients();
    problem.rhs = mps_data.get_constraint_bounds();
    problem.lower_bounds = mps_data.get_variable_lower_bounds();
    problem.upper_bounds = mps_data.get_variable_upper_bounds();

    // Setup solver settings
    cuopt::linear_programming::dual_simplex::simplex_solver_settings_t<i_t, f_t> settings;
    settings.presolve = enable_presolve;
    settings.log_to_console = true;
    settings.log_file = log_file;

    // Solve
    CUOPT_LOG_INFO("Starting dual simplex solve...");
    auto solution = cuopt::linear_programming::dual_simplex::solve(problem, settings);

    // Output results
    CUOPT_LOG_INFO("Solve completed!");
    CUOPT_LOG_INFO("Status: %d", static_cast<int>(solution.status));
    CUOPT_LOG_INFO("Objective value: %.10e", solution.objective_value);
    CUOPT_LOG_INFO("Iterations: %d", solution.num_iterations);
    
    if (solution.x.size() > 0) {
      CUOPT_LOG_INFO("Solution vector size: %zu", solution.x.size());
      std::cout << "\nObjective: " << solution.objective_value << "\n";
      std::cout << "Status: " << static_cast<int>(solution.status) << "\n";
      std::cout << "Iterations: " << solution.num_iterations << "\n";
    }

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
