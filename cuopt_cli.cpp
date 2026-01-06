/*
 * Minimal CPU-only CLI launcher for Dual Simplex
 * This replaces the previous multi-solver CLI and avoids any GPU/RMM initialization.
 */

#include <mps_parser/parser.hpp>
#include <math_optimization/solution_reader.hpp>
#include <utilities/high_res_timer.hpp>

#include <dual_simplex/user_problem.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>

#include <dual_simplex/highs_presolve.hpp>

#include <argparse/argparse.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <cmath>

static cuopt::linear_programming::dual_simplex::user_problem_t<int, double>
mps_to_dual_simplex_user_problem(const cuopt::mps_parser::mps_data_model_t<int, double>& mps) {
    cuopt::linear_programming::dual_simplex::user_problem_t<int, double> up;

    const auto n = mps.get_n_variables();
    const auto m = mps.get_n_constraints();
    const auto nnz = mps.get_constraint_matrix_values().size();

    up.num_rows = m;
    up.num_cols = n;
    up.objective = mps.get_objective_coefficients();

    // Build CSR -> convert to CSC using csr_matrix_t utility
    cuopt::linear_programming::dual_simplex::csr_matrix_t<int, double> csr_A;
    csr_A.m = m;
    csr_A.n = n;
    csr_A.nz_max = static_cast<int>(nnz);
    csr_A.x = mps.get_constraint_matrix_values();
    csr_A.j = mps.get_constraint_matrix_indices();
    csr_A.row_start = mps.get_constraint_matrix_offsets();

    csr_A.to_compressed_col(up.A);

    // Bounds / senses
    const auto& lower = mps.get_constraint_lower_bounds();
    const auto& upper = mps.get_constraint_upper_bounds();
    up.rhs.resize(m);
    up.row_sense.resize(m);
    up.range_rows.clear();
    up.range_value.clear();

    for (int i = 0; i < m; ++i) {
        double lb = (i < (int)lower.size()) ? lower[i] : -std::numeric_limits<double>::infinity();
        double ub = (i < (int)upper.size()) ? upper[i] : std::numeric_limits<double>::infinity();
        if (lb == ub) {
            up.row_sense[i] = 'E';
            up.rhs[i] = lb;
        } else if (ub == std::numeric_limits<double>::infinity()) {
            up.row_sense[i] = 'G';
            up.rhs[i] = lb;
        } else if (lb == -std::numeric_limits<double>::infinity()) {
            up.row_sense[i] = 'L';
            up.rhs[i] = ub;
        } else {
            up.row_sense[i] = 'E';
            up.rhs[i] = lb;
            up.range_rows.push_back(i);
            up.range_value.push_back(ub - lb);
        }
    }
    up.num_range_rows = static_cast<int>(up.range_rows.size());

    up.lower = mps.get_variable_lower_bounds();
    up.upper = mps.get_variable_upper_bounds();

    // names and metadata
    up.problem_name = mps.get_problem_name();
    up.row_names = mps.get_row_names();
    up.col_names = mps.get_variable_names();
    up.obj_constant = mps.get_objective_offset();
    up.obj_scale = mps.get_objective_scaling_factor();

    // variable types
    const auto& vtypes = mps.get_variable_types();
    up.var_types.resize(n);
    for (int j = 0; j < n; ++j) {
        if (j < (int)vtypes.size() && vtypes[j] == 'I')
            up.var_types[j] = cuopt::linear_programming::dual_simplex::variable_type_t::INTEGER;
        else
            up.var_types[j] = cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS;
    }

    return up;
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("cuopt_cli_dual_simplex", "cuOpt minimal dual-simplex CLI");
    program.add_argument("filename").help("input mps file").nargs(1).required();
    program.add_argument("--initial-solution").default_value(std::string(""))
        .help("path to an initial .sol file");
    program.add_argument("--profile").default_value(false).implicit_value(true)
        .help("enable profiling");
    program.add_argument("--highs-presolve").default_value(false).implicit_value(true)
        .help("use HiGHS presolve instead of built-in presolve");
    program.add_argument("--gpu").default_value(false).implicit_value(true)
        .help("enable GPU acceleration");
    program.add_argument("--pinv-slices").default_value(1).scan<'i', int>()
        .help("number of slices for parallel INVERSE computation");
    program.add_argument("--max-iters").default_value(std::numeric_limits<int>::max()).scan<'i', int>()
        .help("set maximum iteration limit");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const std::string file_name = program.get<std::string>("filename");
    const bool profile_enabled = program.get<bool>("--profile");
    const std::string initial_solution_file = program.get<std::string>("--initial-solution");
    const bool use_highs_presolve = program.get<bool>("--highs-presolve");
    const bool use_gpu = program.get<bool>("--gpu");
    const int pinv_slices = program.get<int>("--pinv-slices");
    const int max_iters = program.get<int>("--max-iters");

    cuopt::linear_programming::dual_simplex::user_problem_t<int, double> user_problem;
    if (use_highs_presolve) {
        cuopt::linear_programming::dual_simplex::highs_presolve_t<int, double> presolver;
        presolver.presolve_from_file(file_name, user_problem);
    } else {
        // Parse MPS into host-side model
        cuopt::mps_parser::mps_data_model_t<int, double> mps;

        try {
            mps = cuopt::mps_parser::parse_mps<int, double>(file_name, /*strict=*/false);
        } catch (const std::exception& e) {
            std::cerr << "MPS parse error: " << e.what() << std::endl;
            return 2;
        }

        // Convert to dual-simplex user_problem (host-only)
        user_problem = mps_to_dual_simplex_user_problem(mps);

        // If an initial solution file is provided, try to read primal values
        if (!initial_solution_file.empty()) {
            try {
                auto values = cuopt::linear_programming::solution_reader_t::get_variable_values_from_sol_file(
                    initial_solution_file, mps.get_variable_names());
                if (!values.empty()) { /* initial values not currently applied to solver here */ }
            } catch (...) {
                // ignore read failures — not critical
            }
        }
    }

    // Prepare solver settings and solution container
    cuopt::linear_programming::dual_simplex::simplex_solver_settings_t<int, double> settings;
    settings.profile = profile_enabled;
    settings.gpu = use_gpu;
    settings.pinv_slices = pinv_slices;
    settings.iteration_limit = max_iters;

    cuopt::linear_programming::dual_simplex::lp_solution_t<int, double> solution(
        user_problem.num_rows, user_problem.num_cols);

    // Run solver and time it
    settings.timer.start("Dual Simplex Solve");
    auto status = cuopt::linear_programming::dual_simplex::solve_linear_program(user_problem, settings, solution);
    settings.timer.stop("Dual Simplex Solve");
    settings.timer.display(std::cout, "Dual Simplex Solve");

    std::cout << "Status: ";
    switch (status) {
        case cuopt::linear_programming::dual_simplex::lp_status_t::OPTIMAL:
            std::cout << "OPTIMAL\n";
            break;
        case cuopt::linear_programming::dual_simplex::lp_status_t::INFEASIBLE:
            std::cout << "INFEASIBLE\n";
            break;
        case cuopt::linear_programming::dual_simplex::lp_status_t::UNBOUNDED:
            std::cout << "UNBOUNDED\n";
            break;
        case cuopt::linear_programming::dual_simplex::lp_status_t::TIME_LIMIT:
            std::cout << "TIME_LIMIT\n";
            break;
        case cuopt::linear_programming::dual_simplex::lp_status_t::ITERATION_LIMIT:
            std::cout << "ITERATION_LIMIT\n";
            break;
        case cuopt::linear_programming::dual_simplex::lp_status_t::NUMERICAL_ISSUES:
            std::cout << "NUMERICAL_ISSUES\n";
            break;
        default:
            std::cout << "OTHER\n";
            break;
    }

    if (!std::isnan(solution.user_objective)) {
        std::cout << "Objective (user): " << solution.user_objective << "\n";
    }
    std::cout << "Iterations: " << solution.iterations << "\n";

    if (profile_enabled) {
        std::cout << "=== Profile Summary ===\n";
        settings.timer.display(std::cout);
    }

    return 0;
}
