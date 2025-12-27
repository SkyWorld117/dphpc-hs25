#include <dual_simplex/phase2_dual.hpp>

namespace cuopt::linear_programming::dual_simplex {

namespace phase2 {

template <typename i_t, typename f_t>
class phase2_timers_t {
 public:
  phase2_timers_t(bool should_time)
    : record_time(should_time),
      start_time(0),
      bfrt_time(0),
      pricing_time(0),
      btran_time(0),
      ftran_time(0),
      flip_time(0),
      delta_z_time(0),
      se_norms_time(0),
      se_entering_time(0),
      lu_update_time(0),
      perturb_time(0),
      vector_time(0),
      objective_time(0),
      update_infeasibility_time(0)
  {
  }

  void start_timer()
  {
    if (!record_time) { return; }
    start_time = tic();
  }

  f_t stop_timer()
  {
    if (!record_time) { return 0.0; }
    return toc(start_time);
  }

  void print_timers(const simplex_solver_settings_t<i_t, f_t>& settings) const
  {
    if (!record_time) { return; }
    const f_t total_time = bfrt_time + pricing_time + btran_time + ftran_time + flip_time +
                           delta_z_time + lu_update_time + se_norms_time + se_entering_time +
                           perturb_time + vector_time + objective_time + update_infeasibility_time;
    // clang-format off
    settings.log.printf("BFRT time       %.2fs %4.1f%\n", bfrt_time, 100.0 * bfrt_time / total_time);
    settings.log.printf("Pricing time    %.2fs %4.1f%\n", pricing_time, 100.0 * pricing_time / total_time);
    settings.log.printf("BTran time      %.2fs %4.1f%\n", btran_time, 100.0 * btran_time / total_time);
    settings.log.printf("FTran time      %.2fs %4.1f%\n", ftran_time, 100.0 * ftran_time / total_time);
    settings.log.printf("Flip time       %.2fs %4.1f%\n", flip_time, 100.0 * flip_time / total_time);
    settings.log.printf("Delta_z time    %.2fs %4.1f%\n", delta_z_time, 100.0 * delta_z_time / total_time);
    settings.log.printf("LU update time  %.2fs %4.1f%\n", lu_update_time, 100.0 * lu_update_time / total_time);
    settings.log.printf("SE norms time   %.2fs %4.1f%\n", se_norms_time, 100.0 * se_norms_time / total_time);
    settings.log.printf("SE enter time   %.2fs %4.1f%\n", se_entering_time, 100.0 * se_entering_time / total_time);
    settings.log.printf("Perturb time    %.2fs %4.1f%\n", perturb_time, 100.0 * perturb_time / total_time);
    settings.log.printf("Vector time     %.2fs %4.1f%\n", vector_time, 100.0 * vector_time / total_time);
    settings.log.printf("Objective time  %.2fs %4.1f%\n", objective_time, 100.0 * objective_time / total_time);
    settings.log.printf("Inf update time %.2fs %4.1f%\n", update_infeasibility_time, 100.0 * update_infeasibility_time / total_time);
    settings.log.printf("Sum             %.2fs\n", total_time);
    // clang-format on
  }
  f_t bfrt_time;
  f_t pricing_time;
  f_t btran_time;
  f_t ftran_time;
  f_t flip_time;
  f_t delta_z_time;
  f_t se_norms_time;
  f_t se_entering_time;
  f_t lu_update_time;
  f_t perturb_time;
  f_t vector_time;
  f_t objective_time;
  f_t update_infeasibility_time;

 private:
  f_t start_time;
  bool record_time;
};

}

template <typename i_t, typename f_t>
dual::status_t dual_phase2_cu(
    i_t phase,
    i_t slack_basis,
    f_t start_time,
    const lp_problem_t<i_t, f_t>& lp,
    const simplex_solver_settings_t<i_t, f_t>& settings,
    std::vector<variable_status_t>& vstatus,
    lp_solution_t<i_t, f_t>& sol,
    i_t& iter,
    std::vector<f_t>& delta_y_steepest_edge)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  assert(m <= n);
  assert(vstatus.size() == n);
  assert(lp.A.m == m);
  assert(lp.A.n == n);
  assert(lp.objective.size() == n);
  assert(lp.lower.size() == n);
  assert(lp.upper.size() == n);
  assert(lp.rhs.size() == m);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;
  std::vector<i_t> superbasic_list;

  std::vector<f_t>& x = sol.x;
  std::vector<f_t>& y = sol.y;
  std::vector<f_t>& z = sol.z;

  dual::status_t status = dual::status_t::UNSET;

  // Perturbed objective
  std::vector<f_t> objective = lp.objective;

  settings.log.printf("Dual Simplex Phase %d\n", phase);
  std::vector<variable_status_t> vstatus_old = vstatus;
  std::vector<f_t> z_old                     = z;

  phase2::bound_info(lp, settings);
  get_basis_from_vstatus(m, vstatus, basic_list, nonbasic_list, superbasic_list);
  assert(superbasic_list.size() == 0);
  assert(nonbasic_list.size() == n - m);

  // Compute L*U = A(p, basic_list)
  csc_matrix_t<i_t, f_t> L(m, m, 1);
  csc_matrix_t<i_t, f_t> U(m, m, 1);
  std::vector<i_t> pinv(m);
  std::vector<i_t> p;
  std::vector<i_t> q;
  std::vector<i_t> deficient;
  std::vector<i_t> slacks_needed;

  if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
      -1) {
    settings.log.debug("Initial factorization failed\n");
    basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
    if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
        -1) {
      return dual::status_t::NUMERICAL;
    }
    settings.log.printf("Basis repaired\n");
  }
  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
  assert(q.size() == m);
  reorder_basic_list(q, basic_list);
  basis_update_mpf_t<i_t, f_t> ft(L, U, p, settings.refactor_frequency);

  std::vector<f_t> c_basic(m);
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    c_basic[k]  = objective[j];
  }

  // Solve B'*y = cB
  ft.b_transpose_solve(c_basic, y);
  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
  constexpr bool print_norms = false;
  if constexpr (print_norms) {
    settings.log.printf(
      "|| y || %e || cB || %e\n", vector_norm_inf<i_t, f_t>(y), vector_norm_inf<i_t, f_t>(c_basic));
  }

  phase2::compute_reduced_costs(objective, lp.A, y, basic_list, nonbasic_list, z);
  if constexpr (print_norms) { settings.log.printf("|| z || %e\n", vector_norm_inf<i_t, f_t>(z)); }

#ifdef COMPUTE_DUAL_RESIDUAL
  std::vector<f_t> dual_res1;
  compute_dual_residual(lp.A, objective, y, z, dual_res1);
  f_t dual_res_norm = vector_norm_inf<i_t, f_t>(dual_res1);
  if (dual_res_norm > settings.tight_tol) {
    settings.log.printf("|| A'*y + z - c || %e\n", dual_res_norm);
  }
  assert(dual_res_norm < 1e-3);
#endif

  phase2::set_primal_variables_on_bounds(lp, settings, z, vstatus, x);

#ifdef PRINT_VSTATUS_CHANGES
  i_t num_vstatus_changes;
  i_t num_z_changes;
  phase2::vstatus_changes(vstatus, vstatus_old, z, z_old, num_vstatus_changes, num_z_changes);
  settings.log.printf("Number of vstatus changes %d\n", num_vstatus_changes);
  settings.log.printf("Number of z changes %d\n", num_z_changes);
#endif

  const f_t init_dual_inf =
    phase2::dual_infeasibility(lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
  if (init_dual_inf > settings.dual_tol) {
    settings.log.printf("Initial dual infeasibility %e\n", init_dual_inf);
  }

  for (i_t j = 0; j < n; ++j) {
    if (lp.lower[j] == -inf && lp.upper[j] == inf && vstatus[j] != variable_status_t::BASIC) {
      settings.log.printf("Free variable %d vstatus %d\n", j, vstatus[j]);
    }
  }

  phase2::compute_primal_variables(
    ft, lp.rhs, lp.A, basic_list, nonbasic_list, settings.tight_tol, x);

  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
  if (print_norms) { settings.log.printf("|| x || %e\n", vector_norm2<i_t, f_t>(x)); }

#ifdef COMPUTE_PRIMAL_RESIDUAL
  std::vector<f_t> residual = lp.rhs;
  matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
  f_t primal_residual = vector_norm_inf<i_t, f_t>(residual);
  if (primal_residual > settings.primal_tol) {
    settings.log.printf("|| A*x - b || %e\n", primal_residual);
  }
#endif

  if (delta_y_steepest_edge.size() == 0) {
    delta_y_steepest_edge.resize(n);
    if (slack_basis) {
      phase2::initialize_steepest_edge_norms_from_slack_basis(
        basic_list, nonbasic_list, delta_y_steepest_edge);
    } else {
      std::fill(delta_y_steepest_edge.begin(), delta_y_steepest_edge.end(), -1);
      if (phase2::initialize_steepest_edge_norms(
            lp, settings, start_time, basic_list, ft, delta_y_steepest_edge) == -1) {
        return dual::status_t::TIME_LIMIT;
      }
    }
  } else {
    settings.log.printf("using exisiting steepest edge %e\n",
                        vector_norm2<i_t, f_t>(delta_y_steepest_edge));
  }

  if (phase == 2) {
    settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.     Perturb  Time\n");
  }

  const i_t iter_limit = settings.iteration_limit;
  std::vector<f_t> delta_y(m, 0.0);
  std::vector<f_t> delta_z(n, 0.0);
  std::vector<f_t> delta_x(n, 0.0);
  std::vector<f_t> delta_x_flip(n, 0.0);
  std::vector<f_t> atilde(m, 0.0);
  std::vector<i_t> atilde_mark(m, 0);
  std::vector<i_t> atilde_index;
  std::vector<i_t> nonbasic_mark(n);
  std::vector<i_t> basic_mark(n);
  std::vector<i_t> delta_z_mark(n, 0);
  std::vector<i_t> delta_z_indices;
  std::vector<f_t> v(m, 0.0);
  std::vector<f_t> squared_infeasibilities;
  std::vector<i_t> infeasibility_indices;

  delta_z_indices.reserve(n);

  phase2::reset_basis_mark(basic_list, nonbasic_list, basic_mark, nonbasic_mark);

  std::vector<uint8_t> bounded_variables(n, 0);
  phase2::compute_bounded_info(lp.lower, lp.upper, bounded_variables);

  f_t primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
    lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);

#ifdef CHECK_BASIC_INFEASIBILITIES
  phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 0);
#endif

  csc_matrix_t<i_t, f_t> A_transpose(1, 1, 0);
  lp.A.transpose(A_transpose);

  f_t obj              = compute_objective(lp, x);
  const i_t start_iter = iter;

  i_t sparse_delta_z = 0;
  i_t dense_delta_z  = 0;
  phase2::phase2_timers_t<i_t, f_t> timers(settings.profile && phase == 2);

  while (iter < iter_limit) {
    // Pricing
    i_t direction           = 0;
    i_t basic_leaving_index = -1;
    i_t leaving_index       = -1;
    f_t max_val             = -1.0f;
    timers.start_timer();
    if (settings.use_steepest_edge_pricing) {
      leaving_index = phase2::steepest_edge_pricing_with_infeasibilities(lp,
                                                                         settings,
                                                                         x,
                                                                         delta_y_steepest_edge,
                                                                         basic_mark,
                                                                         squared_infeasibilities,
                                                                         infeasibility_indices,
                                                                         direction,
                                                                         basic_leaving_index,
                                                                         max_val);
    } else {
      // Max infeasibility pricing
      leaving_index = phase2::phase2_pricing(
        lp, settings, x, basic_list, direction, basic_leaving_index, primal_infeasibility);
    }
    timers.pricing_time += timers.stop_timer();
    if (leaving_index == -1) {
      phase2::prepare_optimality(lp,
                                 settings,
                                 ft,
                                 objective,
                                 basic_list,
                                 nonbasic_list,
                                 vstatus,
                                 phase,
                                 start_time,
                                 max_val,
                                 iter,
                                 x,
                                 y,
                                 z,
                                 sol);
      status = dual::status_t::OPTIMAL;
      break;
    }

    // BTran
    // BT*delta_y = -delta_zB = -sigma*ei
    timers.start_timer();
    sparse_vector_t<i_t, f_t> delta_y_sparse(m, 0);
    sparse_vector_t<i_t, f_t> UTsol_sparse(m, 0);
    phase2::compute_delta_y(ft, basic_leaving_index, direction, delta_y_sparse, UTsol_sparse);
    timers.btran_time += timers.stop_timer();

    const f_t steepest_edge_norm_check = delta_y_sparse.norm2_squared();
    if (delta_y_steepest_edge[leaving_index] <
        settings.steepest_edge_ratio * steepest_edge_norm_check) {
      constexpr bool verbose = false;
      if constexpr (verbose) {
        settings.log.printf(
          "iteration restart due to steepest edge. Leaving %d. Actual %.2e "
          "from update %.2e\n",
          leaving_index,
          steepest_edge_norm_check,
          delta_y_steepest_edge[leaving_index]);
      }
      delta_y_steepest_edge[leaving_index] = steepest_edge_norm_check;
      continue;
    }

    timers.start_timer();
    i_t delta_y_nz0      = 0;
    const i_t nz_delta_y = delta_y_sparse.i.size();
    for (i_t k = 0; k < nz_delta_y; k++) {
      if (std::abs(delta_y_sparse.x[k]) > 1e-12) { delta_y_nz0++; }
    }
    const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(m) * 100.0;
    const bool use_transpose        = delta_y_nz_percentage <= 30.0;
    if (use_transpose) {
      sparse_delta_z++;
      phase2::compute_delta_z(A_transpose,
                              delta_y_sparse,
                              leaving_index,
                              direction,
                              nonbasic_mark,
                              delta_z_mark,
                              delta_z_indices,
                              delta_z);
    } else {
      dense_delta_z++;
      // delta_zB = sigma*ei
      delta_y_sparse.to_dense(delta_y);
      phase2::compute_reduced_cost_update(lp,
                                          basic_list,
                                          nonbasic_list,
                                          delta_y,
                                          leaving_index,
                                          direction,
                                          delta_z_mark,
                                          delta_z_indices,
                                          delta_z);
    }
    timers.delta_z_time += timers.stop_timer();

#ifdef COMPUTE_DUAL_RESIDUAL
    std::vector<f_t> dual_residual;
    std::vector<f_t> zeros(n, 0.0);
    phase2::compute_dual_residual(lp.A, zeros, delta_y, delta_z, dual_residual);
    // || A'*delta_y + delta_z ||_inf
    f_t dual_residual_norm = vector_norm_inf<i_t, f_t>(dual_residual);
    settings.log.printf(
      "|| A'*dy - dz || %e use transpose %d\n", dual_residual_norm, use_transpose);
#endif

    // Ratio test
    f_t step_length;
    i_t entering_index          = -1;
    i_t nonbasic_entering_index = -1;
    const bool harris_ratio     = settings.use_harris_ratio;
    const bool bound_flip_ratio = settings.use_bound_flip_ratio;
    if (harris_ratio) {
      f_t max_step_length = phase2::first_stage_harris(lp, vstatus, nonbasic_list, z, delta_z);
      entering_index      = phase2::second_stage_harris(lp,
                                                   vstatus,
                                                   nonbasic_list,
                                                   z,
                                                   delta_z,
                                                   max_step_length,
                                                   step_length,
                                                   nonbasic_entering_index);
    } else if (bound_flip_ratio) {
      timers.start_timer();
      f_t slope = direction == 1 ? (lp.lower[leaving_index] - x[leaving_index])
                                 : (x[leaving_index] - lp.upper[leaving_index]);
      bound_flipping_ratio_test_t<i_t, f_t> bfrt(settings,
                                                 start_time,
                                                 m,
                                                 n,
                                                 slope,
                                                 lp.lower,
                                                 lp.upper,
                                                 bounded_variables,
                                                 vstatus,
                                                 nonbasic_list,
                                                 z,
                                                 delta_z,
                                                 delta_z_indices,
                                                 nonbasic_mark);
      entering_index = bfrt.compute_step_length(step_length, nonbasic_entering_index);
      timers.bfrt_time += timers.stop_timer();
    } else {
      entering_index = phase2::phase2_ratio_test(
        lp, settings, vstatus, nonbasic_list, z, delta_z, step_length, nonbasic_entering_index);
    }
    if (entering_index == -2) { return dual::status_t::TIME_LIMIT; }
    if (entering_index == -3) { return dual::status_t::CONCURRENT_LIMIT; }
    if (entering_index == -1) {
      settings.log.printf("No entering variable found. Iter %d\n", iter);
      settings.log.printf("Scaled infeasibility %e\n", max_val);
      f_t perturbation = phase2::amount_of_perturbation(lp, objective);

      if (perturbation > 0.0 && phase == 2) {
        // Try to remove perturbation
        std::vector<f_t> unperturbed_y(m);
        std::vector<f_t> unperturbed_z(n);
        phase2::compute_dual_solution_from_basis(
          lp, ft, basic_list, nonbasic_list, unperturbed_y, unperturbed_z);
        {
          const f_t dual_infeas = phase2::dual_infeasibility(
            lp, settings, vstatus, unperturbed_z, settings.tight_tol, settings.dual_tol);
          settings.log.printf("Dual infeasibility after removing perturbation %e\n", dual_infeas);
          if (dual_infeas <= settings.dual_tol) {
            settings.log.printf("Removed perturbation of %.2e.\n", perturbation);
            z            = unperturbed_z;
            y            = unperturbed_y;
            perturbation = 0.0;

            std::vector<f_t> unperturbed_x(n);
            phase2::compute_primal_solution_from_basis(
              lp, ft, basic_list, nonbasic_list, vstatus, unperturbed_x);
            x                    = unperturbed_x;
            primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
              lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);
            settings.log.printf("Updated primal infeasibility: %e\n", primal_infeasibility);

            objective = lp.objective;
            // Need to reset the objective value, since we have recomputed x
            obj = phase2::compute_perturbed_objective(objective, x);
            if (dual_infeas <= settings.dual_tol && primal_infeasibility <= settings.primal_tol) {
              phase2::prepare_optimality(lp,
                                         settings,
                                         ft,
                                         objective,
                                         basic_list,
                                         nonbasic_list,
                                         vstatus,
                                         phase,
                                         start_time,
                                         max_val,
                                         iter,
                                         x,
                                         y,
                                         z,
                                         sol);
              status = dual::status_t::OPTIMAL;
              break;
            }
            settings.log.printf(
              "Continuing with perturbation removed and steepest edge norms reset\n");
            // Clear delta_z before restarting the iteration
            phase2::clear_delta_z(
              entering_index, leaving_index, delta_z_mark, delta_z_indices, delta_z);
            continue;
          } else {
            std::vector<f_t> unperturbed_x(n);
            phase2::compute_primal_solution_from_basis(
              lp, ft, basic_list, nonbasic_list, vstatus, unperturbed_x);
            x                    = unperturbed_x;
            primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
              lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);

            const f_t orig_dual_infeas = phase2::dual_infeasibility(
              lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);

            if (primal_infeasibility <= settings.primal_tol &&
                orig_dual_infeas <= settings.dual_tol) {
              phase2::prepare_optimality(lp,
                                         settings,
                                         ft,
                                         objective,
                                         basic_list,
                                         nonbasic_list,
                                         vstatus,
                                         phase,
                                         start_time,
                                         max_val,
                                         iter,
                                         x,
                                         y,
                                         z,
                                         sol);
              status = dual::status_t::OPTIMAL;
              break;
            }
            settings.log.printf("Failed to remove perturbation of %.2e.\n", perturbation);
          }
        }
      }

      if (perturbation == 0.0 && phase == 2) {
        constexpr bool use_farkas = false;
        if constexpr (use_farkas) {
          std::vector<f_t> farkas_y;
          std::vector<f_t> farkas_zl;
          std::vector<f_t> farkas_zu;
          f_t farkas_constant;
          std::vector<f_t> my_delta_y;
          delta_y_sparse.to_dense(my_delta_y);

          // TODO(CMM): Do I use the perturbed or unperturbed objective?
          const f_t obj_val = phase2::compute_perturbed_objective(objective, x);
          phase2::compute_farkas_certificate(lp,
                                             settings,
                                             vstatus,
                                             x,
                                             y,
                                             z,
                                             my_delta_y,
                                             delta_z,
                                             direction,
                                             leaving_index,
                                             obj_val,
                                             farkas_y,
                                             farkas_zl,
                                             farkas_zu,
                                             farkas_constant);
        }
      }

      const f_t dual_infeas =
        phase2::dual_infeasibility(lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
      settings.log.printf("Dual infeasibility %e\n", dual_infeas);
      const f_t primal_inf = phase2::primal_infeasibility(lp, settings, vstatus, x);
      settings.log.printf("Primal infeasibility %e\n", primal_inf);
      settings.log.printf("Updates %d\n", ft.num_updates());
      settings.log.printf("Steepest edge %e\n", max_val);
      if (dual_infeas > settings.dual_tol) {
        settings.log.printf(
          "Numerical issues encountered. No entering variable found with large infeasibility.\n");
        return dual::status_t::NUMERICAL;
      }
      return dual::status_t::DUAL_UNBOUNDED;
    }

    timers.start_timer();
    // Update dual variables
    // y <- y + steplength * delta_y
    // z <- z + steplength * delta_z
    phase2::update_dual_variables(
      delta_y_sparse, delta_z_indices, delta_z, step_length, leaving_index, y, z);
    timers.vector_time += timers.stop_timer();

#ifdef COMPUTE_DUAL_RESIDUAL
    phase2::compute_dual_residual(lp.A, objective, y, z, dual_res1);
    f_t dual_res_norm = vector_norm_inf<i_t, f_t>(dual_res1);
    if (dual_res_norm > settings.dual_tol) {
      settings.log.printf("|| A'*y + z - c || %e steplength %e\n", dual_res_norm, step_length);
    }
#endif

    timers.start_timer();
    // Update primal variable
    const i_t num_flipped = phase2::flip_bounds(lp,
                                                settings,
                                                bounded_variables,
                                                objective,
                                                z,
                                                delta_z_indices,
                                                nonbasic_list,
                                                entering_index,
                                                vstatus,
                                                delta_x_flip,
                                                atilde_mark,
                                                atilde,
                                                atilde_index);

    timers.flip_time += timers.stop_timer();

    sparse_vector_t<i_t, f_t> delta_xB_0_sparse(m, 0);
    if (num_flipped > 0) {
      timers.start_timer();
      phase2::adjust_for_flips(ft,
                               basic_list,
                               delta_z_indices,
                               atilde_index,
                               atilde,
                               atilde_mark,
                               delta_xB_0_sparse,
                               delta_x_flip,
                               x);
      timers.ftran_time += timers.stop_timer();
    }

    timers.start_timer();
    sparse_vector_t<i_t, f_t> utilde_sparse(m, 0);
    sparse_vector_t<i_t, f_t> scaled_delta_xB_sparse(m, 0);
    sparse_vector_t<i_t, f_t> rhs_sparse(lp.A, entering_index);
    if (phase2::compute_delta_x(lp,
                                ft,
                                entering_index,
                                leaving_index,
                                basic_leaving_index,
                                direction,
                                basic_list,
                                delta_x_flip,
                                rhs_sparse,
                                x,
                                utilde_sparse,
                                scaled_delta_xB_sparse,
                                delta_x) == -1) {
      settings.log.printf("Failed to compute delta_x. Iter %d\n", iter);
      return dual::status_t::NUMERICAL;
    }

    timers.ftran_time += timers.stop_timer();

#ifdef CHECK_PRIMAL_STEP
    std::vector<f_t> residual(m);
    matrix_vector_multiply(lp.A, 1.0, delta_x, 1.0, residual);
    f_t primal_step_err = vector_norm_inf<i_t, f_t>(residual);
    if (primal_step_err > 1e-4) { settings.log.printf("|| A * dx || %e\n", primal_step_err); }
#endif

    timers.start_timer();
    const i_t steepest_edge_status = phase2::update_steepest_edge_norms(settings,
                                                                        basic_list,
                                                                        ft,
                                                                        direction,
                                                                        delta_y_sparse,
                                                                        steepest_edge_norm_check,
                                                                        scaled_delta_xB_sparse,
                                                                        basic_leaving_index,
                                                                        entering_index,
                                                                        v,
                                                                        delta_y_steepest_edge);
#ifdef STEEPEST_EDGE_DEBUG
    if (steepest_edge_status == -1) {
      settings.log.printf("Num updates %d\n", ft.num_updates());
      settings.log.printf("|| rhs || %e\n", vector_norm_inf(rhs));
    }
#endif
    assert(steepest_edge_status == 0);
    timers.se_norms_time += timers.stop_timer();

    timers.start_timer();
    // x <- x + delta_x
    phase2::update_primal_variables(scaled_delta_xB_sparse, basic_list, delta_x, entering_index, x);
    timers.vector_time += timers.stop_timer();

#ifdef COMPUTE_PRIMAL_RESIDUAL
    residual = lp.rhs;
    matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
    primal_residual = vector_norm_inf<i_t, f_t>(residual);
    if (iter % 100 == 0 && primal_residual > 10 * settings.primal_tol) {
      settings.log.printf("|| A*x - b || %e\n", primal_residual);
    }
#endif

    timers.start_timer();
    // TODO(CMM): Do I also need to update the objective due to the bound flips?
    // TODO(CMM): I'm using the unperturbed objective here, should this be the perturbed objective?
    phase2::update_objective(
      basic_list, scaled_delta_xB_sparse.i, lp.objective, delta_x, entering_index, obj);
    timers.objective_time += timers.stop_timer();

    timers.start_timer();
    // Update primal infeasibilities due to changes in basic variables
    // from flipping bounds
#ifdef CHECK_BASIC_INFEASIBILITIES
    phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 2);
#endif
    phase2::update_primal_infeasibilities(lp,
                                          settings,
                                          basic_list,
                                          x,
                                          entering_index,
                                          leaving_index,
                                          delta_xB_0_sparse.i,
                                          squared_infeasibilities,
                                          infeasibility_indices,
                                          primal_infeasibility);
    // Update primal infeasibilities due to changes in basic variables
    // from the leaving and entering variables
    phase2::update_primal_infeasibilities(lp,
                                          settings,
                                          basic_list,
                                          x,
                                          entering_index,
                                          leaving_index,
                                          scaled_delta_xB_sparse.i,
                                          squared_infeasibilities,
                                          infeasibility_indices,
                                          primal_infeasibility);
    // Update the entering variable
    phase2::update_single_primal_infeasibility(lp.lower,
                                               lp.upper,
                                               x,
                                               settings.primal_tol,
                                               squared_infeasibilities,
                                               infeasibility_indices,
                                               entering_index,
                                               primal_infeasibility);

    phase2::clean_up_infeasibilities(squared_infeasibilities, infeasibility_indices);

#if CHECK_PRIMAL_INFEASIBILITIES
    phase2::check_primal_infeasibilities(
      lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);
#endif
    timers.update_infeasibility_time += timers.stop_timer();

    // Clear delta_x
    phase2::clear_delta_x(basic_list, entering_index, scaled_delta_xB_sparse, delta_x);

    timers.start_timer();
    f_t sum_perturb = 0.0;
    phase2::compute_perturbation(lp, settings, delta_z_indices, z, objective, sum_perturb);
    timers.perturb_time += timers.stop_timer();

    // Update basis information
    vstatus[entering_index] = variable_status_t::BASIC;
    if (lp.lower[leaving_index] != lp.upper[leaving_index]) {
      vstatus[leaving_index] = static_cast<variable_status_t>(-direction);
    } else {
      vstatus[leaving_index] = variable_status_t::NONBASIC_FIXED;
    }
    basic_list[basic_leaving_index]        = entering_index;
    nonbasic_list[nonbasic_entering_index] = leaving_index;
    nonbasic_mark[entering_index]          = -1;
    nonbasic_mark[leaving_index]           = nonbasic_entering_index;
    basic_mark[leaving_index]              = -1;
    basic_mark[entering_index]             = basic_leaving_index;

#ifdef CHECK_BASIC_INFEASIBILITIES
    phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 5);
#endif

    timers.start_timer();
    // Refactor or update the basis factorization
    bool should_refactor = ft.num_updates() > settings.refactor_frequency;
    if (!should_refactor) {
      i_t recommend_refactor = ft.update(utilde_sparse, UTsol_sparse, basic_leaving_index);
#ifdef CHECK_UPDATE
      phase2::check_update(lp, settings, ft, basic_list, basic_leaving_index);
#endif
      should_refactor = recommend_refactor == 1;
    }

#ifdef CHECK_BASIC_INFEASIBILITIES
    phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 6);
#endif
    if (should_refactor) {
      bool should_recompute_x = false;
      if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
          -1) {
        should_recompute_x = true;
        settings.log.printf("Failed to factorize basis. Iteration %d\n", iter);
        if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
        basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
        i_t count = 0;
        while (factorize_basis(
                 lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) == -1) {
          settings.log.printf("Failed to repair basis. Iteration %d. %d deficient columns.\n",
                              iter,
                              static_cast<int>(deficient.size()));
          if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
          settings.threshold_partial_pivoting_tol = 1.0;
          count++;
          if (count > 10) { return dual::status_t::NUMERICAL; }
          basis_repair(
            lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);

#ifdef CHECK_BASIS_REPAIR
          csc_matrix_t<i_t, f_t> B(m, m, 0);
          form_b(lp.A, basic_list, B);
          for (i_t k = 0; k < deficient.size(); ++k) {
            const i_t j         = deficient[k];
            const i_t col_start = B.col_start[j];
            const i_t col_end   = B.col_start[j + 1];
            const i_t col_nz    = col_end - col_start;
            if (col_nz != 1) {
              settings.log.printf("Deficient column %d has %d nonzeros\n", j, col_nz);
            }
            const i_t i = B.i[col_start];
            if (i != slacks_needed[k]) {
              settings.log.printf("Slack %d needed but found %d instead\n", slacks_needed[k], i);
            }
          }
#endif
        }

        settings.log.printf("Successfully repaired basis. Iteration %d\n", iter);
      }
      reorder_basic_list(q, basic_list);
      ft.reset(L, U, p);
      phase2::reset_basis_mark(basic_list, nonbasic_list, basic_mark, nonbasic_mark);
      if (should_recompute_x) {
        std::vector<f_t> unperturbed_x(n);
        phase2::compute_primal_solution_from_basis(
          lp, ft, basic_list, nonbasic_list, vstatus, unperturbed_x);
        x = unperturbed_x;
      }
      phase2::compute_initial_primal_infeasibilities(
        lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);
    }
#ifdef CHECK_BASIC_INFEASIBILITIES
    phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 7);
#endif
    timers.lu_update_time += timers.stop_timer();

    timers.start_timer();
    phase2::compute_steepest_edge_norm_entering(
      settings, m, ft, basic_leaving_index, entering_index, delta_y_steepest_edge);
    timers.se_entering_time += timers.stop_timer();

#ifdef STEEPEST_EDGE_DEBUG
    if (iter < 100 || iter % 100 == 0))
    {
      phase2::check_steepest_edge_norms(settings, basic_list, ft, delta_y_steepest_edge);
    }
#endif

#ifdef CHECK_BASIS_MARK
    phase2::check_basis_mark(settings, basic_list, nonbasic_list, basic_mark, nonbasic_mark);
#endif

    iter++;

    // Clear delta_z
    phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark, delta_z_indices, delta_z);

    f_t now = toc(start_time);
    if ((iter - start_iter) < settings.first_iteration_log ||
        (iter % settings.iteration_log_frequency) == 0) {
      if (phase == 1 && iter == 1) {
        settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.     Perturb  Time\n");
      }
      settings.log.printf("%5d %+.16e %7d %.8e %.2e %.2f\n",
                          iter,
                          compute_user_objective(lp, obj),
                          infeasibility_indices.size(),
                          primal_infeasibility,
                          sum_perturb,
                          now);
    }

    if (obj >= settings.cut_off) {
      settings.log.printf("Solve cutoff. Current objecive %e. Cutoff %e\n", obj, settings.cut_off);
      return dual::status_t::CUTOFF;
    }

    if (now > settings.time_limit) { return dual::status_t::TIME_LIMIT; }

    if (settings.concurrent_halt != nullptr &&
        settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
      return dual::status_t::CONCURRENT_LIMIT;
    }
  }
  if (iter >= iter_limit) { status = dual::status_t::ITERATION_LIMIT; }

  if (phase == 2) {
    timers.print_timers(settings);
    constexpr bool print_stats = false;
    if constexpr (print_stats) {
      settings.log.printf("Sparse delta_z %8d %8.2f%\n",
                          sparse_delta_z,
                          100.0 * sparse_delta_z / (sparse_delta_z + dense_delta_z));
      settings.log.printf("Dense delta_z  %8d %8.2f%\n",
                          dense_delta_z,
                          100.0 * dense_delta_z / (sparse_delta_z + dense_delta_z));
      ft.print_stats();
    }
  }
  return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template dual::status_t dual_phase2_cu<int, double>(
  int phase,
  int slack_basis,
  double start_time,
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  std::vector<variable_status_t>& vstatus,
  lp_solution_t<int, double>& sol,
  int& iter,
  std::vector<double>& steepest_edge_norms);

#endif

}  // namespace cuopt::linear_programming::dual_simplex