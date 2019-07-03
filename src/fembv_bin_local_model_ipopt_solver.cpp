#include "fembv_bin_local_model_ipopt_solver.hpp"

#include <stdexcept>

namespace fembvpp {

namespace detail {

std::string ipopt_status_to_string(Ipopt::SolverReturn status)
{
   switch (status) {
   case Ipopt::SUCCESS: return "success";
   case Ipopt::MAXITER_EXCEEDED: return "maximum iterations exceeded";
   case Ipopt::CPUTIME_EXCEEDED: return "CPU time exceeded";
   case Ipopt::STOP_AT_TINY_STEP: return "stopped at tiny step";
   case Ipopt::STOP_AT_ACCEPTABLE_POINT: return "stopped at acceptable point";
   case Ipopt::LOCAL_INFEASIBILITY: return "local infeasibility";
   case Ipopt::USER_REQUESTED_STOP: return "user requested stop";
   case Ipopt::FEASIBLE_POINT_FOUND: return "feasible point found";
   case Ipopt::DIVERGING_ITERATES: return "diverging iterates";
   case Ipopt::RESTORATION_FAILURE: return "restoration failure";
   case Ipopt::ERROR_IN_STEP_COMPUTATION: return "error in step computation";
   case Ipopt::INVALID_NUMBER_DETECTED: return "invalid number detected";
   case Ipopt::TOO_FEW_DEGREES_OF_FREEDOM: return "too few degrees of freedom";
   case Ipopt::INVALID_OPTION: return "invalid option";
   case Ipopt::OUT_OF_MEMORY: return "out of memory";
   case Ipopt::INTERNAL_ERROR: return "internal error";
   case Ipopt::UNASSIGNED: return "unassigned error";
   default: return "unknown error";
   }
}

} // namespace detail

FEMBVBin_local_model_ipopt_solver::FEMBVBin_local_model_ipopt_solver(int seed)
   : generator(seed)
{
   // disable console output by default
   ip_solver = new Ipopt::IpoptApplication(false);

   ip_solver->Options()->SetStringValue("jac_d_constant", "yes");
   ip_solver->Options()->SetStringValue("mu_strategy", "adaptive");
   ip_solver->Options()->SetNumericValue("bound_relax_factor", 0);
}

FEMBVBin_local_model_ipopt_solver::FEMBVBin_local_model_ipopt_solver()
{
   // disable console output by default
   ip_solver = new Ipopt::IpoptApplication(false);

   ip_solver->Options()->SetStringValue("jac_d_constant", "yes");
   ip_solver->Options()->SetStringValue("mu_strategy", "adaptive");
   ip_solver->Options()->SetNumericValue("bound_relax_factor", 0);
}

void FEMBVBin_local_model_ipopt_solver::initialize()
{
   Ipopt::ApplicationReturnStatus status = ip_solver->Initialize();

   if (status != Ipopt::Solve_Succeeded) {
      throw std::runtime_error("initialization of solver failed");
   }
   ip_solver->Options()->SetNumericValue("bound_push", -1000);
}

void FEMBVBin_local_model_ipopt_solver::set_max_iterations(int i)
{
   ip_solver->Options()->SetNumericValue("max_iter", i);
}

void FEMBVBin_local_model_ipopt_solver::set_tolerance(double t)
{
  ip_solver->Options()->SetNumericValue("tol", t);
}

} // namespace fembvpp
