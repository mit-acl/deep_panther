#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <solver_ipopt.hpp>

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(py_panther, m)
{
  m.doc() = "pybind11 py_panther plugin";

  m.def("getMinTimeDoubleIntegrator3DFromState", &getMinTimeDoubleIntegrator3DFromState);

  py::class_<mt::state>(m, "state")
      .def(py::init<>())  /////////////////////////
      .def_readwrite("pos", &mt::state::pos)
      .def_readwrite("vel", &mt::state::vel)
      .def_readwrite("accel", &mt::state::accel)
      .def_readwrite("yaw", &mt::state::yaw)
      .def_readwrite("dyaw", &mt::state::dyaw)
      .def_readwrite("ddyaw", &mt::state::ddyaw)
      .def("printHorizontal", &mt::state::printHorizontal)
      .def("__repr__", [](const mt::state &a) { return "<py_panther.state>"; });

  py::class_<si::solOrGuess>(m, "solOrGuess")
      .def(py::init<>())  /////////////////////////
      .def_readwrite("qp", &si::solOrGuess::qp)
      .def_readwrite("qy", &si::solOrGuess::qy)
      .def_readwrite("traj", &si::solOrGuess::traj)
      .def_readwrite("knots_p", &si::solOrGuess::knots_p)
      .def_readwrite("knots_y", &si::solOrGuess::knots_y)
      .def_readwrite("solver_succeeded", &si::solOrGuess::solver_succeeded)
      .def_readwrite("is_repeated", &si::solOrGuess::is_repeated)
      .def_readwrite("cost", &si::solOrGuess::cost)
      .def_readwrite("obst_avoidance_violation", &si::solOrGuess::obst_avoidance_violation)
      .def_readwrite("dyn_lim_violation", &si::solOrGuess::dyn_lim_violation)
      .def_readwrite("aug_cost", &si::solOrGuess::aug_cost)
      .def_readwrite("is_guess", &si::solOrGuess::is_guess)
      .def_readwrite("deg_p", &si::solOrGuess::deg_p)
      .def_readwrite("deg_y", &si::solOrGuess::deg_y)
      .def("printInfo", &si::solOrGuess::printInfo)
      .def("getTotalTime", &si::solOrGuess::getTotalTime)
      .def("isInCollision", &si::solOrGuess::isInCollision)
      .def("fillTraj", &si::solOrGuess::fillTraj)  //, pybind11::arg("dc")
      .def("__repr__", [](const si::solOrGuess &a) { return "<py_panther.solOrGuess>"; });

  py::class_<mt::obstacleForOpt>(m, "obstacleForOpt")
      .def(py::init<>())  /////////////////////////
      .def_readwrite("ctrl_pts", &mt::obstacleForOpt::ctrl_pts)
      .def_readwrite("bbox_inflated", &mt::obstacleForOpt::bbox_inflated)
      .def("printInfo", &mt::obstacleForOpt::printInfo)
      .def("__repr__", [](const mt::obstacleForOpt &a) { return "<py_panther.obstacleForOpt>"; });

  py::class_<mt::parameters>(m, "parameters")
      .def(py::init<>())  /////////////////////////
      // This part below should be copied/pasted from parameters of panther_types.hpp
      // clang-format off
      .def_readwrite("yaw_scaling", &mt::parameters::yaw_scaling)                          
      .def_readwrite("agents_ids", &mt::parameters::agents_ids)                          
      .def_readwrite("is_multiagent", &mt::parameters::is_multiagent)                          
      .def_readwrite("look_teammates", &mt::parameters::look_teammates)                          
      .def_readwrite("perfect_prediction", &mt::parameters::perfect_prediction)                          
      .def_readwrite("use_panther_star", &mt::parameters::use_panther_star)                          
      .def_readwrite("use_delaycheck", &mt::parameters::use_delaycheck)                          
      .def_readwrite("use_ff", &mt::parameters::use_ff)                          
      .def_readwrite("visual", &mt::parameters::visual)                          
      .def_readwrite("color_type_expert", &mt::parameters::color_type_expert)                      
      .def_readwrite("color_type_student", &mt::parameters::color_type_student)                      
      .def_readwrite("n_agents", &mt::parameters::n_agents)                        
      .def_readwrite("num_of_trajs_per_replan", &mt::parameters::num_of_trajs_per_replan)         
      .def_readwrite("max_num_of_initial_guesses", &mt::parameters::max_num_of_initial_guesses)         
      .def_readwrite("dc", &mt::parameters::dc)                              
      .def_readwrite("goal_radius", &mt::parameters::goal_radius)                     
      .def_readwrite("drone_bbox", &mt::parameters::drone_bbox)                    
      .def_readwrite("drone_extra_radius_for_NN", &mt::parameters::drone_extra_radius_for_NN)                    
      .def_readwrite("Ra", &mt::parameters::Ra)                              
      .def_readwrite("impose_FOV_in_trajCB", &mt::parameters::impose_FOV_in_trajCB)            
      .def_readwrite("pause_time_when_replanning", &mt::parameters::pause_time_when_replanning)       
      .def_readwrite("replanning_trigger_time_student", &mt::parameters::replanning_trigger_time_student)         
      .def_readwrite("replanning_trigger_time_expert", &mt::parameters::replanning_trigger_time_expert)         
      .def_readwrite("replanning_lookahead_time", &mt::parameters::replanning_lookahead_time)       
      .def_readwrite("max_runtime_octopus_search", &mt::parameters::max_runtime_octopus_search)      
      .def_readwrite("fov_x_deg", &mt::parameters::fov_x_deg)                       
      .def_readwrite("fov_y_deg", &mt::parameters::fov_y_deg)                       
      .def_readwrite("fov_depth", &mt::parameters::fov_depth)                       
      .def_readwrite("angle_deg_focus_front", &mt::parameters::angle_deg_focus_front)           
      .def_readwrite("x_min", &mt::parameters::x_min)                           
      .def_readwrite("x_max", &mt::parameters::x_max)                           
      .def_readwrite("y_min", &mt::parameters::y_min)                           
      .def_readwrite("y_max", &mt::parameters::y_max)                           
      .def_readwrite("z_min", &mt::parameters::z_min)                           
      .def_readwrite("z_max", &mt::parameters::z_max)                           
      .def_readwrite("ydot_max", &mt::parameters::ydot_max) 
      .def_readwrite("v_max", &mt::parameters::v_max)                           
      .def_readwrite("a_max", &mt::parameters::a_max)
      .def_readwrite("j_max", &mt::parameters::j_max)
      .def_readwrite("factor_alpha", &mt::parameters::factor_alpha)                    
      .def_readwrite("max_seconds_keeping_traj", &mt::parameters::max_seconds_keeping_traj)        
      .def_readwrite("a_star_samp_x", &mt::parameters::a_star_samp_x)                   
      .def_readwrite("a_star_samp_y", &mt::parameters::a_star_samp_y)                   
      .def_readwrite("a_star_samp_z", &mt::parameters::a_star_samp_z)                   
      .def_readwrite("a_star_fraction_voxel_size", &mt::parameters::a_star_fraction_voxel_size)      
      .def_readwrite("a_star_bias", &mt::parameters::a_star_bias)                     
      .def_readwrite("res_plot_traj", &mt::parameters::res_plot_traj)                   
      .def_readwrite("factor_alloc", &mt::parameters::factor_alloc)                    
      .def_readwrite("alpha_shrink", &mt::parameters::alpha_shrink)                    
      .def_readwrite("norminv_prob", &mt::parameters::norminv_prob)                    
      .def_readwrite("disc_pts_per_interval_oct_search", &mt::parameters::disc_pts_per_interval_oct_search)
      .def_readwrite("c_smooth_yaw_search", &mt::parameters::c_smooth_yaw_search)             
      .def_readwrite("c_visibility_yaw_search", &mt::parameters::c_visibility_yaw_search)         
      .def_readwrite("c_maxydot_yaw_search", &mt::parameters::c_maxydot_yaw_search)            
      .def_readwrite("c_pos_smooth", &mt::parameters::c_pos_smooth)                    
      .def_readwrite("c_yaw_smooth", &mt::parameters::c_yaw_smooth)                    
      .def_readwrite("c_fov", &mt::parameters::c_fov)                           
      .def_readwrite("c_final_pos", &mt::parameters::c_final_pos)                     
      .def_readwrite("c_final_yaw", &mt::parameters::c_final_yaw)                     
      .def_readwrite("c_total_time", &mt::parameters::c_total_time)                                
      .def_readwrite("print_graph_yaw_info", &mt::parameters::print_graph_yaw_info)            
      .def_readwrite("z_goal_when_using_rviz", &mt::parameters::z_goal_when_using_rviz)               
      .def_readwrite("mode", &mt::parameters::mode)                            
      .def_readwrite("b_T_c", &mt::parameters::b_T_c)                                                 
      .def_readwrite("basis", &mt::parameters::basis)                           
      .def_readwrite("num_max_of_obst", &mt::parameters::num_max_of_obst)  
      .def_readwrite("fitter_total_time", &mt::parameters::fitter_total_time)                              
      .def_readwrite("num_seg", &mt::parameters::num_seg)                         
      .def_readwrite("deg_pos", &mt::parameters::deg_pos)                         
      .def_readwrite("deg_yaw", &mt::parameters::deg_yaw)                         
      .def_readwrite("num_of_yaw_per_layer", &mt::parameters::num_of_yaw_per_layer)            
      .def_readwrite("fitter_num_samples", &mt::parameters::fitter_num_samples)              
      .def_readwrite("fitter_num_seg", &mt::parameters::fitter_num_seg)                  
      .def_readwrite("fitter_deg_pos", &mt::parameters::fitter_deg_pos)                  
      .def_readwrite("sampler_num_samples", &mt::parameters::sampler_num_samples)
      .def_readwrite("max_dist2goal", &mt::parameters::max_dist2goal)
      .def_readwrite("max_dist2obs", &mt::parameters::max_dist2obs)
      .def_readwrite("max_side_bbox_obs", &mt::parameters::max_side_bbox_obs)
      .def_readwrite("max_dist2BSPoscPoint", &mt::parameters::max_dist2BSPoscPoint)
      .def_readwrite("use_expert", &mt::parameters::use_expert)
      .def_readwrite("use_student", &mt::parameters::use_student)
      .def_readwrite("student_policy_path", &mt::parameters::student_policy_path)
      .def_readwrite("static_planning", &mt::parameters::static_planning)
      .def_readwrite("use_closed_form_yaw_student", &mt::parameters::use_closed_form_yaw_student)
      .def_readwrite("lambda_obst_avoidance_violation", &mt::parameters::lambda_obst_avoidance_violation)
      .def_readwrite("lambda_dyn_lim_violation", &mt::parameters::lambda_dyn_lim_violation)
      .def_readwrite("num_of_intervals", &mt::parameters::num_of_intervals)
      .def_readwrite("gamma", &mt::parameters::gamma)
      .def_readwrite("obstacle_edge_cb_duration", &mt::parameters::obstacle_edge_cb_duration)
      // clang-format on
      .def("__repr__", [](const mt::parameters &a) { return "<py_panther.parameters>"; });

  py::class_<SolverIpopt>(m, "SolverIpopt")
      .def(py::init<mt::parameters>())  /////////////////////////
      .def("optimize", &SolverIpopt::optimize)
      .def("setInitStateFinalStateInitTFinalT", &SolverIpopt::setInitStateFinalStateInitTFinalT)
      .def("setFocusOnObstacle", &SolverIpopt::setFocusOnObstacle)
      .def("setObstaclesForOpt", &SolverIpopt::setObstaclesForOpt)
      .def("getBestSolutions", &SolverIpopt::getBestSolutions)
      .def("getBestSolution", &SolverIpopt::getBestSolution)
      .def("getGuesses", &SolverIpopt::getGuesses)
      .def("computeCost", &SolverIpopt::computeCost)
      .def("computeDynLimitsConstraintsViolation", &SolverIpopt::computeDynLimitsConstraintsViolation)
      .def("getInfoLastOpt", &SolverIpopt::getInfoLastOpt)
      .def_readwrite("par_", &SolverIpopt::par_)
      .def("__repr__", [](const SolverIpopt &a) { return "<py_panther.SolverIpopt>"; });

  py::class_<Fitter>(m, "Fitter")
      .def(py::init<int>())  /////////////////////////
      .def("fit", &Fitter::fit)
      .def("__repr__", [](const Fitter &a) { return "<py_panther.Fitter>"; });

  py::class_<ClosedFormYawSolver>(m, "ClosedFormYawSolver")
      .def(py::init<>())  /////////////////////////
      .def("getyCPsfrompCPSUsingClosedForm", &ClosedFormYawSolver::getyCPsfrompCPSUsingClosedForm)
      .def("__repr__", [](const Fitter &a) { return "<py_panther.ClosedFormYawSolver>"; });
}
