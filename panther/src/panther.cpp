/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include <Eigen/StdVector>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <stdlib.h>

#include "panther.hpp"
#include "timer.hpp"
#include "termcolor.hpp"
#include "bspline_utils.hpp"

#include <ros/package.h>

////////////////////////// Needed to call the student
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
//////////////////////////

// #include "exprtk.hpp"

using namespace termcolor;

// Uncomment the type of timer you want:
// typedef ROSTimer MyTimer;
// typedef ROSWallTimer MyTimer;
typedef PANTHER_timers::Timer MyTimer;

Panther::Panther(mt::parameters par) : par_(par)
{
  drone_status_ == DroneStatus::YAWING;
  G_.pos << 0, 0, 0;
  G_term_.pos << 0, 0, 0;

  mtx_initial_cond.lock();
  stateA_.setZero();
  mtx_initial_cond.unlock();

  changeDroneStatus(DroneStatus::GOAL_REACHED);
  resetInitialization();

  log_ptr_ = std::shared_ptr<mt::log>(new mt::log);

  solver_ = new SolverIpopt(par_);                //, log_ptr_
  fitter_ = new Fitter(par_.fitter_num_samples);  //, log_ptr_

  separator_solver_ = new separator::Separator();

  // std::string folder = ros::package::getPath("panther") + "/matlab/casadi_generated_files/";
  // cf_fit3d_ = casadi::Function::load(folder + "fit3d.casadi");

  if (par_.use_student == true)
  {
    pybind11::initialize_interpreter();

    student_caller_ptr_ = new pybind11::object;
    *student_caller_ptr_ =
        pybind11::module::import("compression.utils.other").attr("StudentCaller")(par_.student_policy_path);

    ///// TODO: The first ~5 calls to the student are very slow. It is due to the
    // because of having declared the variables just after class CostComputer(): (see python file), and not inside
    // init()
    // We put those calls here to avoid this overhead while actually planning
    for (int i = 1; i < 10; i++)
    {
      std::cout << "Calling the student!" << std::endl;
      mt::state A;
      mt::state G_term;
      G_term.pos = Eigen::Vector3d(10, 0.0, 0.0);
      std::vector<mt::obstacleForOpt> obstacles_for_opt;

      mt::obstacleForOpt tmp;
      tmp.bbox_inflated = Eigen::Vector3d(1.0, 1.0, 1.0);

      std::vector<Eigen::Vector3d> samples;
      for (int k = 0; k < par_.fitter_num_samples; k++)
      {
        samples.push_back(10 * Eigen::Vector3d::Ones());
      }

      tmp.ctrl_pts = fitter_->fit(samples);
      obstacles_for_opt.push_back(tmp);

      pybind11::object result = student_caller_ptr_->attr("predict")(A, obstacles_for_opt, G_term.pos);
      std::cout << "Called the student!" << std::endl;
    }
  }
}

Panther::~Panther()
{
  if (par_.use_student == true)
  {
    pybind11::finalize_interpreter();
  }
}

void Panther::dynTraj2dynTrajCompiled(const mt::dynTraj& traj, mt::dynTrajCompiled& traj_compiled)
{
  if (traj.use_pwp_field == true)
  {
    traj_compiled.pwp_mean = traj.pwp_mean;
    traj_compiled.pwp_var = traj.pwp_var;
    traj_compiled.is_static =
        ((traj.pwp_mean.eval(0.0) - traj.pwp_mean.eval(1e30)).norm() < 1e-5);  // TODO: Improve this
  }
  else
  {
    mtx_t_.lock();

    // typedef exprtk::symbol_table<double> symbol_table_t;
    // typedef exprtk::expression<double> expression_t;
    // typedef exprtk::parser<double> parser_t;

    // Compile the mean
    for (auto function_i : traj.s_mean)
    {
      MathEvaluator engine;
      engine.add_var("t", t_);
      engine.compile(function_i);
      traj_compiled.s_mean.push_back(engine);

      // symbol_table_t symbol_table;
      // symbol_table.add_variable("t", t_);
      // symbol_table.add_constants();
      // expression_t expression;
      // expression.register_symbol_table(symbol_table);
      // parser_t parser;
      // parser.compile(function_i, expression);
      // // traj_compiled.s_mean.push_back(expression);

      // t_ = ros::Time::now().toSec();
      // std::cout << "Option 1=" << expression.value() << std::endl;
      // std::cout << "Option 2=" << engine.value() << std::endl;
    }

    // Compile the variance
    for (auto function_i : traj.s_var)
    {
      MathEvaluator engine;
      engine.add_var("t", t_);
      engine.compile(function_i);
      traj_compiled.s_var.push_back(engine);

      // symbol_table_t symbol_table;
      // symbol_table.add_variable("t", t_);
      // symbol_table.add_constants();
      // expression_t expression;
      // expression.register_symbol_table(symbol_table);
      // parser_t parser;
      // parser.compile(function_i, expression);
      // traj_compiled.s_var.push_back(expression);
    }

    mtx_t_.unlock();

    traj_compiled.is_static =
        (traj.s_mean[0].find("t") == std::string::npos) &&  // there is no dependence on t in the coordinate x
        (traj.s_mean[1].find("t") == std::string::npos) &&  // there is no dependence on t in the coordinate y
        (traj.s_mean[2].find("t") == std::string::npos);    // there is no dependence on t in the coordinate z
  }

  traj_compiled.use_pwp_field = traj.use_pwp_field;
  traj_compiled.is_agent = traj.is_agent;
  traj_compiled.bbox = traj.bbox;
  traj_compiled.id = traj.id;
  traj_compiled.time_received = traj.time_received;  // ros::Time::now().toSec();
}

// Note that this function is here because I need t_ for this evaluation
Eigen::Vector3d Panther::evalMeanDynTrajCompiled(const mt::dynTrajCompiled& traj, double t)
{
  Eigen::Vector3d tmp;

  if (traj.use_pwp_field == true)
  {
    tmp = traj.pwp_mean.eval(t);
  }
  else
  {
    mtx_t_.lock();
    t_ = t;
    tmp << traj.s_mean[0].value(),  ////////////////
        traj.s_mean[1].value(),     ////////////////
        traj.s_mean[2].value();     ////////////////

    mtx_t_.unlock();
  }
  return tmp;
}

// Note that this function is here because it needs t_ for this evaluation
Eigen::Vector3d Panther::evalVarDynTrajCompiled(const mt::dynTrajCompiled& traj, double t)
{
  Eigen::Vector3d tmp;

  if (traj.use_pwp_field == true)
  {
    tmp = traj.pwp_var.eval(t);
  }
  else
  {
    mtx_t_.lock();
    t_ = t;
    tmp << traj.s_var[0].value(),  ////////////////
        traj.s_var[1].value(),     ////////////////
        traj.s_var[2].value();     ////////////////

    mtx_t_.unlock();
  }
  return tmp;
}

void Panther::removeOldTrajectories()
{
  double time_now = ros::Time::now().toSec();
  std::vector<int> ids_to_remove;

  mtx_trajs_.lock();

  for (int index_traj = 0; index_traj < trajs_.size(); index_traj++)
  {
    if ((time_now - trajs_[index_traj].time_received) > par_.max_seconds_keeping_traj)
    {
      ids_to_remove.push_back(trajs_[index_traj].id);
    }
  }

  for (auto id : ids_to_remove)
  {
    // ROS_WARN_STREAM("Removing " << id);
    trajs_.erase(
        std::remove_if(trajs_.begin(), trajs_.end(), [&](mt::dynTrajCompiled const& traj) { return traj.id == id; }),
        trajs_.end());
  }

  mtx_trajs_.unlock();
}

// Note that we need to compile the trajectories inside panther.cpp because t_ is in panther.hpp
void Panther::updateTrajObstacles(mt::dynTraj traj)
{
  MyTimer tmp_t(true);

  if (started_check_ == true && traj.is_agent == true)
  {
    have_received_trajectories_while_checking_ = true;
  }

  mt::dynTrajCompiled traj_compiled;
  dynTraj2dynTrajCompiled(traj, traj_compiled);

  double time_now = ros::Time::now().toSec();

  // std::cout << on_blue << bold << "in  updateTrajObstacles(), waiting to lock mtx_trajs_" << reset << std::endl;
  mtx_trajs_.lock();

  ///////////////////////////////////////
  ///////////////// ADD IT TO THE MAP IF IT DOES NOT EXIST
  ///////////////////////////////////////
  std::vector<mt::dynTrajCompiled>::iterator obs_ptr =
      std::find_if(trajs_.begin(), trajs_.end(),
                   [=](const mt::dynTrajCompiled& traj_compiled) { return traj_compiled.id == traj.id; });

  bool exists_in_local_map = (obs_ptr != std::end(trajs_));

  if (exists_in_local_map)
  {  // if that object already exists, substitute its trajectory
    *obs_ptr = traj_compiled;
  }
  else
  {  // if it doesn't exist, add it to the local map
    trajs_.push_back(traj_compiled);
    // ROS_WARN_STREAM("Adding " << traj_compiled.id);
  }

  ///////////////////////////////////////////
  ///////// LEAVE ONLY THE CLOSEST TRAJECTORY
  ///////////////////////////////////////////

  // if (trajs_.size() >= 1)
  // {
  //   double distance_new_traj = (evalMeanDynTrajCompiled(traj_compiled, time_now) - state_.pos).norm();
  //   double distance_old_traj = (evalMeanDynTrajCompiled(trajs_[0], time_now) - state_.pos).norm();

  //   if (distance_new_traj < distance_old_traj)
  //   {
  //     trajs_[0] = traj_compiled;
  //   }
  // }
  // else
  // {
  //   trajs_.push_back(traj_compiled);
  // }

  // verify(trajs_.size() == 1, "the vector trajs_ should have size = 1");

  ///////////////////////////////////
  ///////////////////////////////////
  ///////////////////////////////////

  mtx_trajs_.unlock();
  // std::cout << red << bold << "in updateTrajObstacles(), mtx_trajs_ unlocked" << reset << std::endl;

  have_received_trajectories_while_checking_ = false;
  // std::cout << bold << blue << "updateTrajObstacles took " << tmp_t << reset << std::endl;
}

bool Panther::IsTranslating()
{
  return (drone_status_ == DroneStatus::GOAL_SEEN || drone_status_ == DroneStatus::TRAVELING);
}

void Panther::addDummyObstacle(double t_start, double t_end, std::vector<mt::obstacleForOpt>& obstacles_for_opt,
                               mt::state& A, std::vector<si::solOrGuess>& splines_fitted)
{
  double delta = (t_end - t_start) / par_.fitter_num_samples;

  //
  // get the vector from where you are now to g_term
  //

  mt::state G_term;
  getG_term(G_term);
  mt::state current_state;
  getState(current_state);

  Eigen::Vector3d n_vec_to_g_term = (G_term.pos - current_state.pos).normalized();

  //
  // first construct dummy obstacle in dynTraj
  //

  mt::dynTraj dummy_traj;
  dummy_traj.use_pwp_field = false;

  dummy_traj.s_mean.push_back(std::to_string(G_term.pos[0] + par_.dist_from_gterm_to_dummy * n_vec_to_g_term.x()));
  dummy_traj.s_mean.push_back(std::to_string(G_term.pos[1] + par_.dist_from_gterm_to_dummy * n_vec_to_g_term.y()));
  dummy_traj.s_mean.push_back(std::to_string(G_term.pos[2] + par_.dist_from_gterm_to_dummy * n_vec_to_g_term.z()));

  dummy_traj.bbox = par_.drone_bbox;
  dummy_traj.id = 6000;            // TODO: hardcoded
  dummy_traj.time_received = 0.0;  // TODO: fix if necessary
  dummy_traj.is_agent = false;

  //
  // compile
  //

  mt::dynTrajCompiled dummy_traj_compiled;
  dynTraj2dynTrajCompiled(dummy_traj, dummy_traj_compiled);

  //
  // construct dummy obstacle_for_opt
  //

  mt::obstacleForOpt dummy_obstacle_for_opt;
  std::vector<Eigen::Vector3d> samples;

  for (int k = 0; k < par_.fitter_num_samples; k++)
  {
    double tk = t_start + k * delta;
    Eigen::Vector3d pos_k = evalMeanDynTrajCompiled(dummy_traj_compiled, tk);
    samples.push_back(pos_k);
  }

  dummy_obstacle_for_opt.ctrl_pts = fitter_->fit(samples);
  dummy_obstacle_for_opt.bbox_inflated = dummy_traj_compiled.bbox + 2 * par_.drone_bbox;
  dummy_obstacle_for_opt.is_dummy = true;

  obstacles_for_opt.push_back(dummy_obstacle_for_opt);

  ///////////////////////// FOR VISUALIZATION

  si::solOrGuess spline_fitted;
  spline_fitted.qp = dummy_obstacle_for_opt.ctrl_pts;
  std::vector<double> qy(par_.num_seg + par_.deg_yaw, 0.0);
  spline_fitted.qy = qy;
  spline_fitted.knots_p = constructKnotsClampedUniformSpline(t_start, t_end, par_.fitter_deg_pos, par_.fitter_num_seg);
  spline_fitted.deg_p = par_.fitter_deg_pos;

  // Dummy for yaw
  spline_fitted.knots_y = spline_fitted.knots_p;
  spline_fitted.deg_y = spline_fitted.deg_p;

  spline_fitted.fillTraj(par_.dc);

  splines_fitted.push_back(spline_fitted);
}

std::vector<mt::obstacleForOpt> Panther::getObstaclesForOpt(double t_start, double t_end,
                                                            std::vector<si::solOrGuess>& splines_fitted)
{
  // std::cout << "In getObstaclesForOpt" << std::endl;

  std::vector<mt::obstacleForOpt> obstacles_for_opt;

  double delta = (t_end - t_start) / par_.fitter_num_samples;

  // std::cout << "delta= " << delta << std::endl;

  // std::cout << "trajs_.size() " << trajs_.size() << std::endl;

  for (int i = 0; i < trajs_.size(); i++)
  {
    mt::obstacleForOpt obstacle_for_opt;

    // Take future samples of the trajectory
    std::vector<Eigen::Vector3d> samples;

    for (int k = 0; k < par_.fitter_num_samples; k++)
    {
      double tk = t_start + k * delta;
      Eigen::Vector3d pos_k = evalMeanDynTrajCompiled(trajs_[i], tk);

      samples.push_back(pos_k);
    }

    obstacle_for_opt.ctrl_pts = fitter_->fit(samples);

    Eigen::Vector3d bbox_inflated = trajs_[i].bbox + 2 * par_.drone_bbox;

    obstacle_for_opt.bbox_inflated = bbox_inflated;

    obstacles_for_opt.push_back(obstacle_for_opt);

    ///////////////////////// FOR VISUALIZATION

    si::solOrGuess spline_fitted;
    spline_fitted.qp = obstacle_for_opt.ctrl_pts;
    std::vector<double> qy(par_.num_seg + par_.deg_yaw, 0.0);
    spline_fitted.qy = qy;
    spline_fitted.knots_p =
        constructKnotsClampedUniformSpline(t_start, t_end, par_.fitter_deg_pos, par_.fitter_num_seg);
    spline_fitted.deg_p = par_.fitter_deg_pos;

    // Dummy for yaw
    spline_fitted.knots_y = spline_fitted.knots_p;
    spline_fitted.deg_y = spline_fitted.deg_p;

    // std::cout << "SPLINE HAS BEEN FITTED" << std::endl;
    // spline_fitted.print();
    // std::cout << "Samples are " << std::endl;
    // std::cout << samples_casadi << std::endl;
    // std::cout << "_____________________" << std::endl;

    spline_fitted.fillTraj(par_.dc);

    // abort();

    // for (auto traj_i : spline_fitted.traj)
    // {
    //   traj_i.printHorizontal();
    // }

    splines_fitted.push_back(spline_fitted);

    // ///////////////////////////
  }

  // std::cout << "obstacles_for_opt.size() " << obstacles_for_opt.size() << std::endl;

  return obstacles_for_opt;
}

void Panther::setTerminalGoal(mt::state& term_goal)
{
  mtx_G_term.lock();
  G_term_ = term_goal;
  mtx_G_term.unlock();

  if (state_initialized_ == true)  // because I need plan_size()>=1
  {
    doStuffTermGoal();
  }
  else
  {
    std::cout << "need_to_do_stuff_term_goal_= " << need_to_do_stuff_term_goal_ << std::endl;
    need_to_do_stuff_term_goal_ = true;  // will be done in updateState();
  }
}

void Panther::getG(mt::state& G)
{
  G = G_;
}

void Panther::getState(mt::state& data)
{
  mtx_state.lock();
  data = state_;
  mtx_state.unlock();
}

void Panther::getG_term(mt::state& data)
{
  mtx_G_term.lock();
  data = G_term_;  // Local copy of the terminal terminal goal
  mtx_G_term.unlock();
}

void Panther::updateState(mt::state data)
{
  state_ = data;

  mtx_plan_.lock();
  if (state_initialized_ == false || plan_.size() == 1)
  {
    plan_.clear();  // (actually not needed because done in resetInitialization()
    mt::state tmp;
    tmp.pos = data.pos;
    tmp.yaw = data.yaw;
    plan_.push_back(tmp);
  }
  mtx_plan_.unlock();

  state_initialized_ = true;

  if (need_to_do_stuff_term_goal_)
  {
    // std::cout << "DOING STUFF TERM GOAL -----------" << std::endl;
    doStuffTermGoal();
    need_to_do_stuff_term_goal_ = false;
  }
}

// This function needs to be called once the state has been initialized
void Panther::doStuffTermGoal()
{
  // if (state_initialized_ == false)  // because I need plan_size()>=1
  // {
  //   std::cout << "[Panther::setTerminalGoal] State not initialized yet, doing nothing" << std::endl;
  //   return;
  // }

  mtx_G_term.lock();
  G_.pos = G_term_.pos;
  mtx_G_term.unlock();

  mtx_planner_status_.lock();
  if (drone_status_ == DroneStatus::GOAL_REACHED)
  {
    /////////////////////////////////
    if (par_.static_planning)  // plan from the same position all the time
    {
      changeDroneStatus(DroneStatus::TRAVELING);
    }
    else
    {
      changeDroneStatus(DroneStatus::YAWING);
    }
  }
  else if (drone_status_ == DroneStatus::GOAL_SEEN)
  {
    // changeDroneStatus(DroneStatus::TRAVELING);
    changeDroneStatus(DroneStatus::YAWING);
  }
  terminal_goal_initialized_ = true;

  // std::cout << bold << red << "[FA] Received Term Goal=" << G_term_.pos.transpose() << reset << std::endl;
  // std::cout << bold << red << "[FA] Received Proj Goal=" << G_.pos.transpose() << reset << std::endl;

  is_new_g_term_ = true;
  mtx_planner_status_.unlock();
}

bool Panther::initializedAllExceptPlanner()
{
  if (!state_initialized_ || !terminal_goal_initialized_)
  {
    /*    std::cout << "state_initialized_= " << state_initialized_ << std::endl;
        std::cout << "terminal_goal_initialized_= " << terminal_goal_initialized_ << std::endl;*/
    return false;
  }
  return true;
}

bool Panther::initializedStateAndTermGoal()
{
  if (!state_initialized_ || !terminal_goal_initialized_)
  {
    return false;
  }
  return true;
}

bool Panther::initialized()
{
  if (!state_initialized_ || !terminal_goal_initialized_ || !planner_initialized_)
  {
    /*    std::cout << "state_initialized_= " << state_initialized_ << std::endl;
        std::cout << "terminal_goal_initialized_= " << terminal_goal_initialized_ << std::endl;
        std::cout << "planner_initialized_= " << planner_initialized_ << std::endl;*/
    return false;
  }
  return true;
}

bool Panther::isReplanningNeeded()
{
  if (initializedStateAndTermGoal() == false)
  {
    return false;  // Note that log is not modified --> will keep its default values
  }

  if (par_.static_planning == true)
  {
    return true;
  }

  //////////////////////////////////////////////////////////////////////////
  mtx_G_term.lock();

  mt::state G_term = G_term_;  // Local copy of the terminal terminal goal

  mtx_G_term.unlock();

  // Check if we have reached the goal
  double dist_to_goal = (G_term.pos - plan_.front().pos).norm();
  // std::cout << "dist_to_goal= " << dist_to_goal << std::endl;
  if (dist_to_goal < par_.goal_radius)
  {
    changeDroneStatus(DroneStatus::GOAL_REACHED);
    exists_previous_pwp_ = false;
  }

  // Check if we have seen the goal in the last replan
  mtx_plan_.lock();
  double dist_last_plan_to_goal = (G_term.pos - plan_.back().pos).norm();
  // std::cout << "dist_last_plan_to_goal= " << dist_last_plan_to_goal << std::endl;
  mtx_plan_.unlock();
  if (dist_last_plan_to_goal < par_.goal_radius && drone_status_ == DroneStatus::TRAVELING)
  {
    changeDroneStatus(DroneStatus::GOAL_SEEN);
    std::cout << "Status changed to GOAL_SEEN!" << std::endl;
    exists_previous_pwp_ = false;
  }

  // Don't plan if drone is not traveling
  if (drone_status_ == DroneStatus::GOAL_REACHED || (drone_status_ == DroneStatus::YAWING) ||
      (drone_status_ == DroneStatus::GOAL_SEEN))
  {
    // std::cout << "No replanning needed because" << std::endl;
    // printDroneStatus();
    return false;
  }
  return true;
}

void Panther::pubObstacleEdge(mt::Edges& edges_obstacles_out)
{
  //
  // Get edges_obstacles
  //

  double t_start = ros::Time::now().toSec();
  double t_final =
      t_start + 5.0;  // 5.0 is just a duration into the future in which we want to visualize obstacle edges.

  mtx_trajs_.lock();

  ConvexHullsOfCurves hulls = convexHullsOfCurves(t_start, t_final);

  mtx_trajs_.unlock();

  edges_obstacles_out = cu::vectorGCALPol2edges(hulls);
}

bool Panther::replan(mt::Edges& edges_obstacles_out, si::solOrGuess& best_solution_expert,
                     std::vector<si::solOrGuess>& best_solutions_expert, si::solOrGuess& best_solution_student,
                     std::vector<si::solOrGuess>& best_solutions_student, std::vector<si::solOrGuess>& guesses,
                     std::vector<si::solOrGuess>& splines_fitted, std::vector<Hyperplane3D>& planes, mt::log& log,
                     int& k_index_end)
{
  //////////
  bool this_replan_uses_new_gterm = false;
  if (is_new_g_term_ == true)
  {
    this_replan_uses_new_gterm = true;
  }
  // else
  // {
  //   return false;
  // }
  is_new_g_term_ = false;
  //////////

  (*log_ptr_) = {};  // Reset the struct with the default values

  mtx_G_term.lock();
  mt::state G_term = G_term_;  // Local copy of the terminal terminal goal
  mtx_G_term.unlock();

  log_ptr_->pos = state_.pos;
  log_ptr_->G_term_pos = G_term.pos;
  log_ptr_->drone_status = drone_status_;

  if (isReplanningNeeded() == false)
  {
    log_ptr_->replanning_was_needed = false;
    log = (*log_ptr_);
    return false;
  }

  std::cout << bold << on_white << "**********************IN REPLAN CB*******************" << reset << std::endl;

  log_ptr_->replanning_was_needed = true;
  log_ptr_->tim_total_replan.tic();

  log_ptr_->tim_initial_setup.tic();

  removeOldTrajectories();

  //////////////////////////////////////////////////////////////////////////
  ///////////////////////// Select mt::state A /////////////////////////////
  //////////////////////////////////////////////////////////////////////////

  mt::state A;
  int k_index;

  // If k_index_end=0, then A = plan_.back() = plan_[plan_.size() - 1]

  mtx_plan_.lock();

  // saturate(deltaT_, par_.lower_bound_runtime_snlopt / par_.dc, par_.upper_bound_runtime_snlopt / par_.dc);

  deltaT_ = par_.replanning_lookahead_time / par_.dc;  // Added October 18, 2021

  k_index_end = std::max((int)(plan_.size() - deltaT_), 0);

  if (plan_.size() < 5)
  {
    k_index_end = 0;
  }

  if (par_.static_planning)  // plan from the same position all the time
  {
    k_index_end = plan_.size() - 1;
  }

  k_index = plan_.size() - 1 - k_index_end;
  A = plan_.get(k_index);

  // std::cout << "When selection A, plan_.size()= " << plan_.size() << std::endl;

  mtx_plan_.unlock();

  //////////////////////////////////////////////////////////////////////////
  ///////////////////////// Get point G ////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////

  double distA2TermGoal = (G_term.pos - A.pos).norm();
  double ra = std::min((distA2TermGoal - 0.001), par_.Ra);  // radius of the sphere S
  mt::state G;
  G.pos = A.pos + ra * (G_term.pos - A.pos).normalized();

  double time_now = ros::Time::now().toSec();  // shouldn't have ros dependency here
  double t_start = k_index * par_.dc + time_now;

  mtx_trajs_.lock();

  std::vector<mt::obstacleForOpt> obstacles_for_opt =
      getObstaclesForOpt(t_start, t_start + par_.fitter_total_time, splines_fitted);

  /////////////////////////////////////////////////////////////////////////
  ////////////////////////Compute trajectory to focus on //////////////////
  /////////////////////////////////////////////////////////////////////////

  double time_allocated = getMinTimeDoubleIntegrator3D(A.pos, A.vel, G.pos, G.vel, par_.v_max, par_.a_max);
  double t_final = t_start + par_.factor_alloc * time_allocated;
  double max_prob_collision = -std::numeric_limits<double>::max();  // it's actually a heuristics of the probability
                                                                    // (we are summing below --> can be >1)
  int argmax_prob_collision = -1;  // will contain the index of the trajectory to focus on

  int num_samplesp1 = 20;
  double delta = 1.0 / num_samplesp1;
  Eigen::Vector3d R = par_.drone_bbox;

  std::vector<double> all_probs;

  for (int i = 0; i < trajs_.size(); i++)
  {
    if (!par_.look_teammates)
    {
      if (trajs_[i].is_agent)
      {
        continue;
      }
    }

    double prob_i = 0.0;
    for (int j = 0; j <= num_samplesp1; j++)
    {
      double t = t_start + j * delta * (t_final - t_start);

      Eigen::Vector3d pos_drone = A.pos + j * delta * (G_term_.pos - A.pos);  // not a random variable
      Eigen::Vector3d pos_obs_mean = evalMeanDynTrajCompiled(trajs_[i], t);
      Eigen::Vector3d pos_obs_std = (evalVarDynTrajCompiled(trajs_[i], t)).cwiseSqrt();
      // std::cout << "pos_obs_std= " << pos_obs_std << std::endl;
      prob_i += probMultivariateNormalDist(-R, R, pos_obs_mean - pos_drone, pos_obs_std);
    }

    all_probs.push_back(prob_i);
    // std::cout << "[Selection] Trajectory " << i << " has P(collision)= " << prob_i * pow(10, 15) << "e-15" <<
    // std::endl;

    if (prob_i > max_prob_collision)
    {
      max_prob_collision = prob_i;
      argmax_prob_collision = i;
      std::cout << bold << red << "argmax_prob_collision's id is: " << trajs_[i].id << std::endl;
    }
  }
  mtx_trajs_.unlock();

  std::cout << "[Selection] Probs of coll --> ";
  for (int i = 0; i < all_probs.size(); i++)
  {
    std::cout << all_probs[i] * pow(10, 15) << "e-15,   ";
  }
  std::cout << std::endl;

  // std::cout.precision(30);
  std::cout << bold << "[Selection] Chosen Trajectory " << argmax_prob_collision
            << ", P(collision)= " << max_prob_collision * pow(10, 5) << "e-5" << std::endl;

  if (obstacles_for_opt.size() == 0)
  {
    std::cout << bold << "No obstacle found" << std::endl;
  }
  else
  {
    verify(obstacles_for_opt.size() >= 1, "obstacles_for_opt should have at least 1 element");
    verify(argmax_prob_collision >= 0, "argmax_prob_collision should be >=0");
    verify(argmax_prob_collision <= (obstacles_for_opt.size() - 1), "argmax_prob_collision should be <= "
                                                                    "(obstacles_for_opt.size() - 1)");
    // keep only the obstacle that has the highest probability of collision
    auto tmp = obstacles_for_opt[argmax_prob_collision];
    obstacles_for_opt.clear();
    obstacles_for_opt.push_back(tmp);

    ////////////////////////////////////////
    ////////////////////////////////////////
    ////////////////////////////////////////

    verify(obstacles_for_opt.size() == 1, "obstacles_for_opt should have only 1 element");
  }

  if (obstacles_for_opt.size() > par_.num_max_of_obst)
  {
    std::cout << red << bold << "Too many obstacles. Run Matlab again with a higher num_max_of_obst" << reset
              << std::endl;
    abort();
  }

  int n_safe_trajs_expert = 0;
  int n_safe_trajs_student = 0;
  int n_safe_trajs = 0;

  // si::solOrGuess best_solution;
  if (par_.use_expert)
  {
    //////////////////////////////////////////////////////////////////////////
    ///////////////////////// Set Times in optimization //////////////////////
    //////////////////////////////////////////////////////////////////////////

    // solver_->setMaxRuntimeOctopusSearch(par_.max_runtime_octopus_search);  //, par_.kappa, par_.mu);

    //////////////////////

    // Eigen::Vector3d invsqrt3Vector = (1.0 / sqrt(3)) * Eigen::Vector3d::Ones();

    mtx_trajs_.lock();
    double angle = 3.14;
    if (argmax_prob_collision >= 0)
    {
      Eigen::Vector3d A2G = G_term.pos - A.pos;
      Eigen::Vector3d A2Obstacle = evalMeanDynTrajCompiled(trajs_[argmax_prob_collision], t_start) - A.pos;
      angle = angleBetVectors(A2G, A2Obstacle);
    }

    double angle_deg = angle * 180 / 3.14;

    if (fabs(angle_deg) > par_.angle_deg_focus_front)
    {  //
      std::cout << bold << yellow << "[Selection] Focusing on front of me, angle=" << angle_deg << " deg" << reset
                << std::endl;
      solver_->par_.c_final_yaw = 0.0;
      solver_->par_.c_fov = 0.0;
      solver_->par_.c_yaw_smooth = 0.0;
      solver_->setFocusOnObstacle(false);
      G.yaw = atan2(G_term_.pos[1] - A.pos[1], G_term_.pos[0] - A.pos[0]);
    }
    else
    {
      std::cout << bold << yellow << "[Selection] Focusing on obstacle, angle=" << angle_deg << " deg" << reset
                << std::endl;
      solver_->setFocusOnObstacle(true);
      solver_->par_.c_fov = par_.c_fov;
      solver_->par_.c_final_yaw = par_.c_final_yaw;
      solver_->par_.c_yaw_smooth = par_.c_yaw_smooth;
    }

    mtx_trajs_.unlock();

    //////////////////////////////////////////////////////////////////////////
    ///////////////////////// Set init and final states //////////////////////
    //////////////////////////////////////////////////////////////////////////

    std::cout << "t_start is " << t_start << "\n";
    bool correctInitialCond = solver_->setInitStateFinalStateInitTFinalT(A, G, t_start, t_final);

    if (correctInitialCond == false)
    {
      logAndTimeReplan("Solver cannot guarantee feasibility for v1", false, log);
      return false;
    }

    //
    // Get edges_obstacles
    //

    // mtx_trajs_.lock();

    // ConvexHullsOfCurves hulls = convexHullsOfCurves(t_start, t_final);

    // mtx_trajs_.unlock();

    // edges_obstacles_out = cu::vectorGCALPol2edges(hulls);

    //////////////////////////////////////////////////////////////////////////
    ///////////////////////// Solve optimization! ////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    // removeTrajsThatWillNotAffectMe(A, t_start, t_final);  // TODO: Commented (4-Feb-2021)

    //
    // if there's not obstacle detected, add a dummy one
    //

    std::cout << "obstacles_for_opt.size() " << obstacles_for_opt.size() << std::endl;

    //
    // check if the only trajectory it has is dummy or not
    //

    // if (obstacles_for_opt.size() == 1 && obstacles_for_opt[0].is_dummy)
    // {
    //   obstacles_for_opt.pop_back();
    // }

    if (obstacles_for_opt.size() == 0)
    {
      addDummyObstacle(t_start, t_final, obstacles_for_opt, A, splines_fitted);
    }
    else if (obstacles_for_opt.size() == 1 && obstacles_for_opt[0].is_dummy == true)
    {
      obstacles_for_opt.clear();
      addDummyObstacle(t_start, t_final, obstacles_for_opt, A, splines_fitted);
    }

    solver_->setObstaclesForOpt(obstacles_for_opt);

    //////////////////////
    std::cout << on_cyan << bold << "Solved so far" << solutions_found_ << "/" << total_replannings_ << reset
              << std::endl;

    log_ptr_->tim_initial_setup.toc();
    bool result = solver_->optimize();

    total_replannings_++;
    if (result == false)
    {
      logAndTimeReplan("Solver failed", false, log);
      return false;
    }

    solver_->getPlanes(planes);

    best_solutions_expert = solver_->getBestSolutions();

    n_safe_trajs_expert =
        best_solutions_expert.size();  // Note that all the trajectories produced by the expert are safe

    guesses = solver_->getGuesses();

    best_solution_expert = solver_->fillTrajBestSolutionAndGetIt();
  }
  if (par_.use_student)
  {
    //
    // Get edges_obstacles
    //

    mtx_trajs_.lock();

    ConvexHullsOfCurves hulls = convexHullsOfCurves(t_start, t_final);

    mtx_trajs_.unlock();

    edges_obstacles_out = cu::vectorGCALPol2edges(hulls);

    log_ptr_->tim_initial_setup.toc();
    std::cout << "Calling the student!" << std::endl;
    pybind11::object result = student_caller_ptr_->attr("predict")(A, obstacles_for_opt, G_term.pos);
    std::cout << "Called the student!" << std::endl;
    best_solutions_student = result.cast<std::vector<si::solOrGuess>>();

    pybind11::object tmp = student_caller_ptr_->attr("getIndexBestTraj")();
    int index_smallest_augmented_cost = tmp.cast<int>();

    for (auto z : best_solutions_student)
    {
      if (z.isInCollision() == false)
      {
        n_safe_trajs_student += 1;
      }
    }

    best_solution_student = best_solutions_student[index_smallest_augmented_cost];

    // std::cout << "Chosen cost=" << best_solution_student.augmented_cost << std::endl;

    if (best_solution_student.isInCollision())
    {
      std::cout << red << bold << "All the trajectories found by the student are in collision" << reset << std::endl;

      if (this_replan_uses_new_gterm == true && par_.static_planning == true)
      {
        printInfo(best_solution_student, n_safe_trajs_student);
      }

      return false;
    }

    best_solution_student.fillTraj(par_.dc);  // This could also be done in the predict method of the python class
    best_solution_student.printInfo();
    // abort();
  }

  si::solOrGuess best_solution;

  if (par_.use_student == true && par_.use_expert == false)
  {
    best_solution = best_solution_student;
    n_safe_trajs = n_safe_trajs_student;
  }
  else if (par_.use_student == false && par_.use_expert == true)
  {
    best_solution = best_solution_expert;
    n_safe_trajs = n_safe_trajs_expert;
  }
  else if (par_.use_student == true && par_.use_expert == true)
  {
    best_solution = best_solution_student;
    n_safe_trajs = n_safe_trajs_student;
  }
  else
  {
    std::cout << red << bold << "Either use_student=true or use_expert=true" << reset << std::endl;
    abort();
  }

  solutions_found_++;

  ///////////////////////////////////////////////////////////
  ///////////////       OTHER STUFF    //////////////////////
  //////////////////////////////////////////////////////////

  // Check if we have planned until G_term
  double dist = (G_term_.pos - plan_.back().pos).norm();

  if (dist < par_.goal_radius)
  {
    changeDroneStatus(DroneStatus::GOAL_SEEN);
  }

  planner_initialized_ = true;

  log_ptr_->cost = best_solution.cost;
  log_ptr_->obst_avoidance_violation = best_solution.obst_avoidance_violation;
  log_ptr_->dyn_lim_violation = best_solution.dyn_lim_violation;

  logAndTimeReplan("Success", true, log);

  if ((this_replan_uses_new_gterm == true && par_.static_planning == true) || (par_.static_planning == false))
  {
    printInfo(best_solution, n_safe_trajs);
  }

  return true;
}

bool Panther::addTrajToPlan(const int& k_index_end, mt::log& log, const si::solOrGuess& best_solution,
                            mt::trajectory& X_safe_out)
{
  // std::cout << "Appending to plan" << std::endl;

  //////////////////////////////////////////////////////////////////////////
  ///////////////////////// Append to plan /////////////////////////////////
  //////////////////////////////////////////////////////////////////////////

  mtx_plan_.lock();

  int plan_size = plan_.size();

  if ((plan_size - 1 - k_index_end) < 0)
  {
    std::cout << "plan_size= " << plan_size << std::endl;
    std::cout << "k_index_end= " << k_index_end << std::endl;
    mtx_plan_.unlock();
    logAndTimeReplan("Point A already published", false, log);
    return false;
  }
  else
  {
    plan_.erase(plan_.end() - k_index_end - 1, plan_.end());  // this deletes also the initial condition...

    for (auto& state : best_solution.traj)  //... which is included in best_solution.traj[0]
    {
      plan_.push_back(state);
    }
  }

  mtx_plan_.unlock();

  X_safe_out = plan_.toStdVector();

  return true;
}

//
// --------------------------------------------------------------------------------------------------------------
//

// Checks that I have not received new trajectories that affect me while doing the optimization
bool Panther::safetyCheck(mt::PieceWisePol& pwp)
{
  mtx_trajs_.lock();

  started_check_ = true;

  bool result = true;

  //
  // Check
  //

  for (auto traj : trajs_)
  {
    if (traj.time_received > time_init_opt_ && traj.is_agent == true)
    {
      if (trajsAndPwpAreInCollision(traj, pwp, pwp.times.front(), pwp.times.back()))
      {
        mtx_trajs_.unlock();
        ROS_ERROR_STREAM("Traj collides with " << traj.id);
        result = false;  // will have to redo the optimization
        return result;
      }
    }
  }

  mtx_trajs_.unlock();

  //
  // Recheck
  //

  // and now do another check in case I've received anything while I was checking.
  if (have_received_trajectories_while_checking_ == true)
  {
    ROS_ERROR_STREAM("Recvd traj while checking ");
    result = false;
  }

  started_check_ = false;

  return result;
}

//
// ------------------------------------------------------------------------------------------------------
//

bool Panther::trajsAndPwpAreInCollision(mt::dynTrajCompiled& traj, mt::PieceWisePol& pwp, const double& t_start,
                                        const double& t_end)
{
  Eigen::Vector3d n_i;
  double d_i;
  double deltaT = (t_end - t_start) / (1.0 * par_.num_seg);

  for (int i = 0; i < par_.num_seg; i++)  // for each interval
  {
    // This is my trajectory (no inflation)
    std::vector<Eigen::Vector3d> pointsA =
        vertexesOfInterval(pwp, t_start + i * deltaT, t_start + (i + 1) * deltaT, Eigen::Vector3d::Zero());

    // This is the trajectory of the other agent/obstacle

    std::vector<Eigen::Vector3d> pointsB = vertexesOfInterval(traj, t_start + i * deltaT, t_start + (i + 1) * deltaT);

    if (separator_solver_->solveModel(n_i, d_i, pointsA, pointsB) == false)
    {
      return true;  // There is not a solution --> they collide
    }
  }

  // if reached this point, they don't collide
  return false;
}

//
// ------------------------------------------------------------------------------------------------------
//

std::vector<Eigen::Vector3d> Panther::vertexesOfInterval(mt::dynTrajCompiled& traj, double t_start, double t_end)
{
  Eigen::Vector3d delta = Eigen::Vector3d::Zero();
  Eigen::Vector3d drone_boundarybox = par_.drone_bbox;

  if (traj.is_agent == false)
  {
    std::vector<Eigen::Vector3d> points;
    // delta = traj.bbox / 2.0 + (par_.drone_radius + par_.beta + par_.alpha) *
    //                            Eigen::Vector3d::Ones();  // every side of the box will be increased by 2*delta
    //(+delta on one end, -delta on the other)

    // changeBBox(drone_boundarybox);

    delta = traj.bbox / 2.0 + drone_boundarybox / 2.0;
    // std::cout << "boundary box size" << std::endl;
    // std::cout << drone_boundarybox[0] << std::endl;
    // std::cout << drone_boundarybox[1] << std::endl;
    // std::cout << drone_boundarybox[2] << std::endl;

    // Will always have a sample at the beginning of the interval, and another at the end.
    for (double t = t_start;                           /////////////
         (t < t_end) ||                                /////////////
         ((t > t_end) && ((t - t_end) < par_.gamma));  /////// This is to ensure we have a sample a the end
         t = t + par_.gamma)
    {
      mtx_t_.lock();
      t_ = std::min(t, t_end);  // this min only has effect on the last sample

      double x = traj.s_mean[0].value();
      double y = traj.s_mean[1].value();
      double z = traj.s_mean[2].value();
      mtx_t_.unlock();

      //"Minkowski sum along the trajectory: box centered on the trajectory"
      points.push_back(Eigen::Vector3d(x + delta.x(), y + delta.y(), z + delta.z()));
      points.push_back(Eigen::Vector3d(x + delta.x(), y - delta.y(), z - delta.z()));
      points.push_back(Eigen::Vector3d(x + delta.x(), y + delta.y(), z - delta.z()));
      points.push_back(Eigen::Vector3d(x + delta.x(), y - delta.y(), z + delta.z()));
      points.push_back(Eigen::Vector3d(x - delta.x(), y - delta.y(), z - delta.z()));
      points.push_back(Eigen::Vector3d(x - delta.x(), y + delta.y(), z + delta.z()));
      points.push_back(Eigen::Vector3d(x - delta.x(), y + delta.y(), z - delta.z()));
      points.push_back(Eigen::Vector3d(x - delta.x(), y - delta.y(), z + delta.z()));
    }

    return points;
  }
  else
  {  // is an agent --> use the pwp field

    // delta = traj.bbox / 2.0 + (par_.drone_radius) * Eigen::Vector3d::Ones();
    // delta = traj.bbox / 2.0 + par_.drone_bbox / 2.0;  // instad of using drone_radius

    // changeBBox(drone_boundarybox);

    delta = traj.bbox / 2.0 + drone_boundarybox / 2.0;
    // std::cout << "boundary box size" << std::endl;
    // std::cout << drone_boundarybox[0] << std::endl;
    // std::cout << drone_boundarybox[1] << std::endl;
    // std::cout << drone_boundarybox[2] << std::endl;

    // std::cout << "****traj.bbox = " << traj.bbox << std::endl;
    // std::cout << "****par_.drone_radius = " << par_.drone_radius << std::endl;
    // std::cout << "****Inflation by delta= " << delta.transpose() << std::endl;

    return vertexesOfInterval(traj.pwp_mean, t_start, t_end, delta);
  }
}

//
// ------------------------------------------------------------------------------------------------------
//

std::vector<Eigen::Vector3d> Panther::vertexesOfInterval(mt::PieceWisePol& pwp, double t_start, double t_end,
                                                         const Eigen::Vector3d& delta)
{
  std::vector<Eigen::Vector3d> points;

  std::vector<double>::iterator low = std::lower_bound(pwp.times.begin(), pwp.times.end(), t_start);
  std::vector<double>::iterator up = std::upper_bound(pwp.times.begin(), pwp.times.end(), t_end);

  // // Example: times=[1 2 3 4 5 6 7]
  // // t_start=1.5;
  // // t_end=5.5
  // // then low points to "2" (low - pwp.times.begin() is 1)
  // // and up points to "6" (up - pwp.times.begin() is 5)

  int index_first_interval = low - pwp.times.begin() - 1;  // index of the interval [1,2]
  int index_last_interval = up - pwp.times.begin() - 1;    // index of the interval [5,6]

  saturate(index_first_interval, 0, (int)(pwp.all_coeff_x.size() - 1));
  saturate(index_last_interval, 0, (int)(pwp.all_coeff_x.size() - 1));

  // int index_first_interval = 0;
  // int index_last_interval = static_cast<int>(pwp.coeff_x.size() - 1);

  Eigen::Matrix<double, 3, 4> P;
  Eigen::Matrix<double, 3, 4> V;

  mt::basisConverter basis_converter;
  Eigen::Matrix<double, 4, 4> A_rest_pos_basis_inverse = basis_converter.getArestMinvoDeg3().inverse();

  // push all the complete intervals
  for (int i = index_first_interval; i <= index_last_interval; i++)
  {
    P.row(0) = pwp.all_coeff_x[i];
    P.row(1) = pwp.all_coeff_y[i];
    P.row(2) = pwp.all_coeff_z[i];

    V = P * A_rest_pos_basis_inverse;

    for (int j = 0; j < V.cols(); j++)
    {
      double x = V(0, j);
      double y = V(1, j);
      double z = V(2, j);  //[x,y,z] is the point

      if (delta.norm() < 1e-6)
      {  // no inflation
        points.push_back(Eigen::Vector3d(x, y, z));
      }
      else
      {
        // points.push_back(Eigen::Vector3d(V(1, j), V(2, j), V(3, j)));  // x,y,z

        if (j == V.cols() - 1)
        {
          points.push_back(Eigen::Vector3d(x + delta.x(), y + delta.y(), z + delta.z()));
          points.push_back(Eigen::Vector3d(x + delta.x(), y - delta.y(), z - delta.z()));
          points.push_back(Eigen::Vector3d(x + delta.x(), y + delta.y(), z - delta.z()));
          points.push_back(Eigen::Vector3d(x + delta.x(), y - delta.y(), z + delta.z()));
          points.push_back(Eigen::Vector3d(x - delta.x(), y - delta.y(), z - delta.z()));
          points.push_back(Eigen::Vector3d(x - delta.x(), y + delta.y(), z + delta.z()));
          points.push_back(Eigen::Vector3d(x - delta.x(), y + delta.y(), z - delta.z()));
          points.push_back(Eigen::Vector3d(x - delta.x(), y - delta.y(), z + delta.z()));
        }
        else
        {
          points.push_back(Eigen::Vector3d(x, y, z));
        }
      }
    }
  }

  return points;
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::printInfo(si::solOrGuess& best_solution, int n_safe_trajs)
{
  std::cout << std::setprecision(8)
            << "CostTimeResults: [Cost, obst_avoidance_violation, dyn_lim_violation, total_time_ms, n_safe_trajs]= "
            << best_solution.cost << " " << best_solution.obst_avoidance_violation << " "
            << best_solution.dyn_lim_violation << " " << log_ptr_->tim_total_replan.elapsedSoFarMs() << " "
            << n_safe_trajs << std::endl;
}

void Panther::logAndTimeReplan(const std::string& info, const bool& success, mt::log& log)
{
  log_ptr_->info_replan = info;
  log_ptr_->tim_total_replan.toc();
  log_ptr_->success_replanning = success;

  double total_time_ms = log_ptr_->tim_total_replan.getMsSaved();

  mtx_offsets.lock();
  if (success == false)
  {
    std::cout << bold << red << log_ptr_->info_replan << reset << std::endl;
    int states_last_replan = ceil(total_time_ms / (par_.dc * 1000));  // Number of states that
                                                                      // would have been needed for
                                                                      // the last replan
    deltaT_ = std::max(par_.factor_alpha * states_last_replan, 1.0);
    deltaT_ = std::min(1.0 * deltaT_, 2.0 / par_.dc);
  }
  else
  {
    int states_last_replan = ceil(total_time_ms / (par_.dc * 1000));  // Number of states that
                                                                      // would have been needed for
                                                                      // the last replan
    deltaT_ = std::max(par_.factor_alpha * states_last_replan, 1.0);
  }
  mtx_offsets.unlock();

  log = (*log_ptr_);
}

void Panther::resetInitialization()
{
  planner_initialized_ = false;
  state_initialized_ = false;

  terminal_goal_initialized_ = false;
  plan_.clear();
}

void Panther::yaw(double diff, mt::state& next_goal)
{
  saturate(diff, -par_.dc * par_.ydot_max, par_.dc * par_.ydot_max);
  double dyaw_not_filtered;
  double alpha_filter_dyaw = 0.1;

  dyaw_not_filtered = copysign(1, diff) * par_.ydot_max;

  dyaw_filtered_ = (1 - alpha_filter_dyaw) * dyaw_not_filtered + alpha_filter_dyaw * dyaw_filtered_;
  next_goal.dyaw = dyaw_filtered_;
  next_goal.yaw = previous_yaw_ + dyaw_filtered_ * par_.dc;
  previous_yaw_ = next_goal.yaw;
}

void Panther::getDesiredYaw(mt::state& next_goal)
{
  mt::state current_state;
  getState(current_state);

  double desired_yaw = atan2(G_term_.pos[1] - current_state.pos[1], G_term_.pos[0] - current_state.pos[0]);
  double diff = desired_yaw - current_state.yaw;

  angle_wrap(diff);

  yaw(diff, next_goal);

  if (fabs(diff) < 0.04)
  {
    std::cout << bold << "YAWING IS DONE!!" << std::endl;
    changeDroneStatus(DroneStatus::TRAVELING);
  }
}

bool Panther::getNextGoal(mt::state& next_goal)
{
  if (initializedStateAndTermGoal() == false)  // || (drone_status_ == DroneStatus::GOAL_REACHED && plan_.size() == 1))
                                               // TODO: if included this part commented out, the last state (which is
                                               // the one that has zero accel) will never get published
  {
    // std::cout << "plan_.size() ==" << plan_.size() << std::endl;
    // std::cout << "plan_.content[0] ==" << std::endl;
    // plan_.content[0].print();
    return false;
  }

  mtx_goals.lock();
  mtx_plan_.lock();

  next_goal.setZero();
  next_goal = plan_.front();

  if (plan_.size() > 1)
  {
    if (par_.static_planning == false)
    {
      plan_.pop_front();
    }
  }

  if (drone_status_ == DroneStatus::YAWING)
  {
    getDesiredYaw(next_goal);
  }

  if (par_.mode == "ysweep")
  {
    double t = ros::Time::now().toSec();
    // double T = 1.0;
    double amplitude_deg = 90;
    double amplitude_rd = (amplitude_deg * M_PI / 180);
    next_goal.yaw = amplitude_rd * sin(t * (par_.ydot_max / amplitude_rd));
    next_goal.dyaw = par_.ydot_max * cos(t * (par_.ydot_max / amplitude_rd));
  }

  // if (fabs(next_goal.dyaw) > (par_.ydot_max + 1e-4))
  // {
  //   std::cout << red << "par_.ydot_max not satisfied!!" << reset << std::endl;
  //   std::cout << red << "next_goal.dyaw= " << next_goal.dyaw << reset << std::endl;
  //   std::cout << red << "par_.ydot_max= " << par_.ydot_max << reset << std::endl;
  //   abort();
  // }

  // verify(fabs(next_goal.dyaw) <= par_.ydot_max, "par_.ydot_max not satisfied!!");

  mtx_goals.unlock();
  mtx_plan_.unlock();
  return true;
}

// Debugging functions
void Panther::changeDroneStatus(int new_status)
{
  if (new_status == drone_status_)
  {
    return;
  }

  std::cout << "Changing DroneStatus from ";
  switch (drone_status_)
  {
    case DroneStatus::YAWING:
      std::cout << bold << "YAWING" << reset;
      break;
    case DroneStatus::TRAVELING:
      std::cout << bold << "TRAVELING" << reset;
      break;
    case DroneStatus::GOAL_SEEN:
      std::cout << bold << "GOAL_SEEN" << reset;
      break;
    case DroneStatus::GOAL_REACHED:
      std::cout << bold << "GOAL_REACHED" << reset;
      break;
  }
  std::cout << " to ";

  switch (new_status)
  {
    case DroneStatus::YAWING:
      std::cout << bold << "YAWING" << reset;
      break;
    case DroneStatus::TRAVELING:
      std::cout << bold << "TRAVELING" << reset;
      break;
    case DroneStatus::GOAL_SEEN:
      std::cout << bold << "GOAL_SEEN" << reset;
      break;
    case DroneStatus::GOAL_REACHED:
      std::cout << bold << "GOAL_REACHED" << reset;
      break;
  }

  std::cout << std::endl;

  drone_status_ = new_status;
}

void Panther::printDroneStatus()
{
  switch (drone_status_)
  {
    case DroneStatus::YAWING:
      std::cout << bold << "status_=YAWING" << reset << std::endl;
      break;
    case DroneStatus::TRAVELING:
      std::cout << bold << "status_=TRAVELING" << reset << std::endl;
      break;
    case DroneStatus::GOAL_SEEN:
      std::cout << bold << "status_=GOAL_SEEN" << reset << std::endl;
      break;
    case DroneStatus::GOAL_REACHED:
      std::cout << bold << "status_=GOAL_REACHED" << reset << std::endl;
      break;
  }
}

// Given the control points, this function returns the associated traj and mt::PieceWisePol
void Panther::convertsolOrGuess2pwp(mt::PieceWisePol& pwp_p, si::solOrGuess& solorguess, double dc)
{
  std::vector<Eigen::Vector3d> qp = solorguess.qp;
  std::vector<double> qy = solorguess.qy;
  Eigen::RowVectorXd knots_p = solorguess.knots_p;

  // Right now we use this function only for publishOwnTraj() and it doesn't matter yaw, so we don't have yaw pwp
  // param_pp is degree of position polynomial (p is degree of polynomial (usually p = 3))
  int param_pp = solorguess.deg_p;
  // param_pp is degree of yaw polynomial (p is degree of polynomial (usually p = 2))
  int param_py = solorguess.deg_y;

  assert(((knots_p.size() - 1) == (qp.size() - 1) + param_pp + 1) && "M=N+p+1 not satisfied");

  int num_seg = (knots_p.size() - 1) - 2 * param_pp;  // M-2*p

  // Stack the control points in matrices
  Eigen::Matrix<double, 3, -1> qp_matrix(3, qp.size());
  for (int i = 0; i < qp.size(); i++)
  {
    qp_matrix.col(i) = qp[i];
  }

  Eigen::Matrix<double, 1, -1> qy_matrix(1, qy.size());
  for (int i = 0; i < qy.size(); i++)
  {
    qy_matrix(0, i) = qy[i];
  }

  /////////////////////////////////////////////////////////////////////
  /// CONSTRUCT THE PIECE-WISE POLYNOMIAL FOR POSITION
  /////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double, 4, 4> M;
  M << 1, 4, 1, 0,   //////
      -3, 0, 3, 0,   //////
      3, -6, 3, 0,   //////
      -1, 3, -3, 1;  //////
  M = M / 6.0;       // *1/3!

  pwp_p.clear();

  for (int i = param_pp; i < (param_pp + num_seg + 1); i++)  // i < knots.size() - p
  {
    pwp_p.times.push_back(knots_p(i));
  }

  for (int j = 0; j < num_seg; j++)
  {
    Eigen::Matrix<double, 4, 1> cps_x = (qp_matrix.block(0, j, 1, 4).transpose());
    Eigen::Matrix<double, 4, 1> cps_y = (qp_matrix.block(1, j, 1, 4).transpose());
    Eigen::Matrix<double, 4, 1> cps_z = (qp_matrix.block(2, j, 1, 4).transpose());

    pwp_p.all_coeff_x.push_back((M * cps_x).reverse());  // at^3 + bt^2 + ct + d --> [a b c d]'
    pwp_p.all_coeff_y.push_back((M * cps_y).reverse());  // at^3 + bt^2 + ct + d --> [a b c d]'
    pwp_p.all_coeff_z.push_back((M * cps_z).reverse());  // at^3 + bt^2 + ct + d --> [a b c d]'
  }
}

ConvexHullsOfCurves Panther::convexHullsOfCurves(double t_start, double t_end)
{
  ConvexHullsOfCurves result;

  for (auto traj : trajs_)
  {
    result.push_back(convexHullsOfCurve(traj, t_start, t_end));
  }

  return result;
}

ConvexHullsOfCurve Panther::convexHullsOfCurve(mt::dynTrajCompiled& traj, double t_start, double t_end)
{
  ConvexHullsOfCurve convexHulls;
  double deltaT = (t_end - t_start) / (1.0 * par_.num_seg);  // num_seg is the number of intervals

  for (int i = 0; i <= par_.num_seg; i++)
  {
    convexHulls.push_back(convexHullOfInterval(traj, t_start + i * deltaT, t_start + (i + 1) * deltaT));
  }

  return convexHulls;
}

// See https://doc.cgal.org/Manual/3.7/examples/Convex_hull_3/quickhull_3.cpp
CGAL_Polyhedron_3 Panther::convexHullOfInterval(mt::dynTrajCompiled& traj, double t_start, double t_end)
{
  std::vector<Eigen::Vector3d> points = vertexesOfInterval(traj, t_start, t_end);

  std::vector<Point_3> points_cgal;
  for (auto point_i : points)
  {
    points_cgal.push_back(Point_3(point_i.x(), point_i.y(), point_i.z()));
  }

  return cu::convexHullOfPoints(points_cgal);
}