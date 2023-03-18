/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#pragma once
#ifndef PANTHER_HPP
#define PANTHER_HPP

#include <vector>
#include <mutex>
#include <numeric>
#include <algorithm>

#include "cgal_utils.hpp"

#include "panther_types.hpp"

#include "solver_ipopt.hpp"

// status_ : YAWING-->TRAVELING-->GOAL_SEEN-->GOAL_REACHED-->YAWING-->TRAVELING-->...

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

using namespace termcolor;

class Panther
{
public:
  Panther(mt::parameters par);
  ~Panther();
  bool replan(mt::Edges& edges_obstacles_out, si::solOrGuess& best_solution_expert,
              std::vector<si::solOrGuess>& best_solutions_expert, si::solOrGuess& best_solution_student,
              std::vector<si::solOrGuess>& best_solutions_student, std::vector<si::solOrGuess>& guesses,
              std::vector<si::solOrGuess>& splines_fitted, std::vector<Hyperplane3D>& planes, mt::log& log,
              int& k_index_end);
  void pubObstacleEdge(mt::Edges& edges_obstacles_out);
  bool addTrajToPlan(const int& k_index_end, mt::log& log, const si::solOrGuess& best_solution,
                     mt::trajectory& X_safe_out);
  bool safetyCheck(mt::PieceWisePol& pwp);
  bool trajsAndPwpAreInCollision(mt::dynTrajCompiled& traj, mt::PieceWisePol& pwp, const double& t_start,
                                 const double& t_end);
  std::vector<Eigen::Vector3d> vertexesOfInterval(mt::PieceWisePol& pwp, double t_start, double t_end,
                                                  const Eigen::Vector3d& delta);
  std::vector<Eigen::Vector3d> vertexesOfInterval(mt::dynTrajCompiled& traj, double t_start, double t_end);

  void updateState(mt::state data);

  bool getNextGoal(mt::state& next_goal);
  void getG(mt::state& G);
  void getState(mt::state& data);
  void getG_term(mt::state& data);
  void setTerminalGoal(mt::state& term_goal);
  void resetInitialization();

  void yaw(double diff, mt::state& next_goal);
  void getDesiredYaw(mt::state& next_goal);

  bool IsTranslating();
  void updateTrajObstacles(mt::dynTraj traj);
  void convertsolOrGuess2pwp(mt::PieceWisePol& pwp_p, si::solOrGuess& solorguess, double dc);

private:
  // pybind11::module calc_;
  // pybind11::scoped_interpreter guard{};
  // pybind11::object student_caller_;
  // pybind11::module calc_;
  // pybind11::detail::str_attr_accessor tmp_;
  // std::shared_ptr<pybind11::scoped_interpreter> guard_ptr_;

  pybind11::object* student_caller_ptr_;

  // pybind11::scoped_interpreter guard;

  // mt::state M_;
  mt::committedTrajectory plan_;

  ConvexHullsOfCurves convexHullsOfCurves(double t_start, double t_end);
  ConvexHullsOfCurve convexHullsOfCurve(mt::dynTrajCompiled& traj, double t_start, double t_end);
  CGAL_Polyhedron_3 convexHullOfInterval(mt::dynTrajCompiled& traj, double t_start, double t_end);

  std::vector<mt::obstacleForOpt> getObstaclesForOpt(double t_start, double t_end,
                                                     std::vector<si::solOrGuess>& splines_fitted);
  void addDummyObstacle(double t_start, double t_end, std::vector<mt::obstacleForOpt>& obstacles_for_opt, mt::state& A,
                        std::vector<si::solOrGuess>& splines_fitted);

  Eigen::Vector3d evalMeanDynTrajCompiled(const mt::dynTrajCompiled& traj, double t);
  Eigen::Vector3d evalVarDynTrajCompiled(const mt::dynTrajCompiled& traj, double t);

  bool isReplanningNeeded();

  void logAndTimeReplan(const std::string& info, const bool& success, mt::log& log);

  void dynTraj2dynTrajCompiled(const mt::dynTraj& traj, mt::dynTrajCompiled& traj_compiled);

  bool initializedStateAndTermGoal();

  // bool safetyCheckAfterOpt(mt::PieceWisePol pwp_optimized);

  // bool trajsAndPwpAreInCollision(mt::dynTrajCompiled traj, mt::PieceWisePol pwp_optimized, double t_start,
  //                                double t_end);

  // void removeTrajsThatWillNotAffectMe(const mt::state& A, double t_start, double t_end);

  void updateInitialCond(int i);

  void changeDroneStatus(int new_status);

  bool appendToPlan(int k_end_whole, const std::vector<mt::state>& whole, int k_safe,
                    const std::vector<mt::state>& safe);

  bool initialized();
  bool initializedAllExceptPlanner();

  void printDroneStatus();

  void sampleFeaturePos(int argmax_prob_collision, double t_start, double t_end, std::vector<Eigen::Vector3d>& pos);

  void removeOldTrajectories();

  void doStuffTermGoal();

  void printInfo(si::solOrGuess& best_solution, int n_safe_trajs);

  mt::parameters par_;

  double t_;  // variable where the expressions of the trajs of the dyn obs are evaluated

  std::mutex mtx_trajs_;
  std::vector<mt::dynTrajCompiled> trajs_;

  bool state_initialized_ = false;
  bool planner_initialized_ = false;

  int deltaT_ = 75;

  bool terminal_goal_initialized_ = false;

  int drone_status_ = DroneStatus::TRAVELING;  // status_ can be TRAVELING, GOAL_SEEN, GOAL_REACHED
  int planner_status_ = PlannerStatus::FIRST_PLAN;

  std::mutex mtx_goals;

  std::mutex mtx_k;

  std::mutex mtx_planner_status_;
  std::mutex mtx_initial_cond;
  std::mutex mtx_state_;
  std::mutex mtx_offsets;
  std::mutex mtx_plan_;
  // std::mutex mtx_factors;

  std::mutex mtx_G_term;
  std::mutex mtx_t_;

  // casadi::Function cf_fit3d_;
  mt::state stateA_;  // It's the initial condition for the solver

  mt::state state_;
  mt::state G_;       // This goal is always inside of the map
  mt::state G_term_;  // This goal is the clicked goal

  int solutions_found_ = 0;
  int total_replannings_ = 0;

  mt::PieceWisePol pwp_prev_;

  bool exists_previous_pwp_ = false;

  bool started_check_ = false;

  bool have_received_trajectories_while_checking_ = false;

  double time_init_opt_;

  double av_improvement_nlopt_ = 0.0;

  SolverIpopt* solver_;  // pointer to the optimization solver
  Fitter* fitter_;

  Eigen::Matrix<double, 2, 2> A_basis_deg1_rest_;
  Eigen::Matrix<double, 2, 2> A_basis_deg1_rest_inverse_;

  Eigen::Matrix<double, 3, 3> A_basis_deg2_rest_;
  Eigen::Matrix<double, 3, 3> A_basis_deg2_rest_inverse_;

  Eigen::Matrix<double, 4, 4> A_basis_deg3_rest_;
  Eigen::Matrix<double, 4, 4> A_basis_deg3_rest_inverse_;

  separator::Separator* separator_solver_;

  std::shared_ptr<mt::log> log_ptr_;

  mt::state last_state_tracked_;

  bool need_to_do_stuff_term_goal_ = false;
  bool is_new_g_term_ = false;
  double dyaw_filtered_ = 0.0;
  double previous_yaw_ = 0.0;
};

#endif