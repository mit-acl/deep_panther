/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include <casadi/casadi.hpp>

#include "solver_ipopt.hpp"
#include "termcolor.hpp"
#include "bspline_utils.hpp"
#include "ros/ros.h"

#include <unsupported/Eigen/Splines>
#include <iostream>
#include <list>
#include <random>
#include <iostream>
#include <vector>
#include <fstream>

#include <ros/package.h>

using namespace termcolor;

struct PrintSupresser
{
  PrintSupresser(){};
  ~PrintSupresser()
  {
    end();
  };
  void start()
  {
    std::cout.setstate(std::ios_base::failbit);  // https://stackoverflow.com/a/30185095
  }
  void end()
  {
    std::cout.clear();
  }
};

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

std::vector<Eigen::Vector3d> casadiMatrix2StdVectorEigen3d(const casadi::DM &qp_casadi)
{
  std::vector<Eigen::Vector3d> qp;
  for (int i = 0; i < qp_casadi.columns(); i++)
  {
    qp.push_back(Eigen::Vector3d(double(qp_casadi(0, i)), double(qp_casadi(1, i)), double(qp_casadi(2, i))));
  }
  return qp;
}

std::vector<double> casadiMatrix2StdVectorDouble(const casadi::DM &qy_casadi)
{
  return static_cast<std::vector<double>>(qy_casadi);
}

casadi::DM stdVectorEigen3d2CasadiMatrix(const std::vector<Eigen::Vector3d> &qp)
{
  casadi::DM casadi_matrix(3, qp.size());  // TODO: do this just once?
  for (int i = 0; i < casadi_matrix.columns(); i++)
  {
    casadi_matrix(0, i) = qp[i].x();
    casadi_matrix(1, i) = qp[i].y();
    casadi_matrix(2, i) = qp[i].z();
  }
  return casadi_matrix;
}

casadi::DM eigen3d2CasadiMatrix(const Eigen::Vector3d &data)
{
  casadi::DM casadi_matrix(3, 1);

  casadi_matrix(0, 0) = data.x();
  casadi_matrix(1, 0) = data.y();
  casadi_matrix(2, 0) = data.z();

  return casadi_matrix;
}

SolverIpopt::SolverIpopt(const mt::parameters &par)
{
  par_ = par;
  // log_ptr_ = log_ptr;

  // All these values are for the position spline

  si::splineParam sp_tmp(par_.deg_pos, par_.num_seg);
  si::splineParam sy_tmp(par_.deg_yaw, par_.num_seg);

  sp_ = sp_tmp;
  sy_ = sy_tmp;

  ///////////////////////////////////////

  mt::basisConverter basis_converter;
  // basis used for collision
  if (par_.basis == "MINVO")
  {
    basis_ = MINVO;
    M_pos_bs2basis_ = basis_converter.getMinvoDeg3Converters(par_.num_seg);
    M_vel_bs2basis_ = basis_converter.getMinvoDeg2Converters(par_.num_seg);
  }
  else if (par_.basis == "BEZIER")
  {
    basis_ = BEZIER;
    M_pos_bs2basis_ = basis_converter.getBezierDeg3Converters(par_.num_seg);
    M_vel_bs2basis_ = basis_converter.getBezierDeg2Converters(par_.num_seg);
  }
  else if (par_.basis == "B_SPLINE")
  {
    basis_ = B_SPLINE;
    M_pos_bs2basis_ = basis_converter.getBSplineDeg3Converters(par_.num_seg);
    M_vel_bs2basis_ = basis_converter.getBSplineDeg2Converters(par_.num_seg);
  }
  else
  {
    std::cout << red << "Basis " << par_.basis << " not implemented yet" << reset << std::endl;
    std::cout << red << "============================================" << reset << std::endl;
    abort();
  }

  ///////////////////////////////////////
  ///////////////////////////////////////

  // // TODO: if C++14, use std::make_unique instead
  separator_solver_ptr_ = std::unique_ptr<separator::Separator>(new separator::Separator());
  octopusSolver_ptr_ =
      std::unique_ptr<OctopusSearch>(new OctopusSearch(par_.basis, par_.num_seg, par_.deg_pos, par_.alpha_shrink));

  std::string folder = ros::package::getPath("panther") + "/matlab/casadi_generated_files/";
  std::fstream myfile(folder + "index_instruction.txt", std::ios_base::in);
  myfile >> index_instruction_;
  cf_op_ = casadi::Function::load(folder + "op.casadi");
  // cf_op_force_final_pos_ = casadi::Function::load(folder + "op_force_final_pos.casadi");
  cf_fixed_pos_op_ = casadi::Function::load(folder + "op_fixed_pos.casadi");
  cf_fit_yaw_ = casadi::Function::load(folder + "fit_yaw.casadi");
  // cf_fit3d_ = casadi::Function::load(folder + "fit3d.casadi");
  cf_visibility_ = casadi::Function::load(folder + "visibility.casadi");

  b_Tmatrixcasadi_c_ = casadi::DM(4, 4);

  // std::cout << "par_.b_T_c= " << par_.b_T_c << std::endl;

  for (int i = 0; i < par_.b_T_c.rows(); i++)
  {
    for (int j = 0; j < par_.b_T_c.cols(); j++)
    {
      b_Tmatrixcasadi_c_(i, j) = par_.b_T_c(i, j);
    }
  }

  //////////////////////////////////////// CONSTRUCT THE GRAPH FOR THE YAW SEARCH
  ////////////////////////////////////////////////////////////////////////////////

  num_of_yaw_per_layer_ = par_.num_of_yaw_per_layer;
  num_of_layers_ = par_.sampler_num_samples;

  vector_yaw_samples_ = casadi::DM::zeros(1, num_of_yaw_per_layer_);
  for (int j = 0; j < num_of_yaw_per_layer_; j++)
  {
    vector_yaw_samples_(j) = -M_PI + j * 2 * M_PI / num_of_yaw_per_layer_;  // \in [-pi, pi]
  }

  // mygraph_t mygraph_(0);  // start a graph with 0 vertices
  mygraph_.clear();

  // create all the vertexes and add them to the graph
  std::vector<std::vector<vd>> all_vertexes_tmp(num_of_layers_ - 1, std::vector<vd>(num_of_yaw_per_layer_));  // TODO
  all_vertexes_ = all_vertexes_tmp;
  std::vector<vd> tmp(1);  // first layer only one element
  all_vertexes_.insert(all_vertexes_.begin(), tmp);

  // https://stackoverflow.com/questions/47904550/should-i-keep-track-of-vertex-descriptors-in-boost-graph-library

  double y0_tmp = 0.0;  // this value will be updated at the start of each iteration

  // add rest of the vertexes
  for (size_t i = 0; i < num_of_layers_; i++)  // i is the index of each layer
  {
    size_t num_of_circles_layer_i = (i == 0) ? 1 : num_of_yaw_per_layer_;
    for (size_t j = 0; j < num_of_circles_layer_i; j++)  // j is the index of each  circle in the layer i
    {
      vd vertex1 = boost::add_vertex(mygraph_);
      all_vertexes_[i][j] = vertex1;
      mygraph_[vertex1].yaw = (i == 0) ? y0_tmp : double(vector_yaw_samples_(j));
      mygraph_[vertex1].layer = i;
      mygraph_[vertex1].circle = j;
      // mygraph_[vertex1].print();
      // std::cout << "So far, the graph has " << num_vertices(mygraph_) << "vertices" << std::endl;
    }
  }

  for (size_t i = 0; i < (num_of_layers_ - 1); i++)  // i is the number of layers
  {
    size_t num_of_circles_layer_i = (i == 0) ? 1 : num_of_yaw_per_layer_;

    for (size_t j = 0; j < num_of_circles_layer_i; j++)  // j is the circle index of layer i
    {
      for (size_t j_next = 0; j_next < num_of_yaw_per_layer_; j_next++)
      {
        vd index_vertex1 = all_vertexes_[i][j];
        vd index_vertex2 = all_vertexes_[i + 1][j_next];

        // std::cout <<  mygraph_[index_vertex2].layer << ", " << mygraph_[index_vertex2].circle
        //           << std::endl;
        // std::cout <<  i + 1 << ", " << j_next << std::endl;

        edge_descriptor e;
        bool inserted;
        boost::tie(e, inserted) = add_edge(index_vertex1, index_vertex2, mygraph_);
      }
    }
  }

  ////////////////////////////////////////
  ////////////////////////////////////////
}

SolverIpopt::~SolverIpopt()
{
}

void SolverIpopt::getPlanes(std::vector<Hyperplane3D> &planes)
{
  planes = planes_;
}

// void SolverIpopt::setMaxRuntimeKappaAndMu(double max_runtime, double kappa, double mu)
// {
//   kappa_ = kappa;
//   mu_ = mu;
//   max_runtime_ = max_runtime;
// }

// void SolverIpopt::setHulls(ConvexHullsOfCurves_Std &hulls)
// {
//   hulls_.clear();
//   hulls_ = hulls;
//   num_of_obst_ = hulls_.size();
//   num_of_normals_ = par_.num_seg * num_of_obst_;
// }

//////////////////////////////////////////////////////////

bool SolverIpopt::isInCollision(mt::state state, double t)
{
  for (const auto &obstacle_i : obstacles_for_opt_)
  {
    Eigen::RowVectorXd knots_p =
        constructKnotsClampedUniformSpline(t_init_, t_final_guess_, par_.fitter_deg_pos, par_.fitter_num_seg);

    mt::state state_obs = getStatePosSplineT(obstacle_i.ctrl_pts, knots_p, sp_.p, t);

    Eigen::Array<double, 3, 1> distance = (state_obs.pos - state.pos).array().abs();
    Eigen::Vector3d delta = obstacle_i.bbox_inflated / 2.0;

    std::cout << "state_obs.pos= " << state_obs.pos.transpose() << std::endl;
    std::cout << "state.pos= " << state.pos.transpose() << std::endl;
    std::cout << "distance= " << distance.transpose() << std::endl;
    std::cout << "delta= " << delta.transpose() << std::endl;
    // std::cout << "obstacle_i.bbox_inflated= " << obstacle_i.bbox_inflated.transpose() << std::endl;

    if ((distance < delta.array()).all())
    {
      return true;
    }
  }
  return false;
}

void SolverIpopt::setObstaclesForOpt(const std::vector<mt::obstacleForOpt> &obstacles_for_opt)
{
  obstacles_for_opt_ = obstacles_for_opt;

  ////// Set the hulls for use in Octopus Search
  hulls_.clear();

  Eigen::RowVectorXd knots_p =
      constructKnotsClampedUniformSpline(t_init_, t_final_guess_, par_.fitter_deg_pos, par_.fitter_num_seg);

  double deltaT = (t_final_guess_ - t_init_) / (1.0 * par_.num_seg);  // num_seg is the number of intervals

  for (const auto &obstacle_i : obstacles_for_opt_)
  {
    VertexesObstacle vertexes_obstacle_i;

    // std::vector<Eigen::Vector3d> ctrl_pts_obs_i = casadiMatrix2StdVectorEigen3d(obstacle_i.ctrl_pts);

    for (int j = 0; j < par_.num_seg; j++)
    {
      std::vector<double> times =
          linspace(t_init_ + j * deltaT, t_init_ + (j + 1) * deltaT, par_.disc_pts_per_interval_oct_search);

      // std::cout << "times.size()= " << times.size() << std::endl;

      VertexesInterval vertexes_interval_j(3, 8 * times.size());  // For each sample, there are 8 vertexes

      for (int k = 0; k < times.size(); k++)
      {
        // std::cout << "times[k]= " << times[k] << std::endl;

        mt::state state = getStatePosSplineT(obstacle_i.ctrl_pts, knots_p, sp_.p, times[k]);

        Eigen::Vector3d delta = obstacle_i.bbox_inflated / 2.0;

        // clang-format off
         vertexes_interval_j.col(8*k)=     (Eigen::Vector3d(state.pos.x() + delta.x(), state.pos.y() + delta.y(), state.pos.z() + delta.z()));
         vertexes_interval_j.col(8*k+1)=   (Eigen::Vector3d(state.pos.x() + delta.x(), state.pos.y() - delta.y(), state.pos.z() - delta.z()));
         vertexes_interval_j.col(8*k+2)=   (Eigen::Vector3d(state.pos.x() + delta.x(), state.pos.y() + delta.y(), state.pos.z() - delta.z()));
         vertexes_interval_j.col(8*k+3)=   (Eigen::Vector3d(state.pos.x() + delta.x(), state.pos.y() - delta.y(), state.pos.z() + delta.z()));
         vertexes_interval_j.col(8*k+4)=   (Eigen::Vector3d(state.pos.x() - delta.x(), state.pos.y() - delta.y(), state.pos.z() - delta.z()));
         vertexes_interval_j.col(8*k+5)=   (Eigen::Vector3d(state.pos.x() - delta.x(), state.pos.y() + delta.y(), state.pos.z() + delta.z()));
         vertexes_interval_j.col(8*k+6)=   (Eigen::Vector3d(state.pos.x() - delta.x(), state.pos.y() + delta.y(), state.pos.z() - delta.z()));
         vertexes_interval_j.col(8*k+7)=   (Eigen::Vector3d(state.pos.x() - delta.x(), state.pos.y() - delta.y(), state.pos.z() + delta.z()));
        // clang-format on

        // std::cout << "vertexes_interval_j= \n" << vertexes_interval_j << std::endl;
      }

      vertexes_obstacle_i.push_back(vertexes_interval_j);
    }

    hulls_.push_back(vertexes_obstacle_i);
  }

  num_of_obst_ = hulls_.size();
  num_of_normals_ = par_.num_seg * num_of_obst_;

  //////
}

casadi::DM SolverIpopt::eigen2casadi(const Eigen::Vector3d &a)
{
  casadi::DM b = casadi::DM::zeros(3, 1);
  b(0, 0) = a(0);
  b(1, 0) = a(1);
  b(2, 0) = a(2);
  return b;
};

// Note that t_final will be updated in case the saturation in deltaT has had effect
bool SolverIpopt::setInitStateFinalStateInitTFinalT(mt::state initial_state, mt::state final_state, double t_init,
                                                    double &t_final)
{
  ///////////////////////////
  Eigen::Vector3d p0 = initial_state.pos;
  Eigen::Vector3d v0 = initial_state.vel;
  Eigen::Vector3d a0 = initial_state.accel;

  initial_state_ = initial_state;
  final_state_ = final_state;

  initial_state_.yaw = wrapFromMPitoPi(initial_state_.yaw);
  final_state_.yaw = wrapFromMPitoPi(final_state_.yaw);

  /// Now shift final_state_.yaw  so that the difference wrt initial_state_.yaw is <=pi

  double previous_phi = initial_state_.yaw;
  double phi_i = final_state_.yaw;
  double difference = previous_phi - phi_i;

  double phi_i_f = phi_i + floor(difference / (2 * M_PI)) * 2 * M_PI;
  double phi_i_c = phi_i + ceil(difference / (2 * M_PI)) * 2 * M_PI;

  final_state_.yaw = (fabs(previous_phi - phi_i_f) < fabs(previous_phi - phi_i_c)) ? phi_i_f : phi_i_c;

  /// Just for debugging
  if (fabs(initial_state_.yaw - final_state_.yaw) > M_PI)
  {
    std::cout << red << bold << "This diff must be <= pi" << reset << std::endl;
    abort();
  }
  ///

  // std::cout << "initial_state= " << std::endl;
  // initial_state.printHorizontal();

  // std::cout << "final_state= " << std::endl;
  // final_state.printHorizontal();

  //////////////////////////////

  double deltaT = (t_final - t_init) / (1.0 * (sp_.M - 2 * sp_.p - 1 + 1));

  // double old_deltaT = deltaT;

  // //////////////////////////////
  // // Now make sure deltaT in knots_p_guess_ is such that -v_max<=v1<=v_max is satisfied:
  // for (int axis = 0; axis < 3; axis++)
  // {
  //   double upper_bound, lower_bound;
  //   if (fabs(a0(axis)) > 1e-7)
  //   {
  //     upper_bound = ((sp_.p - 1) * (sgn(a0(axis)) * par_.v_max(axis) - v0(axis)) / (a0(axis)));
  //     lower_bound = ((sp_.p - 1) * (-sgn(a0(axis)) * par_.v_max(axis) - v0(axis)) / (a0(axis)));

  //     ////////////////// Debugging
  //     // if (upper_bound < lower_bound)
  //     // {
  //     //   std::cout << red << bold << "This should never happen, aborting" << std::endl;
  //     //   abort();
  //     // }
  //     //////////////////

  //     if (upper_bound <= 0)
  //     {
  //       std::cout << red << bold << "There is no way to satisfy v1" << reset << std::endl;  //(deltat will be zero)
  //       return false;
  //     }

  //     saturate(deltaT, std::max(0.0, lower_bound), upper_bound);
  //   }
  //   else
  //   {
  //     // do nothing: a0 ==0 for that axis, so that means that v1==v0, and therefore v1 satisfies constraints for that
  //     // axis
  //   }
  // }

  // if (old_deltaT != deltaT)
  // {
  //   std::cout << red << bold << "old_deltaT= " << old_deltaT << reset << std::endl;
  //   std::cout << red << bold << "deltaT= " << deltaT << reset << std::endl;
  // }

  // Eigen::Vector3d bound1 = ((p_ - 1) * (par_.v_max - v0).array() / (a0.array()));
  // Eigen::Vector3d bound2 = ((p_ - 1) * (-par_.v_max - v0).array() / (a0.array()));

  // // note that if any element of a0 is ==0.0, then its corresponding element in bound1 (or bound2) is +-infinity,
  // but
  // // valid  for the saturation below

  // saturate(deltaT, std::min(bound1.x(), bound2.x()), std::max(bound1.x(), bound2.x()));
  // saturate(deltaT, std::min(bound1.y(), bound2.y()), std::max(bound1.y(), bound2.y()));
  // saturate(deltaT, std::min(bound1.z(), bound2.z()), std::max(bound1.z(), bound2.z()));

  // std::cout << "std::min(bound1.x(), bound2.x()= " << std::min(bound1.x(), bound2.x()) << std::endl;
  // std::cout << "std::max(bound1.x(), bound2.x()= " << std::max(bound1.x(), bound2.x()) << std::endl;

  // std::cout << "std::min(bound1.y(), bound2.y()= " << std::min(bound1.y(), bound2.y()) << std::endl;
  // std::cout << "std::max(bound1.y(), bound2.y()= " << std::max(bound1.y(), bound2.y()) << std::endl;

  // std::cout << "std::min(bound1.z(), bound2.z()= " << std::min(bound1.z(), bound2.z()) << std::endl;
  // std::cout << "std::max(bound1.z(), bound2.z()= " << std::max(bound1.z(), bound2.z()) << std::endl;

  // std::cout << bold << "deltaT after= " << deltaT << reset << std::endl;

  t_final = t_init + (1.0 * (sp_.M - 2 * sp_.p - 1 + 1)) * deltaT;

  t_init_ = t_init;
  t_final_guess_ = t_final;

  // std::cout << "total_time_guess= " << t_final_guess_ - t_init_ << std::endl;

  return true;
}

std::vector<si::solOrGuess> SolverIpopt::getBestSolutions()
{
  return solutions_;
}

si::solOrGuess SolverIpopt::getBestSolution()
{
  double min_cost = std::numeric_limits<double>::max();
  int argmin = -1;
  for (int i = 0; i < solutions_.size(); i++)
  {
    if (solutions_[i].solver_succeeded && (solutions_[i].cost < min_cost))
    {
      min_cost = solutions_[i].cost;
      argmin = i;
    }
  }

  if (argmin < 0)
  {
    std::cout << bold << red << "Aborting: You called fillTrajBestSololutionAndGetIt after optimize() was false"
              << reset << std::endl;
    abort();
  }

  /// Debugging
  // solutions_[argmin].fillTraj(0.05);

  // for (auto &state_i : solutions_[argmin].traj)
  // {
  //   state_i.printHorizontal();
  // }
  /////////////////////////////////////

  return solutions_[argmin];
}

si::solOrGuess SolverIpopt::fillTrajBestSolutionAndGetIt()
{
  si::solOrGuess best_solution = getBestSolution();

  best_solution.fillTraj(par_.dc);

  // Force last vel and jerk =final_state_ (which it's not guaranteed because of the discretization with par_.dc)
  best_solution.traj.back().vel = final_state_.vel;
  best_solution.traj.back().accel = final_state_.accel;
  best_solution.traj.back().jerk = Eigen::Vector3d::Zero();
  best_solution.traj.back().ddyaw = final_state_.ddyaw;

  return best_solution;
}

std::vector<si::solOrGuess> SolverIpopt::getGuesses()
{
  return guesses_;
}

bool SolverIpopt::optimize(bool supress_all_prints)
{
  PrintSupresser print_supresser;
  if (supress_all_prints)
  {
    print_supresser.start();
  }
  std::cout << "in SolverIpopt::optimize" << std::endl;

  std::cout << "initial_state= " << std::endl;
  initial_state_.printHorizontal();

  std::cout << "final_state= " << std::endl;
  final_state_.printHorizontal();

  std::vector<os::solution> p_guesses;

  // reset some stuff
  solutions_.clear();

  bool guess_found = generateAStarGuess(p_guesses);  // I obtain p_guesses
  if (guess_found == false)
  {
    std::cout << bold << red << "Necessary guesses for pos haven't been found" << reset << std::endl;
    return false;
  }

  int max_num_of_planes = par_.num_max_of_obst * par_.num_seg;
  if ((p_guesses[0].n.size() > max_num_of_planes))
  {
    std::cout << red << bold << "the casadi function does not support so many planes" << reset << std::endl;
    std::cout << red << bold << "you have " << num_of_obst_ << "*" << par_.num_seg << "=" << p_guesses[0].n.size()
              << " planes" << std::endl;
    std::cout << red << bold << "and max is  " << par_.num_max_of_obst << "*" << par_.num_seg << "="
              << max_num_of_planes << " planes" << std::endl;
    return false;
  }

  ////////////////////////////////////
  //////////////////////////////////// CASADI

  // Conversion DM <--> Eigen:  https://github.com/casadi/casadi/issues/2563
  auto eigen2std = [](Eigen::Vector3d &v) { return std::vector<double>{ v.x(), v.y(), v.z() }; };

  std::map<std::string, casadi::DM> map_arguments;
  map_arguments["thetax_FOV_deg"] = par_.fov_x_deg;
  map_arguments["thetay_FOV_deg"] = par_.fov_y_deg;
  map_arguments["b_T_c"] = b_Tmatrixcasadi_c_;
  map_arguments["Ra"] = par_.Ra;
  map_arguments["p0"] = eigen2std(initial_state_.pos);
  map_arguments["v0"] = eigen2std(initial_state_.vel);
  map_arguments["a0"] = eigen2std(initial_state_.accel);
  map_arguments["pf"] = eigen2std(final_state_.pos);
  map_arguments["vf"] = eigen2std(final_state_.vel);
  map_arguments["af"] = eigen2std(final_state_.accel);
  map_arguments["y0"] = initial_state_.yaw;
  map_arguments["yf"] = final_state_.yaw;

  // if (fabs(final_state_.yaw) > 1e-5 || par_.c_final_yaw > 0.0)
  // {
  //   std::cout << red << bold << "Implement this!" << std::endl;
  //   abort();
  // }

  map_arguments["ydot0"] = initial_state_.dyaw;
  map_arguments["ydotf"] =
      final_state_.dyaw;  // Needed: if not (and if you are minimizing ddyaw), ddyaw=cte --> yaw will explode
  map_arguments["v_max"] = eigen2std(par_.v_max);
  map_arguments["a_max"] = eigen2std(par_.a_max);
  map_arguments["j_max"] = eigen2std(par_.j_max);
  map_arguments["ydot_max"] = par_.ydot_max;
  map_arguments["x_lim"] = std::vector<double>{ par_.x_min, par_.x_max };
  map_arguments["y_lim"] = std::vector<double>{ par_.y_min, par_.y_max };
  map_arguments["z_lim"] = std::vector<double>{ par_.z_min, par_.z_max };
  double alpha_guess = (t_final_guess_ - t_init_);
  map_arguments["alpha"] = alpha_guess;  // Initial guess for alpha

  for (int i = 0; i < par_.num_max_of_obst; i++)
  {  // clang-format off
    map_arguments["obs_" + std::to_string(i) + "_ctrl_pts"] = stdVectorEigen3d2CasadiMatrix(obstacles_for_opt_[i].ctrl_pts);
    map_arguments["obs_" + std::to_string(i) + "_bbox_inflated"] = eigen3d2CasadiMatrix(obstacles_for_opt_[i].bbox_inflated);
     // clang-format on
  }

  map_arguments["c_pos_smooth"] = par_.c_pos_smooth;
  map_arguments["c_yaw_smooth"] = par_.c_yaw_smooth;
  map_arguments["c_fov"] = par_.c_fov;
  map_arguments["c_final_pos"] = par_.c_final_pos;
  map_arguments["c_final_yaw"] = par_.c_final_yaw;
  map_arguments["c_total_time"] = par_.c_total_time;
  map_arguments["c_dyn_lim"] = par_.c_dyn_lim;

  /////////////////////////////////////////// SOLVE AN OPIMIZATION FOR EACH OF THE GUESSES FOUND

  std::vector<si::solOrGuess> solutions;
  std::vector<si::solOrGuess> guesses;

  // #pragma omp parallel for
  for (auto p_guess : p_guesses)
  {
    static casadi::DM all_nd(4, max_num_of_planes);
    all_nd = casadi::DM::zeros(4, max_num_of_planes);
    for (int i = 0; i < p_guess.n.size(); i++)
    {
      // The optimized curve is on the side n'x+d <= -1
      // The obstacle is on the side n'x+d >= 1
      all_nd(0, i) = p_guess.n[i].x();
      all_nd(1, i) = p_guess.n[i].y();
      all_nd(2, i) = p_guess.n[i].z();
      all_nd(3, i) = p_guess.d[i];
    }

    map_arguments["all_nd"] = all_nd;

    ///////////////// GUESS FOR POSITION CONTROL POINTS

    casadi::DM matrix_qp_guess = stdVectorEigen3d2CasadiMatrix(p_guess.qp);

    map_arguments["pCPs"] = matrix_qp_guess;

    ////////////////////////////////Generate Yaw Guess
    casadi::DM matrix_qy_guess(1, sy_.N);  // TODO: do this just once?

    matrix_qy_guess = generateYawGuess(matrix_qp_guess, initial_state_.yaw, initial_state_.dyaw, final_state_.dyaw,
                                       t_init_, t_final_guess_);

    map_arguments["yCPs"] = matrix_qy_guess;

    si::solOrGuess guess;
    guess.deg_p = par_.deg_pos;
    guess.deg_y = par_.deg_yaw;
    guess.is_guess = true;
    guess.qp = p_guess.qp;
    guess.qy = casadiMatrix2StdVectorDouble(matrix_qy_guess);
    guess.knots_p = constructKnotsClampedUniformSpline(t_init_, t_final_guess_, sp_.p, sp_.num_seg);
    guess.knots_y = constructKnotsClampedUniformSpline(t_init_, t_final_guess_, sy_.p, sy_.num_seg);

    // for(std::map<std::string, casadi::DM>::const_iterator it = map_arguments.begin();
    //     it != map_arguments.end(); ++it)
    // {
    //     std::cout << it->first << " " << it->second<< "\n";
    // }
    ////////////////////////// CALL THE SOLVER
    std::map<std::string, casadi::DM> result;
    // log_ptr_->tim_opt.tic();

    /////////////////////////////////

    if (par_.mode == "panther" && focus_on_obstacle_ == true)
    {
      map_arguments["c_yaw_smooth"] = par_.c_yaw_smooth;
      map_arguments["c_fov"] = par_.c_fov;
      std::cout << bold << green << "Optimizing for YAW and POSITION!" << reset << std::endl;

      // printMap(map_arguments);

      result = cf_op_(map_arguments);
    }
    else if (par_.mode == "py" && focus_on_obstacle_ == true)
    {
      // first solve for the position spline
      map_arguments["c_yaw_smooth"] = 0.0;
      map_arguments["c_fov"] = 0.0;
      std::cout << bold << green << "Optimizing first for POSITION!" << reset << std::endl;
      result = cf_op_(map_arguments);

      // Use the position control points obtained for solve for yaw. Note that here the pos spline is FIXED
      map_arguments["c_yaw_smooth"] = par_.c_yaw_smooth;
      map_arguments["c_fov"] = par_.c_fov;
      map_arguments["pCPs"] = result["pCPs"];

      std::cout << bold << green << "and then for YAW!" << reset << std::endl;

      std::map<std::string, casadi::DM> result_for_yaw = cf_fixed_pos_op_(map_arguments);

      //////////// Debugging
      if (result["yCPs"].columns() != result_for_yaw["yCPs"].columns())
      {
        std::cout << "Sizes do not match. This is likely because you did not run main.m with both pos_is_fixed=true "
                     "and "
                     "pos_is_fixed=false"
                  << std::endl;
        abort();
      }
      ///////////////////

      result["yCPs"] = result_for_yaw["yCPs"];

      // The costs logged will not be the right ones, so don't use them in this mode
    }
    else if (par_.mode == "noPA" || par_.mode == "ysweep" || focus_on_obstacle_ == false)
    {
      map_arguments["c_yaw_smooth"] = 0.0;
      map_arguments["c_fov"] = 0.0;
      std::cout << bold << green << "Optimizing for POSITION!" << reset << std::endl;
      result = cf_op_(map_arguments);
    }
    else
    {
      std::cout << "Mode not implemented yet. Aborting" << std::endl;
      abort();
    }

    ////////////////////////////

    // log_ptr_->tim_opt.toc();

    ///////////////// GET STATUS FROM THE SOLVER
    // See discussion at https://groups.google.com/g/casadi-users/c/1061E0eVAXM/m/dFHpw1CQBgAJ
    // Inspired from https://gist.github.com/jgillis/9d12df1994b6fea08eddd0a3f0b0737f
    std::string optimstatus =
        std::string(cf_op_.instruction_MX(index_instruction_).which_function().stats(1)["return_status"]);

    //////////////// LOG COSTS OBTAINED
    // log_ptr_->pos_smooth_cost = double(result["pos_smooth_cost"]);
    // log_ptr_->yaw_smooth_cost = double(result["yaw_smooth_cost"]);
    // log_ptr_->fov_cost = double(result["fov_cost"]);
    // log_ptr_->final_pos_cost = double(result["final_pos_cost"]);
    // log_ptr_->final_yaw_cost = double(result["final_yaw_cost"]);

    ///////////////////

    si::solOrGuess solution;
    solution.is_guess = false;
    solution.cost = double(result["total_cost"]);

    ///////////////// DECIDE ACCORDING TO STATUS OF THE SOLVER

    // std::vector<Eigen::Vector3d> qp;  // Solution found (Control points for position)
    // std::vector<double> qy;           // Solution found (Control points for yaw)
    std::cout << "optimstatus= " << optimstatus << std::endl;

    bool success_opt;

    // See names here:
    // https://github.com/casadi/casadi/blob/fadc86444f3c7ab824dc3f2d91d4c0cfe7f9dad5/casadi/interfaces/ipopt/ipopt_interface.cpp
    if (optimstatus == "Solve_Succeeded" || optimstatus == "Solved_To_Acceptable_Level")
    {
      std::cout << green << "IPOPT found a solution" << reset << std::endl;
      // log_ptr_->success_opt = true;
      success_opt = true;
      // copy the solution
      // auto qp_casadi = result["pCPs"];
      // for (int i = 0; i < qp_casadi.columns(); i++)
      // {
      //   qp.push_back(Eigen::Vector3d(double(qp_casadi(0, i)), double(qp_casadi(1, i)), double(qp_casadi(2, i))));
      // }

      solution.qp = casadiMatrix2StdVectorEigen3d(result["pCPs"]);

      solution.knots_p = getKnotsSolution(guess.knots_p, alpha_guess, double(result["alpha"]));
      solution.knots_y = getKnotsSolution(guess.knots_y, alpha_guess, double(result["alpha"]));

      double total_time_solution = (solution.knots_p(solution.knots_p.cols() - 1) - solution.knots_p(0));

      if (total_time_solution > (par_.fitter_total_time + 1e-4))
      {
        std::cout << yellow << bold
                  << "WARNING: total_time_solution>par_.fitter_total_time (visibility/obstacle samples are not taken "
                     "in t>par_.fitter_total_time)"
                  << reset << std::endl;

        std::cout << "total_time_solution= " << total_time_solution << std::endl;
        std::cout << " par_.fitter_total_time= " << par_.fitter_total_time << std::endl;

        std::cout << yellow << bold << "Increase fitter.total_time (or decrease Ra)" << reset << std::endl;

        abort();  // Debugging
      }

      // deltaT is the same one
      double deltaT = total_time_solution / num_of_segments_;

      ///////////////////////////////////
      if (par_.mode == "panther" || par_.mode == "py")
      {
        if (focus_on_obstacle_ == true)
        {
          solution.qy = casadiMatrix2StdVectorDouble(result["yCPs"]);  // static_cast<std::vector<double>>();
        }
        else
        {  // find the yaw spline that goes to final_state_.yaw as fast as possible
          solution.qy = yawCPsToGoToFinalYaw(deltaT);
        }
      }
      else if (par_.mode == "noPA" || par_.mode == "ysweep")
      {  // constant yaw
        // Note that in ysweep, the yaw will be a sinusoidal function, see Panther::getNextGoal
        solution.qy.clear();
        for (int i = 0; i < result["yCPs"].columns(); i++)
        {
          solution.qy.push_back(initial_state_.yaw);
        }
      }
      else
      {
        std::cout << "Mode not implemented yet. Aborting" << std::endl;
        abort();
      }

      ///////////////////////////
      // solution.fillTraj(par_.dc);

      // CPs2Traj(solution.qp, solution.qy, knots_p_solution, knots_y_solution, solution.traj, par_.deg_pos,
      // par_.deg_yaw,
      //          par_.dc);
    }
    else
    {
      std::cout << red << "IPOPT failed to find a solution" << reset << std::endl;
      // log_ptr_->success_opt = false;

      if (isInCollision(initial_state_, t_init_))
      {
        std::cout << bold << red << "The initial state was in collision." << reset << std::endl;
        abort();
      }

      success_opt = false;
      // qp = p_guesses.qp;
      // qy = qy_guess_;
      // TODO: If I want to commit to the guesses, they need to be feasible (right now they aren't
      // because of j_max and yaw_dot_max) For now, let's not commit to them and return false
    }

    solution.solver_succeeded = success_opt;

    ////////////////////////////////////
    //////// Only needed for visualization:

    // guess.fillTraj(par_.dc);

    solutions.push_back(solution);
    guesses.push_back(guess);
    //////////////////////////////////
    //////////////////////////////////

    // TODO: Fill here n and d (if they are included as decision variables)
  }

  for (int i = 0; i < solutions.size(); i++)
  {
    std::cout << bold << "\n===================================" << std::endl;
    std::cout << bold << "=======Guess:" << reset << std::endl;
    guesses[i].printInfo();
    std::cout << bold << "=======Solution:" << reset << std::endl;
    solutions[i].printInfo();
  }

  std::cout << "solutions.size()=" << solutions.size() << std::endl;

  solutions_ = solutions;
  guesses_ = guesses;

  if (anySolutionSucceeded())
  {
    std::cout << "Returning true" << std::endl;
    return true;
  }
  else
  {
    std::cout << "Returning false" << std::endl;
    return false;
  }

  // if (solutions[0].solver_succeeded == true)
  // {
  //   ///////////////// Fill  traj_solution_ and pwp_solution_
  //   traj_solution_.clear();

  //   traj_solution_ = solutions[0].traj;
  //   // pwp_solution_ = solutions[0].pwp;

  //   // Uncomment the following line if you wanna visualize the planes
  //   // fillPlanesFromNDQ(n_, d_, qp);
  //   return true;
  // }
  // else
  // {
  //   return false;
  // }
}

bool SolverIpopt::anySolutionSucceeded()
{
  for (auto &solution : solutions_)
  {
    if (solution.solver_succeeded)
    {
      return true;
    }
  }
  return false;
}

std::vector<double> SolverIpopt::yawCPsToGoToFinalYaw(double deltaT)
{
  std::vector<double> qy;

  double y0 = initial_state_.yaw;
  double yf = final_state_.yaw;
  double ydot0 = initial_state_.dyaw;
  int p = par_.deg_yaw;

  qy.clear();
  qy.push_back(y0);
  qy.push_back(y0 + deltaT * ydot0 / (double(p)));  // y0 and ydot0 fix the second control point

  int num_cps_yaw = par_.num_seg + p;

  for (int i = 0; i < (num_cps_yaw - 3); i++)
  {
    double v_needed = p * (yf - qy.back()) / (p * deltaT);

    saturate(v_needed, -par_.ydot_max, par_.ydot_max);  // Make sure it's within the limits

    double next_qy = qy.back() + (p * deltaT) * v_needed / (double(p));

    qy.push_back(next_qy);
  }

  qy.push_back(qy.back());  // TODO: HERE I'M ASSUMMING FINAL YAW VELOCITY=0 (i.e., final_state_.dyaw==0)

  return qy;
}

Eigen::RowVectorXd SolverIpopt::getKnotsSolution(const Eigen::RowVectorXd &knots_guess, const double alpha_guess,
                                                 const double alpha_solution)
{
  int num_knots = knots_guess.cols();

  // std::cout << "knots_guess= " << knots_guess << std::endl;

  Eigen::RowVectorXd shift = knots_guess(0) * Eigen::RowVectorXd::Ones(1, num_knots);

  // std::cout << "shift= " << shift << std::endl;

  Eigen::RowVectorXd knots_solution = (knots_guess - shift) * (alpha_solution / alpha_guess) + shift;

  // std::cout << "knots_solution= " << knots_solution << std::endl;

  return knots_solution;
}
// void SolverIpopt::getSolution(mt::PieceWisePol &solution)
// {
//   solution = pwp_solution_;
// }

void SolverIpopt::fillPlanesFromNDQ(const std::vector<Eigen::Vector3d> &n, const std::vector<double> &d,
                                    const std::vector<Eigen::Vector3d> &q)
{
  planes_.clear();

  for (int obst_index = 0; obst_index < num_of_obst_; obst_index++)
  {
    for (int i = 0; i < par_.num_seg; i++)
    {
      int ip = obst_index * par_.num_seg + i;  // index plane
      Eigen::Vector3d centroid_hull;
      findCentroidHull(hulls_[obst_index][i], centroid_hull);

      Eigen::Vector3d point_in_plane;

      Eigen::Matrix<double, 3, 4> Qmv, Qbs;  // minvo. each column contains a MINVO control point
      Qbs.col(0) = q[i];
      Qbs.col(1) = q[i + 1];
      Qbs.col(2) = q[i + 2];
      Qbs.col(3) = q[i + 3];

      transformPosBSpline2otherBasis(Qbs, Qmv, i);

      Eigen::Vector3d centroid_cps = Qmv.rowwise().mean();

      // the colors refer to the second figure of
      // https://github.com/mit-acl/separator/tree/06c0ddc6e2f11dbfc5b6083c2ea31b23fd4fa9d1

      // Equation of the red planes is n'x+d == 1
      // Convert here to equation [A B C]'x+D ==0
      double A = n[ip].x();
      double B = n[ip].y();
      double C = n[ip].z();
      double D = d[ip] - 1;

      /////////////////// OPTION 1: point_in_plane = intersection between line  centroid_cps --> centroid_hull
      // bool intersects = getIntersectionWithPlane(centroid_cps, centroid_hull, Eigen::Vector4d(A, B, C, D),
      //                                            point_in_plane);  // result saved in point_in_plane

      //////////////////////////

      /////////////////// OPTION 2: point_in_plane = intersection between line  centroid_cps --> closest_vertex
      double dist_min = std::numeric_limits<double>::max();  // delta_min will contain the minimum distance between
                                                             // the centroid_cps and the vertexes of the obstacle
      int index_closest_vertex = 0;
      for (int j = 0; j < hulls_[obst_index][i].cols(); j++)
      {
        Eigen::Vector3d vertex = hulls_[obst_index][i].col(j);

        double distance_to_vertex = (centroid_cps - vertex).norm();
        if (distance_to_vertex < dist_min)
        {
          dist_min = distance_to_vertex;
          index_closest_vertex = j;
        }
      }

      Eigen::Vector3d closest_vertex = hulls_[obst_index][i].col(index_closest_vertex);

      bool intersects = getIntersectionWithPlane(centroid_cps, closest_vertex, Eigen::Vector4d(A, B, C, D),
                                                 point_in_plane);  // result saved in point_in_plane

      //////////////////////////

      if (intersects == false)
      {
        // TODO: this msg is printed sometimes in Multi-Agent simulations. Find out why
        std::cout << red << "There is no intersection, this should never happen (TODO)" << reset << std::endl;
        continue;  // abort();
      }

      Hyperplane3D plane(point_in_plane, n[i]);
      planes_.push_back(plane);
    }
  }
}

// returns 1 if there is an intersection between the segment P1-P2 and the plane given by coeff=[A B C D]
// (Ax+By+Cz+D==0)  returns 0 if there is no intersection.
// The intersection point is saved in "intersection"
bool SolverIpopt::getIntersectionWithPlane(const Eigen::Vector3d &P1, const Eigen::Vector3d &P2,
                                           const Eigen::Vector4d &coeff, Eigen::Vector3d &intersection)
{
  double A = coeff[0];
  double B = coeff[1];
  double C = coeff[2];
  double D = coeff[3];
  // http://www.ambrsoft.com/TrigoCalc/Plan3D/PlaneLineIntersection_.htm
  double x1 = P1[0];
  double a = (P2[0] - P1[0]);
  double y1 = P1[1];
  double b = (P2[1] - P1[1]);
  double z1 = P1[2];
  double c = (P2[2] - P1[2]);
  double t = -(A * x1 + B * y1 + C * z1 + D) / (A * a + B * b + C * c);

  (intersection)[0] = x1 + a * t;
  (intersection)[1] = y1 + b * t;
  (intersection)[2] = z1 + c * t;

  bool result = (t < 0 || t > 1) ? false : true;  // False if the intersection is with the line P1-P2, not with the
                                                  // segment P1 - P2

  return result;
}

//  casadi::DM all_nd(casadi::Sparsity::dense(4, max_num_of_planes));
// casadi::DM::rand(4, 0);

// std::string getPathName(const std::string &s)
// {
//   char sep = '/';

// #ifdef _WIN32
//   sep = '\\';
// #endif

//   size_t i = s.rfind(sep, s.length());
//   if (i != std::string::npos)
//   {
//     return (s.substr(0, i));
//   }

//   return ("");
// }