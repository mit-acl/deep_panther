/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#ifndef SOLVER_IPOPT_HPP
#define SOLVER_IPOPT_HPP

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <iomanip>  //set precision
#include <iostream>

#include "panther_types.hpp"
#include "utils.hpp"
#include <casadi/casadi.hpp>
#include "timer.hpp"
#include "separator.hpp"
#include "octopus_search.hpp"
#include "bspline_utils.hpp"

#include <decomp_geometry/polyhedron.h>
#include <decomp_geometry/polyhedron.h>

// For the yaw search:
#include <boost/graph/astar_search.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/random.hpp>

typedef PANTHER_timers::Timer MyTimer;

std::vector<Eigen::Vector3d> casadiMatrix2StdVectorEigen3d(const casadi::DM &qp_casadi);
std::vector<double> casadiMatrix2StdVectorDouble(const casadi::DM &qy_casadi);
casadi::DM stdVectorEigen3d2CasadiMatrix(const std::vector<Eigen::Vector3d> &qp);
casadi::DM stdVectorDouble2CasadiRowVector(const std::vector<Eigen::Vector3d> &qp);
casadi::DM eigen3d2CasadiMatrix(const Eigen::Vector3d &data);

namespace si  // Solver Ipopt
{
struct solOrGuess
{
  std::vector<Eigen::Vector3d> qp;  // control points for position
  std::vector<double> qy;           // control points for yaw

  // mt::PieceWisePol pwp;
  Eigen::RowVectorXd knots_p;  // contains time information
  Eigen::RowVectorXd knots_y;  // contains time information

  bool solver_succeeded = false;
  // double cost = std::numeric_limits<double>::max();
  double cost;
  double obst_avoidance_violation = 0.0;
  double dyn_lim_violation = 0.0;
  double aug_cost = 0.0;
  bool is_guess = true;
  // double prob = 1.0;
  bool is_repeated = false;

  int deg_p;
  int deg_y;

  mt::trajectory traj;

  bool isInCollision()
  {
    return (obst_avoidance_violation > 1e-5);
  }

  void printInfo()  // avoid naming it print() [for compatibility with pybind11]
  {
    using namespace termcolor;
    std::cout << "Pos Control points:" << std::endl;

    Eigen::Matrix<double, 3, -1> tmp(3, qp.size());

    for (int i = 0; i < qp.size(); i++)
    {
      tmp.col(i) = qp[i];
    }

    std::cout << red << tmp << reset << std::endl;

    std::cout << "knots_p= " << knots_p << std::endl;

    std::cout << "Yaw Control points:" << std::endl;

    for (auto qy_i : qy)
    {
      std::cout << yellow << qy_i << " ";
    }
    std::cout << reset << std::endl;

    std::cout << "knots_y= " << knots_y << std::endl;

    std::cout << bold << "solver_succeeded= " << solver_succeeded << reset << std::endl;

    if (is_guess == false && solver_succeeded == true)
    {
      std::cout << blue << std::setprecision(5) << "Cost= " << cost << reset << std::endl;
      std::cout << blue << std::setprecision(5) << "obst_avoidance_violation= " << obst_avoidance_violation << reset
                << std::endl;
      std::cout << blue << std::setprecision(5) << "dyn_lim_violation= " << dyn_lim_violation << reset << std::endl;
    }
  }

  void fillTraj(double dc)
  {
    CPs2Traj(qp, qy, knots_p, knots_y, traj, deg_p, deg_y, dc);
  }

  double getTotalTime()
  {
    double total_time_p = knots_p(0, knots_p.cols() - 1) - knots_p(0, 0);
    double total_time_y = knots_y(0, knots_y.cols() - 1) - knots_y(0, 0);
    verify(fabs(total_time_p - total_time_y) < 1e-6, "This solOrGuess has different total time in p and y");
    return total_time_p;
  }

  // casadi::DM getQpAsCasadiMatrix()
  // {
  //   StdVectorEigen3d2CasadiMatrix
  // }
};

struct splineParam
{
  splineParam()
  {
  }

  splineParam(int degree, int num_segments)
  {
    p = degree;
    M = num_segments + 2 * p;
    N = M - p - 1;
    num_seg = num_segments;
  }

  int p;
  int M;
  int N;
  int num_seg;
};

struct novale
{
  std::vector<double> value_vector;
  double value;
  novale(double value_tmp)
  {
    value = value;
    std::cout << "Hi, there!" << std::endl;
  }
  void printVector()
  {
    for (auto value_i : value_vector)
    {
      std::cout << value_i << ", " << std::endl;
    }
  }
};

}  // namespace si

class Fitter
{
public:
  Fitter(const int fitter_num_samples);

  ~Fitter();

  std::vector<Eigen::Vector3d> fit(std::vector<Eigen::Vector3d> &samples);

protected:
private:
  casadi::Function cf_fit3d_;
  int fitter_num_samples_;
};

class ClosedFormYawSolver
{
public:
  ClosedFormYawSolver();

  ~ClosedFormYawSolver();

  std::vector<double> getyCPsfrompCPSUsingClosedForm(std::vector<Eigen::Vector3d> &pCPs, double total_time,
                                                     const std::vector<Eigen::Vector3d> &pCPs_feature, const double y0,
                                                     const double ydot0, const double ydotf);

protected:
private:
  casadi::Function cf_;
};

class SolverIpopt
{
public:
  SolverIpopt(const mt::parameters &par);

  ~SolverIpopt();

  bool optimize(bool supress_all_prints = false);
  bool setInitStateFinalStateInitTFinalT(mt::state initial_state, mt::state final_state, double t_init,
                                         double &t_final);
  void setFocusOnObstacle(bool focus_on_obstacle)
  {
    focus_on_obstacle_ = focus_on_obstacle;
  }
  void setObstaclesForOpt(const std::vector<mt::obstacleForOpt> &obstacles_for_opt);
  mt::parameters par_;
  // mt::trajectory traj_solution_;
  si::solOrGuess getBestSolution();
  std::vector<si::solOrGuess> getBestSolutions();
  std::vector<si::solOrGuess> getGuesses();

  void getPlanes(std::vector<Hyperplane3D> &planes);

  si::solOrGuess fillTrajBestSolutionAndGetIt();
  double computeCost(si::solOrGuess guess);
  double computeDynLimitsConstraintsViolation(si::solOrGuess guess);

  std::string getInfoLastOpt();

protected:
private:
  bool isInCollision(mt::state state, double t);

  bool anySolutionSucceeded();
  int numSolutionsSucceeded();
  std::vector<si::solOrGuess> getOnlySucceeded(std::vector<si::solOrGuess> solutions);
  std::vector<si::solOrGuess> getOnlySucceededAndDifferent(std::vector<si::solOrGuess> solutions);

  std::vector<si::solOrGuess> solutions_;
  std::vector<si::solOrGuess> guesses_;

  int B_SPLINE = 1;  // B-Spline Basis
  int MINVO = 2;     // Minimum volume basis
  int BEZIER = 3;    // Bezier basis

  // https://stackoverflow.com/a/11498248/6057617
  double wrapFromMPitoPi(double x)
  {
    x = fmod(x + M_PI, 2 * M_PI);
    if (x < 0)
      x += 2 * M_PI;
    return x - M_PI;
  }

  std::vector<double> yawCPsToGoToFinalYaw(double deltaT);

  void printMap(const std::map<std::string, casadi::DM> &m)
  {
    for (auto it = m.cbegin(); it != m.cend(); ++it)
    {
      std::cout << it->first << " " << it->second << "\n";
    }
  }

  Eigen::RowVectorXd getKnotsSolution(const Eigen::RowVectorXd &knots_guess, const double alpha_guess,
                                      const double alpha_solution);

  bool getIntersectionWithPlane(const Eigen::Vector3d &P1, const Eigen::Vector3d &P2, const Eigen::Vector4d &coeff,
                                Eigen::Vector3d &intersection);

  void addObjective();
  void addConstraints();

  void saturateQ(std::vector<Eigen::Vector3d> &q);

  // transform functions (with Eigen)
  void transformPosBSpline2otherBasis(const Eigen::Matrix<double, 3, 4> &Qbs, Eigen::Matrix<double, 3, 4> &Qmv,
                                      int interval);
  void transformVelBSpline2otherBasis(const Eigen::Matrix<double, 3, 3> &Qbs, Eigen::Matrix<double, 3, 3> &Qmv,
                                      int interval);

  void generateRandomGuess();
  bool generateAStarGuess(std::vector<os::solution> &p_guesses);
  // bool generateStraightLineGuess(std::vector<os::solution> &p_guesses);

  void printStd(const std::vector<Eigen::Vector3d> &v);
  void printStd(const std::vector<double> &v);
  void generateGuessNDFromQ(const std::vector<Eigen::Vector3d> &q, std::vector<Eigen::Vector3d> &n,
                            std::vector<double> &d);

  void fillPlanesFromNDQ(const std::vector<Eigen::Vector3d> &n, const std::vector<double> &d,
                         const std::vector<Eigen::Vector3d> &q);

  void generateRandomD(std::vector<double> &d);
  void generateRandomN(std::vector<Eigen::Vector3d> &n);
  void generateRandomQ(std::vector<Eigen::Vector3d> &q);

  std::map<std::string, casadi::DM> getMapConstantArguments();

  void printQVA(const std::vector<Eigen::Vector3d> &q);

  void printQND(std::vector<Eigen::Vector3d> &q, std::vector<Eigen::Vector3d> &n, std::vector<double> &d);

  void findCentroidHull(const Polyhedron_Std &hull, Eigen::Vector3d &centroid);

  casadi::DM generateYawGuess(casadi::DM matrix_qp_guess, double y0, double ydot0, double ydotf, double t0, double tf);

  std::vector<Eigen::Vector3d> n_;  // Each n_[i] has 3 elements (nx,ny,nz)
  std::vector<double> d_;           // d_[i] has 1 element

  // mt::PieceWisePol pwp_solution_;

  int basis_ = B_SPLINE;

  si::splineParam sp_;
  si::splineParam sy_;

  double index_instruction_;

  int num_of_normals_;

  int num_of_obst_;
  int num_of_segments_;

  std::vector<Hyperplane3D> planes_;

  std::vector<mt::obstacleForOpt> obstacles_for_opt_;

  double t_init_;
  double t_final_guess_;

  mt::state initial_state_;
  mt::state final_state_;

  ConvexHullsOfCurves_Std hulls_;

  MyTimer opt_timer_;

  double max_runtime_ = 2;  //[seconds]

  // transformation between the B-spline control points and other basis
  std::vector<Eigen::Matrix<double, 4, 4>> M_pos_bs2basis_;
  std::vector<Eigen::Matrix<double, 3, 3>> M_vel_bs2basis_;
  std::vector<Eigen::Matrix<double, 4, 4>> A_pos_bs_;

  std::unique_ptr<separator::Separator> separator_solver_ptr_;
  std::unique_ptr<OctopusSearch> octopusSolver_ptr_;

  casadi::Function cf_op_;
  // casadi::Function cf_op_force_final_pos_;
  casadi::Function cf_fixed_pos_op_;
  casadi::Function cf_fit_yaw_;
  casadi::Function cf_visibility_;
  casadi::Function cf_compute_cost_;
  casadi::Function cf_compute_dyn_limits_constraints_violation_;

  casadi::DM b_Tmatrixcasadi_c_;
  struct data
  {
    float yaw;
    size_t layer;
    size_t circle;

    void print()
    {
      std::cout << "yaw= " << yaw << ", layer= " << layer << std::endl;
    }
  };

  typedef float cost_graph;

  ///////////////////////////////// Things for the yaw search
  // specify some types
  // typedef adjacency_list<listS, vecS, undirectedS, no_property, property<edge_weight_t, cost_graph>> mygraph_t;
  typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, data,
                                boost::property<boost::edge_weight_t, cost_graph>>
      mygraph_t;
  typedef boost::property_map<mygraph_t, boost::edge_weight_t>::type WeightMap;
  typedef mygraph_t::vertex_descriptor vd;
  typedef mygraph_t::edge_descriptor edge_descriptor;
  // typedef std::pair<int, int> edge;

  mygraph_t mygraph_;

  double num_of_yaw_per_layer_;  // = par_.num_of_yaw_per_layer;
  double num_of_layers_;         // = par_.num_samples_simpson;
  // WeightMap weightmap_;
  std::vector<std::vector<vd>> all_vertexes_;
  casadi::DM vector_yaw_samples_;

  // std::shared_ptr<mt::log> log_ptr_;

  casadi::DM eigen2casadi(const Eigen::Vector3d &a);

  bool focus_on_obstacle_;

  std::string info_last_opt_ = "";

  // std::vector<os::solution> p_guesses_;

  // casadi::DM fitter_ctrl_pts_;

  // std::unique_ptr<mygraph_t> mygraph_ptr;
  //////////////////////////////////

  // PImpl idiom
  // https://www.geeksforgeeks.org/pimpl-idiom-in-c-with-examples/
  // struct PImpl;
  // std::unique_ptr<PImpl> m_casadi_ptr_;  // Opaque pointer

  // double Ra_ = 1e10;
};

// struct SolverIpopt::PImpl  // TODO: Not use PImpl
// {
//   casadi::Function casadi_function_;
//   casadi::DM all_w_fe_;
// };

#endif