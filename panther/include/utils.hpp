/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#ifndef UTILS_HPP
#define UTILS_HPP

#include "panther_types.hpp"
#include <deque>
#include <iterator>
#include <casadi/casadi.hpp>

#ifndef __has_include
  static_assert(false, "__has_include not supported");
#else
#  if __cplusplus >= 201703L && __has_include(<filesystem>)
#    include <filesystem>
     namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
     namespace fs = std::experimental::filesystem;
#  elif __has_include(<boost/filesystem.hpp>)
#    include <boost/filesystem.hpp>
     namespace fs = boost::filesystem;
#  endif
#endif

// #define STATE 0
// #define INPUT 1

// #define POS 0
// #define VEL 1
// #define ACCEL 2
// #define JERK 3

// #define WHOLE_TRAJ 0
// #define RESCUE_PATH 1

// #define OCCUPIED_SPACE 1

// #define UNKOWN_AND_OCCUPIED_SPACE 2

std::vector<Eigen::VectorXd> dynamicVecOfVec(std::vector<Eigen::VectorXd> &v);
std::vector<Eigen::VectorXd> dynamicVecOfVec(std::vector<Eigen::Vector2d> &v);
std::vector<Eigen::VectorXd> dynamicVecOfVec(std::vector<Eigen::Vector3d> &v);

// TODO: put this in a namespace

// https://stackoverflow.com/questions/27028226/python-linspace-in-c
template <typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{
  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0)
  {
    return linspaced;
  }
  if (num == 1)
  {
    linspaced.push_back(start);
    return linspaced;
  }

  double delta = (end - start) / (num - 1);

  for (int i = 0; i < num - 1; ++i)
  {
    linspaced.push_back(start + delta * i);
  }
  linspaced.push_back(end);  // I want to ensure that start and end
                             // are exactly the same as the input
  return linspaced;
}

void verify(bool cond, std::string info_if_false);

mt::PieceWisePol composePieceWisePol(const double t, const double dc, mt::PieceWisePol& p1, mt::PieceWisePol& p2);

std::vector<std::string> pieceWisePol2String(const mt::PieceWisePol& pwp);

double getMinTimeDoubleIntegrator1D(const double& p0, const double& v0, const double& pf, const double& vf,
                                    const double& v_max, const double& a_max);

double getMinTimeDoubleIntegratorND(const Eigen::VectorXd& p0, const Eigen::VectorXd& v0, const Eigen::VectorXd& pf,
                                    const Eigen::VectorXd& vf, const Eigen::VectorXd& v_max,
                                    const Eigen::VectorXd& a_max);

double getMinTimeDoubleIntegratorNDFromState(mt::state initial_state, mt::state final_state,
                                             const Eigen::VectorXd& v_max, const Eigen::VectorXd& a_max);

bool boxIntersectsSphere(Eigen::Vector3d center, double r, Eigen::Vector3d c1, Eigen::Vector3d c2);

void printStateDeque(std::deque<mt::state>& data);

void printStateVector(std::vector<mt::state>& data);

void saturate(int& var, const int min, const int max);

void saturate(double& var, const double min, const double max);

void saturate(Eigen::VectorXd& tmp, const Eigen::VectorXd& min, const Eigen::VectorXd& max);

double angleBetVectors(const Eigen::VectorXd& a, const Eigen::VectorXd& b);

void angle_wrap(double& diff);

std::vector<double> eigen2std(const Eigen::Vector2d &v);

std::vector<double> eigen2std(const Eigen::Vector3d &v);

std::vector<double> eigen2std(const Eigen::VectorXd &v);

int nChoosek(int n, int k);

void linearTransformPoly(const Eigen::VectorXd& coeff_old, Eigen::VectorXd& coeff_new, double a, double b);

void changeDomPoly(const Eigen::VectorXd& coeff_p, double tp1, double tp2, Eigen::VectorXd& coeff_q, double tq1,
                   double tq2);
// sign function
template <typename T>
int sign(T val)
{
  return (T(0) < val) - (val < T(0));
}

double cdfUnivariateNormalDist(double x, double mu, double std_deviation);

double probUnivariateNormalDistAB(double a, double b, double mu, double std_deviation);

double probMultivariateNormalDist(const Eigen::VectorXd& a, const Eigen::VectorXd& b, const Eigen::VectorXd& mu,
                                  const Eigen::VectorXd& std_deviation);

// Overload to be able to print a std::vector
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v)
{
  if (!v.empty())
  {
    out << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

casadi::DM checkDimensions(casadi::DM matrix, int dim);

std::vector<Eigen::VectorXd> checkDimensions(std::vector<Eigen::VectorXd> matrix, int dim);

std::vector<Eigen::VectorXd> checkDimensions(std::vector<Eigen::Vector3d> matrix, int dim);

std::vector<Eigen::VectorXd> checkDimensions(std::vector<Eigen::Vector2d> matrix, int dim);

std::string casadi_folder();

#endif