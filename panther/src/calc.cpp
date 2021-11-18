#include <pybind11/embed.h>
#include <iostream>
#include <panther_types.hpp>
#include <solver_ipopt.hpp>  //TODO: remove this dependency?
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;

int main()
{
  // See https://stackoverflow.com/questions/47762543/segfault-when-using-pybind11-wrappers-in-c/47799238
  py::scoped_interpreter guard{};
  /////////////////////////////

  py::module calc = py::module::import("panther.calc");

  std::string policy_path = "/home/jtorde/Desktop/ws/src/panther_plus_plus/panther_compression/evals/tmp_dagger/1/"
                            "final_policy.pt";

  py::object tmp = calc.attr("StudentCaller")(policy_path);

  mt::state A;
  mt::state G;
  double total_time = 10;

  mt::obstacleForOpt obstacle_for_opt;

  obstacle_for_opt.bbox_inflated = Eigen::Vector3d::Zero();

  int fitter_num_seg = 7;
  int fitter_deg_pos = 3;

  for (int i = 0; i < (fitter_num_seg + fitter_deg_pos); i++)
  {
    obstacle_for_opt.ctrl_pts.push_back(Eigen::Vector3d::Zero());
  }

  std::vector<mt::obstacleForOpt> obstacles_for_opt;
  obstacles_for_opt.push_back(obstacle_for_opt);

  Eigen::Vector3d w_g = Eigen::Vector3d::Zero();

  Eigen::Vector3d w_gterm = 10 * Eigen::Vector3d::Ones();

  py::object result = tmp.attr("predict")(A, obstacles_for_opt, w_gterm);

  si::solOrGuess solution = result.cast<si::solOrGuess>();
  solution.printInfo();
}