#include <pybind11/embed.h>
#include <iostream>
#include <panther_types.hpp>
#include <solver_ipopt.hpp>  //TODO: remove this dependency?
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;
class Test
{
public:
  Test()
  {
    pybind11::initialize_interpreter();

    std::string policy_path = "/home/jtorde/Desktop/ws/src/panther_plus_plus/panther_compression/evals/tmp_dagger/1/"
                              "final_policy.pt";

    student_caller_ptr_ = new pybind11::object;
    *student_caller_ptr_ = pybind11::module::import("compression.utils.StudentCaller").attr("StudentCaller")(policy_path);
  };

  ~Test()
  {
    pybind11::finalize_interpreter();
  }
  void run()
  {
    mt::state A;
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
    Eigen::Vector3d w_gterm = 10 * Eigen::Vector3d::Ones();

    pybind11::object result = student_caller_ptr_->attr("predict")(A, obstacles_for_opt, w_gterm);

    si::solOrGuess solution = result.cast<si::solOrGuess>();
    solution.printInfo();
  };

private:
  pybind11::object* student_caller_ptr_;
};

int main()
{
  // pybind11::initialize_interpreter();
  std::cout << "In 3" << std::endl;

  Test a;

  std::cout << "In 4" << std::endl;

  a.run();
  a.run();
}

// More info here:
// https://stackoverflow.com/a/51069948
// https://stackoverflow.com/a/54770779

//// WORKS:
// int main()
// {
//   pybind11::scoped_interpreter guard{};
//   pybind11::module calc = pybind11::module::import("compression.utils.other");
// }

//// WORKS:
// int main()
// {
// pybind11::initialize_interpreter();
// {
//   pybind11::module calc = pybind11::module::import("compression.utils.other");
// }
// pybind11::finalize_interpreter();
// }

//// Crashes: (because "calc" is still alive when we call pybind11::finalize_interpreter)
// int main()
// {
// pybind11::initialize_interpreter();
//   pybind11::module calc = pybind11::module::import("compression.utils.other");
// pybind11::finalize_interpreter();
// }