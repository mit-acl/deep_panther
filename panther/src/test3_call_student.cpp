#include "utils.hpp"
#include "ros_utils.hpp"
#include "panther.hpp"
#include <pybind11/embed.h>
#include <iostream>
#include <panther_types.hpp>
#include <solver_ipopt.hpp>  //TODO: remove this dependency?
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <ros/callback_queue.h>

#include <ros/ros.h>

// namespace py = pybind11;

class Test
{
public:
  Test(ros::NodeHandle nh1)
  {
    nh1_ = nh1;
    runCBTimer_ = nh1_.createTimer(ros::Duration(0.1), &Test::runCB, this);
    runCBTimer_.start();

    pybind11::initialize_interpreter();

    std::string policy_path = "/home/jtorde/Desktop/ws/src/panther_plus_plus/panther_compression/evals/tmp_dagger/1/"
                              "final_policy.pt";

    student_caller_ptr_ = new pybind11::object;
    *student_caller_ptr_ = pybind11::module::import("compression.utils.StudentCaller").attr("StudentCaller")(policy_path);
    std::cout << "Timer has been created" << std::endl;
  };

  ~Test()
  {
    pybind11::finalize_interpreter();
  }
  void runCB(const ros::TimerEvent& e)
  {
    std::cout << "callback has been called!" << std::endl;

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
  ros::Timer runCBTimer_;
  ros::NodeHandle nh1_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test");

  ros::NodeHandle nh1;  //("");

  // Option 1 --> Does not work when using pybind inside the callback (it gets stuck)
  // ros::CallbackQueue custom_queue1;
  // nh1.setCallbackQueue(&custom_queue1);
  // Test a(nh1);  // Note that this needs to be created after calling (setCallbackQueue)
  // ros::AsyncSpinner spinner1(1, &custom_queue1);
  // spinner1.start();
  // ros::waitForShutdown();

  // Option 2 --> Works when using pybind inside the callback
  Test a(nh1);
  ros::spin();

  return 0;
}
