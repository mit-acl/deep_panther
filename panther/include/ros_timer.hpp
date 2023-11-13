#ifndef ROS_TIMER_HPP_
#define ROS_TIMER_HPP_

#include <ros/ros.h>

namespace PANTHER_timers {

class ROSTimer
{
public:
  ROSTimer(bool run = false)
  {
    if (run)
      tic();
  }
  void tic()
  {
    start_ = ros::Time::now().toSec();
  }
  double elapsedSoFarMs() const
  {
    return 1000 * (ros::Time::now().toSec() - start_);
  }
  template <typename T, typename Traits>
  friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, const ROSTimer& timer)
  {
    return out << timer.elapsedSoFarMs();
  }

private:
  double start_;
};

class ROSWallTimer
{
public:
  ROSWallTimer(bool run = false)
  {
    if (run)
      tic();
  }
  void tic()
  {
    start_ = ros::WallTime::now().toSec();
  }
  double elapsedSoFarMs() const
  {
    return 1000 * (ros::WallTime::now().toSec() - start_);
  }
  template <typename T, typename Traits>
  friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, const ROSWallTimer& timer)
  {
    return out << timer.elapsedSoFarMs();
  }

private:
  double start_;
};
}  // namespace PANTHER_timers

#endif
