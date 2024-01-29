#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>

namespace PANTHER_timers
{
class Timer
{
  typedef std::chrono::high_resolution_clock high_resolution_clock;
  typedef std::chrono::milliseconds milliseconds;
  typedef std::chrono::microseconds microseconds;
  typedef std::chrono::nanoseconds nanoseconds;

public:
  Timer(bool run = false)
  {
    if (run)
    {
      tic();
    }
  }
  void tic()
  {
    start_ = high_resolution_clock::now();
  }
  double elapsedSoFarMs() const
  {
    return (std::chrono::duration_cast<nanoseconds>(high_resolution_clock::now() - start_)).count() / (1e6);
  }
  void toc()
  {
    end_ = high_resolution_clock::now();
  }
  double getMsSaved()
  {
    return (std::chrono::duration_cast<nanoseconds>(end_ - start_)).count() / (1e6);
  }
  void reset()
  {
    start_ = high_resolution_clock::now();
    // end_ = high_resolution_clock::now();
  }
  // double elapsedUs() const
  // {
  //   return (std::chrono::duration_cast<microseconds>(high_resolution_clock::now() - start_)).count();
  // }
  template <typename T, typename Traits>
  friend std::basic_ostream<T, Traits>& operator<<(std::basic_ostream<T, Traits>& out, const Timer& timer)
  {
    return out << " " << timer.elapsedSoFarMs() << " ms ";
  }

private:
  high_resolution_clock::time_point start_;
  high_resolution_clock::time_point end_;
};
} // namesapce PANTHER_timers

#endif  // TIMER_HPP_