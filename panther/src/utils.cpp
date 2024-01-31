/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */
#include "utils.hpp"
#include <casadi/casadi.hpp>

#include "termcolor.hpp"

// https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
double cdfUnivariateNormalDist(double x, double mu, double std_deviation)
{
  return 0.5 * (1 + erf((x - mu) / (std_deviation * sqrt(2))));
}

double getMinTimeDoubleIntegrator1D(const double& p0, const double& v0, const double& pf, const double& vf,
                                    const double& v_max, const double& a_max)
{
  // See also minimum_time.m
  // %The notation of this function is based on the paper "Constrained time-optimal
  // %control of double integrator system and its application in MPC"
  // % https://iopscience.iop.org/article/10.1088/1742-6596/783/1/012024

  double x1 = v0;
  double x2 = p0;
  double x1r = vf;
  double x2r = pf;

  double k1 = a_max;  // Note that the paper uses u\in[-1, 1].But setting k1 to a_max has the same effect k2 = 1.0;
  double k2 = 1.0;

  double x1_bar = v_max;

  double B = (k2 / (2 * k1)) * sign(-x1 + x1r) * (pow(x1, 2) - pow(x1r, 2)) + x2r;
  double C = (k2 / (2 * k1)) * (pow(x1, 2) + pow(x1r, 2)) - (k2 / k1) * pow(x1_bar, 2) + x2r;
  double D = (-k2 / (2 * k1)) * (pow(x1, 2) + pow(x1r, 2)) + (k2 / k1) * pow(x1_bar, 2) + x2r;

  double time;

  if ((x2 <= B) && (x2 >= C))
  {
    time = (-k2 * (x1 + x1r) +
            2 * sqrt(pow(k2, 2) * pow(x1, 2) - k1 * k2 * ((k2 / (2 * k1)) * (pow(x1, 2) - pow(x1r, 2)) + x2 - x2r))) /
           (k1 * k2);
  }

  else if ((x2 <= B) && (x2 < C))
  {
    time = (x1_bar - x1 - x1r) / k1 + (pow(x1, 2) + pow(x1r, 2)) / (2 * k1 * x1_bar) + (x2r - x2) / (k2 * x1_bar);
  }

  else if ((x2 > B) && (x2 <= D))
  {
    time = (k2 * (x1 + x1r) +
            2 * sqrt(pow(k2, 2) * pow(x1, 2) + k1 * k2 * ((k2 / (2 * k1)) * (-pow(x1, 2) + pow(x1r, 2)) + x2 - x2r))) /
           (k1 * k2);
  }

  else
  {  // (x2 > B) && (x2 > D)

    time = (x1_bar + x1 + x1r) / k1 + (pow(x1, 2) + pow(x1r, 2)) / (2 * k1 * x1_bar) + (-x2r + x2) / (k2 * x1_bar);
  }

  return time;
}

double getMinTimeDoubleIntegrator3D(const Eigen::Vector3d& p0, const Eigen::Vector3d& v0, const Eigen::Vector3d& pf,
                                    const Eigen::Vector3d& vf, const Eigen::Vector3d& v_max,
                                    const Eigen::Vector3d& a_max)
{
  double min_x = getMinTimeDoubleIntegrator1D(p0.x(), v0.x(), pf.x(), vf.x(), v_max.x(), a_max.x());
  double min_y = getMinTimeDoubleIntegrator1D(p0.y(), v0.y(), pf.y(), vf.y(), v_max.y(), a_max.y());
  double min_z = getMinTimeDoubleIntegrator1D(p0.z(), v0.z(), pf.z(), vf.z(), v_max.z(), a_max.z());

  double min_time = std::max({ min_x, min_y, min_z });  // Note that it's the maximum of all the axes

  return min_time;
}

double getMinTimeDoubleIntegrator3DFromState(mt::state initial_state, mt::state final_state,
                                             const Eigen::Vector3d& v_max, const Eigen::Vector3d& a_max)
{
  return getMinTimeDoubleIntegrator3D(initial_state.pos, initial_state.vel, final_state.pos, final_state.vel, v_max,
                                      a_max);
}

// note that b>=a is a requirement
double probUnivariateNormalDistAB(double a, double b, double mu, double std_deviation)
{
  if (b < a)
  {
    /////// Debugging
    std::cout << "Needed: b>=a. ABORTING!" << std::endl;
    abort();
    ////////////////
  }

  return (cdfUnivariateNormalDist(b, mu, std_deviation) - cdfUnivariateNormalDist(a, mu, std_deviation));
}

// This works when the covariance matris is diagonal (and the diagonal is given by the vector std_deviation.^2 )
// TODO: we are assumming that all the vectors have the same size (if not it'll crash) --> make template
double probMultivariateNormalDist(const Eigen::VectorXd& a, const Eigen::VectorXd& b, const Eigen::VectorXd& mu,
                                  const Eigen::VectorXd& std_deviation)
{
  double prob_less_a = 1.0;
  for (int i = 0; i < a.size(); i++)
  {
    prob_less_a *= cdfUnivariateNormalDist(a(i), mu(i), std_deviation(i));
  }

  double prob_less_b = 1.0;
  for (int i = 0; i < b.size(); i++)
  {
    prob_less_b *= cdfUnivariateNormalDist(b(i), mu(i), std_deviation(i));
  }
  return (prob_less_b - prob_less_a);
}

// https://www.geeksforgeeks.org/binomial-coefficient-dp-9/#highlighter_72035:~:text=Following%20is%20a%20space%2Doptimized%20version%20of%20the%20above%20code
int nChoosek(int n, int k)
{
  int C[k + 1];
  memset(C, 0, sizeof(C));

  C[0] = 1;  // nC0 is 1

  for (int i = 1; i <= n; i++)
  {
    // Compute next row of pascal triangle using
    // the previous row
    for (int j = std::min(i, k); j > 0; j--)
      C[j] = C[j] + C[j - 1];
  }
  return C[k];
}

// https://stackoverflow.com/questions/141422/how-can-a-transform-a-polynomial-to-another-coordinate-system

// Denote T=[t^n,...,t,1]'
// given a polynomial p(t)=coeff_p'*T
// compute the coeff_q of the polynomial q(t)=p(a*t + b).
// q(t)=coeff_q'*T
void linearTransformPoly(const Eigen::VectorXd& coeff_p, Eigen::VectorXd& coeff_q, double a, double b)
{
  Eigen::VectorXd c = coeff_p.reverse();

  Eigen::MatrixXd Q(coeff_p.size(), coeff_p.size());
  for (int i = 0; i < Q.rows(); i++)
  {
    for (int j = 0; j < Q.cols(); j++)
    {
      int jchoosei = (i > j) ? 0 : nChoosek(j, i);
      Q(i, j) = jchoosei * pow(a, i) * pow(b, j - i);
    }
  }
  Eigen::VectorXd d = Q * c;
  coeff_q = d.reverse();  // Note that d = d.reverse() is wrong!

  // std::cout << "Q= \n" << Q << std::endl;
  // std::cout << "coeff_p= \n" << coeff_p << std::endl;
  // std::cout << "coeff_q= \n" << coeff_q << std::endl;
  // std::cout << "coeff_q= \n" << coeff_q << std::endl;
}

// Obtain the coefficients of q(t) such that
// p(tp1)==q(tq1)
// p(tp2)==q(tq2)
void changeDomPoly(const Eigen::VectorXd& coeff_p, double tp1, double tp2, Eigen::VectorXd& coeff_q, double tq1,
                   double tq2)
{
  // The equations below are from executing this in Matlab:
  // syms tq1 tq2 tp1 tp2 a b;
  // s=solve([tp1==a*tq1+b, tp2==a*tq2+b],[a,b])
  double a = (tp1 - tp2) / (tq1 - tq2);
  double b = -(tp1 * tq2 - tp2 * tq1) / (tq1 - tq2);
  linearTransformPoly(coeff_p, coeff_q, a, b);
}

void verify(bool cond, std::string info_if_false)
{
  if (cond == false)
  {
    std::cout << termcolor::bold << termcolor::red << info_if_false << termcolor::reset << std::endl;
    std::cout << termcolor::red << "Aborting" << termcolor::reset << std::endl;
    abort();
  }
}

// returns a mt::PieceWisePol, taking the polynomials of p1 and p2 that should satisfy p1>t, p2>t
mt::PieceWisePol composePieceWisePol(const double t, const double dc, mt::PieceWisePol& p1, mt::PieceWisePol& p2)
{
  // if t is in between p1 and p2, force p2[0] to be t
  if (t > p1.times.back() && t < p2.times.front())  // && fabs(t - p2.times.front()) <= dc) TODO Sometimes fabs(t -
                                                    // p2.times.front()) is 0.18>c
  {
    p2.times.front() = t;
  }

  if (p1.times.back() < p2.times.front())  // && fabs(p1.times.back() - p2.times.front()) <= dc) TODO
  {
    p2.times.front() = p1.times.back();
  }

  if (t < p1.times.front())  // TODO
  {
    p1.times.front() = t;
  }

  if (fabs(t - p2.times.front()) < 1e-5)
  {
    return p2;
  }

  if (p1.times.back() < p2.times.front() || t > p2.times.back() || t < p1.times.front())
  {
    // TODO?
    // std::cout << "Error composing the piecewisePol" << std::endl;
    // std::cout << std::setprecision(30) << "t= " << t << std::endl;
    // std::cout << std::setprecision(30) << "p1.times.front()= " << p1.times.front() << std::endl;
    // std::cout << std::setprecision(30) << "p1.times.back()= " << p1.times.back() << std::endl;
    // std::cout << std::setprecision(30) << "p2.times.front() = " << p2.times.front() << std::endl;
    // std::cout << std::setprecision(30) << "p2.times.back() = " << p2.times.back() << std::endl;
    mt::PieceWisePol tmp;
    return tmp;
  }

  std::vector<int> indexes1, indexes2;

  for (int i = 0; i < p1.times.size(); i++)
  {
    if (p1.times[i] > t && p1.times[i] < p2.times[0])
    {
      indexes1.push_back(i);
    }
  }

  for (int i = 0; i < p2.times.size(); i++)
  {
    if (p2.times[i] > t)
    {
      indexes2.push_back(i);
    }
  }

  mt::PieceWisePol p;
  p.times.push_back(t);

  for (auto index_1_i : indexes1)
  {
    p.times.push_back(p1.times[index_1_i]);
    p.all_coeff_x.push_back(p1.all_coeff_x[index_1_i - 1]);
    p.all_coeff_y.push_back(p1.all_coeff_y[index_1_i - 1]);
    p.all_coeff_z.push_back(p1.all_coeff_z[index_1_i - 1]);
  }

  for (auto index_2_i : indexes2)
  {
    if (index_2_i == 0)
    {
      p.all_coeff_x.push_back(p1.all_coeff_x.back());
      p.all_coeff_y.push_back(p1.all_coeff_y.back());
      p.all_coeff_z.push_back(p1.all_coeff_z.back());
      p.times.push_back(p2.times[index_2_i]);
      continue;
    }
    p.times.push_back(p2.times[index_2_i]);
    p.all_coeff_x.push_back(p2.all_coeff_x[index_2_i - 1]);
    p.all_coeff_y.push_back(p2.all_coeff_y[index_2_i - 1]);
    p.all_coeff_z.push_back(p2.all_coeff_z[index_2_i - 1]);
  }

  return p;
}

std::vector<std::string> pieceWisePol2String(const mt::PieceWisePol& pwp)
{
  // Define strings
  std::string s_x = "0.0";
  std::string s_y = "0.0";
  std::string s_z = "0.0";

  int deg = pwp.getDeg();

  //(pwp.times - 1) is the number of intervals
  for (int i = 0; i < (pwp.times.size() - 1); i++)  // i is the index of the interval
  {
    std::string div_by_delta = "/ (" + std::to_string(pwp.times[i + 1] - pwp.times[i]) + ")";

    std::string t = "(min(t," + std::to_string(pwp.times.back()) + "))";

    std::string u = "(" + t + "-" + std::to_string(pwp.times[i]) + ")" + div_by_delta;
    u = "(" + u + ")";

    // std::string u = "(" + t + "-" + std::to_string(pwp.times[i]) + ")" + div_by_delta;
    // u = "(" + u + ")";
    // std::string uu = u + "*" + u;
    // std::string uuu = u + "*" + u + "*" + u;
    /*    std::cout << "pwp.times[i]= " << pwp.times[i] << std::endl;
        std::cout << "pwp.times[i+1]= " << pwp.times[i + 1] << std::endl;*/

    std::string cond;
    if (i == (pwp.times.size() - 2))  // if the last interval
    {
      cond = "(t>=" + std::to_string(pwp.times[i]) + ")";
    }
    else
    {
      cond = "(t>=" + std::to_string(pwp.times[i]) + " and " + "t<" + std::to_string(pwp.times[i + 1]) + ")";
    }

    std::string s_x_i = "";
    std::string s_y_i = "";
    std::string s_z_i = "";
    for (int j = 0; j <= deg; j++)
    {
      std::string power_u = "(" + u + "^" + std::to_string(deg - j) + ")";

      s_x_i = s_x_i + "+" + std::to_string((double)pwp.all_coeff_x[i](j)) + "*" + power_u;
      s_y_i = s_y_i + "+" + std::to_string((double)pwp.all_coeff_y[i](j)) + "*" + power_u;
      s_z_i = s_z_i + "+" + std::to_string((double)pwp.all_coeff_z[i](j)) + "*" + power_u;
    }

    s_x_i = cond + "*(" + s_x_i + ")";
    s_y_i = cond + "*(" + s_y_i + ")";
    s_z_i = cond + "*(" + s_z_i + ")";

    // std::string s_x_i = std::to_string((double)pwp.all_coeff_x[i](0)) + "*" + uuu;   //////////////////
    // s_x_i = s_x_i + "+" + std::to_string((double)pwp.all_coeff_x[i](1)) + "*" + uu;  //////////////////
    // s_x_i = s_x_i + "+" + std::to_string((double)pwp.all_coeff_x[i](2)) + "*" + u;   //////////////////
    // s_x_i = s_x_i + "+" + std::to_string((double)pwp.all_coeff_x[i](3));             //////////////////
    // s_x_i = cond + "*(" + s_x_i + ")";

    // std::string s_y_i = std::to_string((double)pwp.all_coeff_y[i](0)) + "*" + uuu;   //////////////////
    // s_y_i = s_y_i + "+" + std::to_string((double)pwp.all_coeff_y[i](1)) + "*" + uu;  //////////////////
    // s_y_i = s_y_i + "+" + std::to_string((double)pwp.all_coeff_y[i](2)) + "*" + u;   //////////////////
    // s_y_i = s_y_i + "+" + std::to_string((double)pwp.all_coeff_y[i](3));             //////////////////
    // s_y_i = cond + "*(" + s_y_i + ")";

    // std::string s_z_i = std::to_string((double)pwp.all_coeff_z[i](0)) + "*" + uuu;   //////////////////
    // s_z_i = s_z_i + "+" + std::to_string((double)pwp.all_coeff_z[i](1)) + "*" + uu;  //////////////////
    // s_z_i = s_z_i + "+" + std::to_string((double)pwp.all_coeff_z[i](2)) + "*" + u;   //////////////////
    // s_z_i = s_z_i + "+" + std::to_string((double)pwp.all_coeff_z[i](3));             //////////////////
    // s_z_i = cond + "*(" + s_z_i + ")";

    s_x = s_x + " + " + s_x_i;
    s_y = s_y + " + " + s_y_i;
    s_z = s_z + " + " + s_z_i;
  }

  std::vector<std::string> s;
  s.push_back(s_x);
  s.push_back(s_y);
  s.push_back(s_z);

  return s;
}

void printStateDeque(std::deque<mt::state>& data)
{
  for (int i = 0; i < data.size(); i++)
  {
    data[i].printHorizontal();
  }
}

void printStateVector(std::vector<mt::state>& data)
{
  for (int i = 0; i < data.size(); i++)
  {
    data[i].printHorizontal();
  }
}

// It assummes that the box is aligned with x, y, z
// C1 is the corner with lowest x,y,z
// C2 is the corner with highest x,y,z
// center is the center of the sphere
// r is the radius of the sphere
bool boxIntersectsSphere(Eigen::Vector3d center, double r, Eigen::Vector3d c1, Eigen::Vector3d c2)
{
  // https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
  //(Section Sphere vs AABB)

  Eigen::Vector3d closest_point;  // closest point from the center of the sphere to the box

  closest_point(0) = std::max(c1.x(), std::min(center.x(), c2.x()));
  closest_point(1) = std::max(c1.y(), std::min(center.y(), c2.y()));
  closest_point(2) = std::max(c1.z(), std::min(center.z(), c2.z()));

  // this is the same as isPointInsideSphere
  double distance_to_closest_point = (center - closest_point).norm();

  return (distance_to_closest_point < r);  // true if the box intersects the sphere
}

void saturate(Eigen::Vector3d& tmp, const Eigen::Vector3d& min, const Eigen::Vector3d& max)
{
  saturate(tmp(0), min(0), max(0));
  saturate(tmp(1), min(1), max(1));
  saturate(tmp(2), min(2), max(2));
}

void saturate(int& var, const int min, const int max)
{
  if (var < min)
  {
    var = min;
  }
  else if (var > max)
  {
    var = max;
  }
}

// TODO: Make a template here
void saturate(double& var, const double min, const double max)
{
  if (var < min)
  {
    var = min;
  }
  else if (var > max)
  {
    var = max;
  }
}

double angleBetVectors(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
  double tmp = a.dot(b) / (a.norm() * b.norm());

  saturate(tmp, -1, 1);
  return acos(tmp);
}

void angle_wrap(double& diff)
{
  diff = fmod(diff + M_PI, 2 * M_PI);
  if (diff < 0)
    diff += 2 * M_PI;
  diff -= M_PI;
}


std::vector<double> eigen2std(const Eigen::Vector3d& v)
{
  return std::vector<double>{ v.x(), v.y(), v.z() };
}

// std::vector<float> eigen2std(const Eigen::Vector3f& v)
// {
//   return std::vector<float>{ v.x(), v.y(), v.z() };
// }

casadi::DM throwOutThirdDimension(casadi::DM matrix)
{
  casadi::DM matrix_2D(2, matrix.columns());
  for (int j = 0; j < matrix.columns(); j++)
  {
    matrix_2D(0, j) = matrix(0, j);
    matrix_2D(1, j) = matrix(1, j);
  }
  return matrix_2D;
}

std::vector<double> throwOutThirdDimension(std::vector<double> vector)
{
  return std::vector<double> { vector[0], vector[1] };
}

std::string casadi_folder()
{
  std::string this_file = __FILE__;
  fs::path folder = fs::path(this_file);
  folder.remove_filename();
  folder = folder / fs::path("../matlab/casadi_generated_files/");
  folder = fs::canonical(folder);
  std::string string_folder{folder.u8string()};
  return string_folder + "/";
}

