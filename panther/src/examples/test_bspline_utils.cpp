/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include "bspline_utils.hpp"

int main()
{
  std::vector<double> times;
  // times.push_back(0.0);
  // times.push_back(0.5);
  // times.push_back(0.7);
  // times.push_back(1.0);

  times.push_back(1591133329.99584);
  times.push_back(1591133330.29584);
  times.push_back(1591133330.59584);
  times.push_back(1591133330.89584);
  times.push_back(1591133331.19584);
  times.push_back(1591133331.49585);
  times.push_back(1591133331.79585);

  std::vector<Eigen::Vector3d> positions;
  // positions.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
  // positions.push_back(Eigen::Vector3d(5.0, 0.0, 0.0));
  // positions.push_back(Eigen::Vector3d(7.0, 0.0, 0.0));
  // positions.push_back(Eigen::Vector3d(7.5, 0.0, 0.0));

  positions.push_back(Eigen::Vector3d(1.42555006072, -0.535729530809, 1));
  positions.push_back(Eigen::Vector3d(1.42555006072, -0.535729530809, 1));
  positions.push_back(Eigen::Vector3d(1.42555006072, -0.535729530809, 1));
  positions.push_back(Eigen::Vector3d(1.42555006072, -0.535729530809, 1));
  positions.push_back(Eigen::Vector3d(1.42555006072, -0.535729530809, 1));
  positions.push_back(Eigen::Vector3d(1.42555006072, -0.535729530809, 1));
  positions.push_back(Eigen::Vector3d(1.42555006072, -0.535729530809, 1));

  Eigen::Spline3d spline = findInterpolatingBsplineNormalized(times, positions);
  // std::cout << "Control points are " << spline.ctrls() << std::endl;
  // std::cout << "Knots are " << spline.knots() << std::endl;

  ////////////////////////
  ////////////////////////
  ////////////////////////
  ////////////////////////
  std::vector<double> knots_p_std;

  knots_p_std.push_back(3.235);
  knots_p_std.push_back(3.235);
  knots_p_std.push_back(3.235);
  knots_p_std.push_back(3.235);
  knots_p_std.push_back(3.90167);
  knots_p_std.push_back(4.56833);
  knots_p_std.push_back(5.235);
  knots_p_std.push_back(6.56833);
  knots_p_std.push_back(7.235);
  knots_p_std.push_back(7.235);
  knots_p_std.push_back(7.235);
  knots_p_std.push_back(7.235);

  std::vector<Eigen::Vector3d> qp;
  qp.push_back(Eigen::Vector3d(3.95869, 0.0061970, -0.00464359));
  qp.push_back(Eigen::Vector3d(4.16085, 0.191152, 0.412312));
  qp.push_back(Eigen::Vector3d(4.47408, 0.865258, 1.80363));
  qp.push_back(Eigen::Vector3d(4.24042, 1.90309, 1.28377));
  qp.push_back(Eigen::Vector3d(3.60723, 2.06505, -0.796185));
  qp.push_back(Eigen::Vector3d(3.33301, 1.25081, -0.299195));
  qp.push_back(Eigen::Vector3d(3.65292, 0.582939, 1.11423));
  qp.push_back(Eigen::Vector3d(3.88078, 0.373237, 1.34436));

  Eigen::Matrix<double, 3, -1> qp_matrix(3, qp.size());
  for (int i = 0; i < qp.size(); i++)
  {
    qp_matrix.col(i) = qp[i];
  }

  Eigen::RowVectorXd knots_p(1, knots_p_std.size());
  for (int i = 0; i < knots_p.cols(); i++)
  {
    knots_p(i) = knots_p_std[i];
  }

  std::cout << "qp_matrix= \n" << qp_matrix << std::endl;
  std::cout << "knots_p= \n" << knots_p << std::endl;

  Eigen::Spline<double, 3, Eigen::Dynamic> spline_p(knots_p, qp_matrix);

  for (double t = knots_p_std.front(); t <= knots_p_std.back(); t = t + 0.5)
  {
    // std::cout << std::setprecision(20) << "t= " << t << std::endl;

    Eigen::MatrixXd derivatives_p = spline_p.derivatives(t, 4);  // compute the derivatives up to that order

    mt::state state_i;

    state_i.setPos(derivatives_p.col(0));  // First column
    state_i.setVel(derivatives_p.col(1));
    state_i.setAccel(derivatives_p.col(2));
    state_i.setJerk(derivatives_p.col(3));

    state_i.printHorizontal();
  }
}