/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include "ros_utils.hpp"
#include "termcolor.hpp"

panther_msgs::Log log2LogMsg(mt::log log)
{
  panther_msgs::Log log_msg;

  log_msg.replanning_was_needed = log.replanning_was_needed;

  log_msg.ms_initial_setup = log.tim_initial_setup.getMsSaved();
  log_msg.ms_convex_hulls = log.tim_convex_hulls.getMsSaved();
  log_msg.ms_opt = log.tim_opt.getMsSaved();
  log_msg.ms_guess_pos = log.tim_guess_pos.getMsSaved();
  log_msg.ms_guess_yaw_search_graph = log.tim_guess_yaw_search_graph.getMsSaved();
  log_msg.ms_guess_yaw_fit_poly = log.tim_guess_yaw_fit_poly.getMsSaved();
  log_msg.ms_total_replan = log.tim_total_replan.getMsSaved();

  log_msg.cost = log.cost;
  log_msg.obst_avoidance_violation = log.obst_avoidance_violation;
  log_msg.dyn_lim_violation = log.dyn_lim_violation;

  log_msg.tracking_now_pos.x = log.tracking_now_pos.x();
  log_msg.tracking_now_pos.y = log.tracking_now_pos.y();
  log_msg.tracking_now_pos.z = log.tracking_now_pos.z();

  log_msg.tracking_now_vel.x = log.tracking_now_vel.x();
  log_msg.tracking_now_vel.y = log.tracking_now_vel.y();
  log_msg.tracking_now_vel.z = log.tracking_now_vel.z();

  log_msg.pos.x = log.pos.x();
  log_msg.pos.y = log.pos.y();
  log_msg.pos.z = log.pos.z();

  log_msg.G_term_pos.x = log.G_term_pos.x();
  log_msg.G_term_pos.y = log.G_term_pos.y();
  log_msg.G_term_pos.z = log.G_term_pos.z();

  log_msg.success_guess_pos = log.success_guess_pos;
  log_msg.success_guess_yaw = log.success_guess_yaw;
  log_msg.success_opt = log.success_opt;
  log_msg.success_replanning = log.success_replanning;

  log_msg.info_replan = log.info_replan;
  log_msg.header.stamp = ros::Time::now();

  switch (log.drone_status)
  {
    case DroneStatus::YAWING:
      log_msg.drone_status = "YAWING";
      break;
    case DroneStatus::TRAVELING:
      log_msg.drone_status = "TRAVELING";
      break;
    case DroneStatus::GOAL_SEEN:
      log_msg.drone_status = "GOAL_SEEN";
      break;
    case DroneStatus::GOAL_REACHED:
      log_msg.drone_status = "GOAL_REACHED";
      break;
    default:
      log_msg.drone_status = "";
      break;
  }

  return log_msg;
}

visualization_msgs::MarkerArray pwp2ColoredMarkerArray(mt::PieceWisePol& pwp, double t_init, double t_final,
                                                       int samples, std::string ns, Eigen::Vector3d& color)
{
  visualization_msgs::MarkerArray marker_array;

  if (t_final < t_init)
  {
    // std::cout << "t_final<t_init" << std::endl;
    abort();
    return marker_array;
  }

  // std::cout << "t_init= " << std::setprecision(15) << t_init << std::endl;
  // std::cout << "t_final= " << std::setprecision(15) << t_final << std::endl;

  double deltaT = (t_final - t_init) / (1.0 * samples);

  geometry_msgs::Point p_last = eigen2point(pwp.eval(t_init));

  int j = 7 * 9000;  // TODO

  for (double t = t_init; t <= t_final; t = t + deltaT)
  {
    visualization_msgs::Marker m;
    m.type = visualization_msgs::Marker::ARROW;
    m.header.frame_id = "world";
    m.header.stamp = ros::Time::now();
    m.ns = ns;
    m.action = visualization_msgs::Marker::ADD;
    m.id = j;

    m.color.r = color.x();  // color(RED_NORMAL);
    m.color.g = color.y();
    m.color.b = color.z();
    m.color.a = 1.0;

    m.scale.x = 0.1;
    m.scale.y = 0.0000001;  // rviz complains if not
    m.scale.z = 0.0000001;  // rviz complains if not

    m.pose.orientation.w = 1.0;

    geometry_msgs::Point p = eigen2point(pwp.eval(t));

    m.points.push_back(p_last);
    m.points.push_back(p);

    p_last = p;
    marker_array.markers.push_back(m);
    j = j + 1;
  }

  return marker_array;
}

panther_msgs::PieceWisePolTraj pwp2PwpMsg(const mt::PieceWisePol& pwp)
{
  panther_msgs::PieceWisePolTraj pwp_msg;

  for (int i = 0; i < pwp.times.size(); i++)
  {
    // std::cout << termcolor::red << "in pwp2PwpMsg, pushing back" << std::setprecision(20) << pwp.times[i]
    //           << termcolor::reset << std::endl;
    pwp_msg.times.push_back(pwp.times[i]);
  }

  // push x
  for (auto coeff_x_i : pwp.all_coeff_x)
  {
    panther_msgs::CoeffPoly coeff_poly3;

    for (int i = 0; i < coeff_x_i.size(); i++)
    {
      coeff_poly3.data.push_back(coeff_x_i(i));
    }
    pwp_msg.all_coeff_x.push_back(coeff_poly3);
  }

  // push y
  for (auto coeff_y_i : pwp.all_coeff_y)
  {
    panther_msgs::CoeffPoly coeff_poly3;
    for (int i = 0; i < coeff_y_i.size(); i++)
    {
      coeff_poly3.data.push_back(coeff_y_i(i));
    }
    pwp_msg.all_coeff_y.push_back(coeff_poly3);
  }

  // push z
  for (auto coeff_z_i : pwp.all_coeff_z)
  {
    panther_msgs::CoeffPoly coeff_poly3;
    for (int i = 0; i < coeff_z_i.size(); i++)
    {
      coeff_poly3.data.push_back(coeff_z_i(i));
    }
    pwp_msg.all_coeff_z.push_back(coeff_poly3);
  }

  return pwp_msg;
}

mt::PieceWisePol pwpMsg2Pwp(const panther_msgs::PieceWisePolTraj& pwp_msg)
{
  mt::PieceWisePol pwp;

  if (pwp_msg.all_coeff_x.size() != pwp_msg.all_coeff_y.size() ||
      pwp_msg.all_coeff_x.size() != pwp_msg.all_coeff_z.size())
  {
    std::cout << " coeff_x,coeff_y,coeff_z of pwp_msg should have the same elements" << std::endl;
    std::cout << " ================================" << std::endl;
    abort();
  }

  for (int i = 0; i < pwp_msg.times.size(); i++)
  {
    pwp.times.push_back(pwp_msg.times[i]);
  }

  for (int i = 0; i < pwp_msg.all_coeff_x.size(); i++)  // For each of the intervals
  {
    int degree = pwp_msg.all_coeff_x[i].data.size() - 1;  // Should be the same for all i, and for x, y, z

    Eigen::VectorXd tmp_x(degree + 1);
    Eigen::VectorXd tmp_y(degree + 1);
    Eigen::VectorXd tmp_z(degree + 1);

    for (int j = 0; j < (degree + 1); j++)
    {
      tmp_x(j) = pwp_msg.all_coeff_x[i].data[j];
      tmp_y(j) = pwp_msg.all_coeff_y[i].data[j];
      tmp_z(j) = pwp_msg.all_coeff_z[i].data[j];
    }

    // std::cout << termcolor::on_blue << "pwpMsg2Pwp: " << pwp_msg.all_coeff_z[i].a << ", " << pwp_msg.all_coeff_z[i].b
    // << ", "
    //           << pwp_msg.all_coeff_z[i].c << ", " << pwp_msg.all_coeff_z[i].d << termcolor::reset << std::endl;

    pwp.all_coeff_x.push_back(tmp_x);
    pwp.all_coeff_y.push_back(tmp_y);
    pwp.all_coeff_z.push_back(tmp_z);
  }

  return pwp;
}

visualization_msgs::Marker edges2Marker(const mt::Edges& edges, std_msgs::ColorRGBA color_marker)
{
  visualization_msgs::Marker marker;

  if (edges.size() == 0)  // there are no edges
  {
    // std::cout << "there are no edges" << std::endl;
    return marker;
  }

  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "markertacles";
  marker.id = 0;
  marker.type = marker.LINE_LIST;
  marker.action = marker.ADD;
  marker.pose = identityGeometryMsgsPose();

  marker.points.clear();

  for (auto edge : edges)
  {
    marker.points.push_back(eigen2point(edge.first));
    marker.points.push_back(eigen2point(edge.second));
  }

  marker.scale.x = 0.03;
  // marker.scale.y = 0.00001;
  // marker.scale.z = 0.00001;
  marker.color = color_marker;

  return marker;
}

mt::PieceWisePol createPwpFromStaticPosition(const mt::state& current_state)
{
  mt::PieceWisePol pwp;
  pwp.times = { ros::Time::now().toSec(), ros::Time::now().toSec() + 1e10 };

  // In this case we encode it as a third-order polynomial
  Eigen::Matrix<double, 4, 1> coeff_x_interv0;  // [a b c d]' of the interval 0
  Eigen::Matrix<double, 4, 1> coeff_y_interv0;  // [a b c d]' of the interval 0
  Eigen::Matrix<double, 4, 1> coeff_z_interv0;  // [a b c d]' of the interval 0

  coeff_x_interv0 << 0.0, 0.0, 0.0, current_state.pos.x();
  coeff_y_interv0 << 0.0, 0.0, 0.0, current_state.pos.y();
  coeff_z_interv0 << 0.0, 0.0, 0.0, current_state.pos.z();

  pwp.all_coeff_x.push_back(coeff_x_interv0);
  pwp.all_coeff_y.push_back(coeff_y_interv0);
  pwp.all_coeff_z.push_back(coeff_z_interv0);

  return pwp;
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

geometry_msgs::Pose identityGeometryMsgsPose()
{
  geometry_msgs::Pose pose;
  pose.position.x = 0;
  pose.position.y = 0;
  pose.position.z = 0;
  pose.orientation.x = 0;
  pose.orientation.y = 0;
  pose.orientation.z = 0;
  pose.orientation.w = 1;
  return pose;
}

std_msgs::ColorRGBA getColorInterpBetween2Colors(double v, double vmin, double vmax, std_msgs::ColorRGBA min_color,
                                                 std_msgs::ColorRGBA max_color)
{
  double dv;

  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  double fraction_max_color = (v - vmin) / dv;

  // std_msgs::ColorRGBA first = color(YELLOW_NORMAL);
  // std_msgs::ColorRGBA last = color(GREEN_NORMAL);

  std_msgs::ColorRGBA c;
  c.r = (max_color.r - min_color.r) * fraction_max_color + min_color.r;
  c.g = (max_color.g - min_color.g) * fraction_max_color + min_color.g;
  c.b = (max_color.b - min_color.b) * fraction_max_color + min_color.b;
  c.a = 1;

  return c;
}

std_msgs::ColorRGBA getColorJet(double v, double vmin, double vmax)
{
  vmax = std::max(vmax, 1.0001 * vmin);  // Ensure vmax>vmin

  std_msgs::ColorRGBA c;
  c.r = 1;
  c.g = 1;
  c.b = 1;
  c.a = 1;
  // white
  double dv;

  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv))
  {
    c.r = 0;
    c.g = 4 * (v - vmin) / dv;
  }
  else if (v < (vmin + 0.5 * dv))
  {
    c.r = 0;
    c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
  }
  else if (v < (vmin + 0.75 * dv))
  {
    c.r = 4 * (v - vmin - 0.5 * dv) / dv;
    c.b = 0;
  }
  else
  {
    c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
    c.b = 0;
  }

  return (c);
}

std_msgs::ColorRGBA color(int id)
{
  std_msgs::ColorRGBA red;
  red.r = 1;
  red.g = 0;
  red.b = 0;
  red.a = 1;
  std_msgs::ColorRGBA red_trans;
  red_trans.r = 1;
  red_trans.g = 0;
  red_trans.b = 0;
  red_trans.a = 0.7;
  std_msgs::ColorRGBA red_trans_trans;
  red_trans_trans.r = 1;
  red_trans_trans.g = 0;
  red_trans_trans.b = 0;
  red_trans_trans.a = 0.4;
  std_msgs::ColorRGBA blue;
  blue.r = 0;
  blue.g = 0;
  blue.b = 1;
  blue.a = 1;
  std_msgs::ColorRGBA blue_trans;
  blue_trans.r = 0;
  blue_trans.g = 0;
  blue_trans.b = 1;
  blue_trans.a = 0.7;
  std_msgs::ColorRGBA blue_trans_trans;
  blue_trans_trans.r = 0;
  blue_trans_trans.g = 0;
  blue_trans_trans.b = 1;
  blue_trans_trans.a = 0.4;
  std_msgs::ColorRGBA blue_light;
  blue_light.r = 0.5;
  blue_light.g = 0.7;
  blue_light.b = 1;
  blue_light.a = 1;
  std_msgs::ColorRGBA green;
  green.r = 0;
  green.g = 1;
  green.b = 0;
  green.a = 1;
  std_msgs::ColorRGBA yellow;
  yellow.r = 1;
  yellow.g = 1;
  yellow.b = 0;
  yellow.a = 1;
  std_msgs::ColorRGBA orange_trans;  // orange transparent
  orange_trans.r = 1;
  orange_trans.g = 0.5;
  orange_trans.b = 0;
  orange_trans.a = 0.7;
  std_msgs::ColorRGBA teal_normal;  // teal transparent
  teal_normal.r = 25 / 255.0;
  teal_normal.g = 1.0;
  teal_normal.b = 240.0 / 255.0;
  teal_normal.a = 1.0;
  std_msgs::ColorRGBA black_trans;  // black transparent
  black_trans.r = 0.0;
  black_trans.g = 0.0;
  black_trans.b = 0.0;
  black_trans.a = 0.2;
  std_msgs::ColorRGBA black;  // black
  black.r = 0.0;
  black.g = 0.0;
  black.b = 0.0;
  black.a = 1.0;

  switch (id)
  {
    case RED_NORMAL:
      return red;
      break;
    case RED_TRANS:
      return red_trans;
      break;
    case RED_TRANS_TRANS:
      return red_trans_trans;
      break;
    case BLUE_NORMAL:
      return blue;
      break;
    case BLUE_TRANS:
      return blue_trans;
      break;
    case BLUE_TRANS_TRANS:
      return blue_trans_trans;
      break;
    case BLUE_LIGHT:
      return blue_light;
      break;
    case GREEN_NORMAL:
      return green;
      break;
    case YELLOW_NORMAL:
      return yellow;
      break;
    case ORANGE_TRANS:
      return orange_trans;
      break;
    case BLACK_TRANS:
      return black_trans;
      break;
    case BLACK:
      return black;
      break;
    case TEAL_NORMAL:
      return teal_normal;
      break;
    default:
      std::cout << "COLOR NOT DEFINED, returning RED" << std::endl;
      return red;
  }
}

//## From Wikipedia - http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
void quaternion2Euler(tf2::Quaternion q, double& roll, double& pitch, double& yaw)
{
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
}

void quaternion2Euler(geometry_msgs::Quaternion q, double& roll, double& pitch, double& yaw)
{
  tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
  quaternion2Euler(tf_q, roll, pitch, yaw);
}

void quaternion2Euler(Eigen::Quaterniond q, double& roll, double& pitch, double& yaw)
{
  tf2::Quaternion tf_q(q.x(), q.y(), q.z(), q.w());
  quaternion2Euler(tf_q, roll, pitch, yaw);
}

visualization_msgs::Marker getMarkerSphere(double scale, int my_color)
{
  visualization_msgs::Marker marker;

  marker.header.frame_id = "world";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;
  marker.color = color(my_color);

  return marker;
}

geometry_msgs::Point pointOrigin()
{
  geometry_msgs::Point tmp;
  tmp.x = 0;
  tmp.y = 0;
  tmp.z = 0;
  return tmp;
}

Eigen::Vector3d vec2eigen(geometry_msgs::Vector3 vector)
{
  Eigen::Vector3d tmp;
  tmp << vector.x, vector.y, vector.z;
  return tmp;
}

geometry_msgs::Vector3 eigen2rosvector(Eigen::Vector3d vector)
{
  geometry_msgs::Vector3 tmp;
  tmp.x = vector(0, 0);
  tmp.y = vector(1, 0);
  tmp.z = vector(2, 0);
  return tmp;
}

geometry_msgs::Point eigen2point(Eigen::Vector3d vector)
{
  geometry_msgs::Point tmp;
  tmp.x = vector[0];
  tmp.y = vector[1];
  tmp.z = vector[2];
  return tmp;
}

geometry_msgs::Vector3 vectorNull()
{
  geometry_msgs::Vector3 tmp;
  tmp.x = 0;
  tmp.y = 0;
  tmp.z = 0;
  return tmp;
}

geometry_msgs::Vector3 vectorUniform(double a)
{
  geometry_msgs::Vector3 tmp;
  tmp.x = a;
  tmp.y = a;
  tmp.z = a;
  return tmp;
}

visualization_msgs::MarkerArray trajectory2ColoredMarkerArray(const mt::trajectory& data, double max_value, int increm,
                                                              std::string ns, double scale, std::string color_type,
                                                              int id_agent, int n_agents, double min_aug_cost,
                                                              double max_aug_cost, double aug_cost, bool collides)
{
  visualization_msgs::MarkerArray marker_array;

  if (data.size() == 0)
  {
    return marker_array;
  }
  geometry_msgs::Point p_last;
  p_last.x = data[0].pos(0);
  p_last.y = data[0].pos(1);
  p_last.z = data[0].pos(2);

  increm = (increm < 1.0) ? 1 : increm;

  int j = 9000;
  for (int i = 0; i < data.size(); i = i + increm)
  {
    double vel = data[i].vel.norm();
    visualization_msgs::Marker m;
    m.type = visualization_msgs::Marker::ARROW;
    m.header.frame_id = "world";
    m.header.stamp = ros::Time::now();
    m.ns = ns;
    m.action = visualization_msgs::Marker::ADD;
    m.id = j;
    if (color_type == "vel")  // TODO: "vel" is hand-coded
    {
      m.color = getColorJet(vel, 0, max_value);  // note that par_.v_max is per axis!
    }
    else if (color_type == "time")  // TODO: "time" is hand-coded
    {
      m.color = getColorJet(i, 0, data.size());
    }
    else if (color_type == "aug_cost")  // TODO: "time" is hand-coded
    {
      m.color = getColorJet(aug_cost, min_aug_cost, max_aug_cost);
    }
    else if (color_type == "agent")  // TODO: "time" is hand-coded
    {
      m.color = getColorJet(id_agent, 0, n_agents);
    }
    else if (color_type == "black")  // TODO: "time" is hand-coded
    {
      m.color = color(BLACK);
    }
    else
    {
      std::cout << "color_type CHOSEN IS NOT SUPPORTED" << std::endl;
      abort();
    }

    if (collides)
    {
      m.color.a = 0.0;  // Make it invisible
    }

    // m.color.a = alpha;
    m.scale.x = scale;
    m.scale.y = 0.0000001;  // rviz complains if not
    m.scale.z = 0.0000001;  // rviz complains if not

    m.pose.orientation.w = 1.0;
    // std::cout << "Mandando bloque" << X.block(i, 0, 1, 3) << std::endl;
    geometry_msgs::Point p;
    p.x = data[i].pos(0);
    p.y = data[i].pos(1);
    p.z = data[i].pos(2);
    m.points.push_back(p_last);
    m.points.push_back(p);
    // std::cout << "pushing marker\n" << m << std::endl;
    p_last = p;
    marker_array.markers.push_back(m);
    j = j + 1;
  }
  return marker_array;
}
