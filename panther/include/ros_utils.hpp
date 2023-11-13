/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>

#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "panther_types.hpp"
#include <deque>

#include <panther_msgs/PieceWisePolTraj.h>
#include <panther_msgs/CoeffPoly.h>
#include <panther_msgs/Log.h>

#include "ros/ros.h"

template <typename T>
bool safeGetParam(ros::NodeHandle& nh, std::string const& param_name, T& param_value)
{
  if (!nh.getParam(param_name, param_value))
  {
    ROS_ERROR("Failed to find parameter: %s", nh.resolveName(param_name, true).c_str());
    exit(1);
  }
  return true;
}

panther_msgs::Log log2LogMsg(mt::log log);

visualization_msgs::MarkerArray pwp2ColoredMarkerArray(mt::PieceWisePol& pwp, double t_init, double t_final,
                                                       int samples, std::string ns, Eigen::Vector3d& color);

mt::PieceWisePol createPwpFromStaticPosition(const mt::state& current_state);

mt::PieceWisePol pwpMsg2Pwp(const panther_msgs::PieceWisePolTraj& pwp_msg);

panther_msgs::PieceWisePolTraj pwp2PwpMsg(const mt::PieceWisePol& pwp);

visualization_msgs::Marker edges2Marker(const mt::Edges& edges, std_msgs::ColorRGBA color_marker);

geometry_msgs::Pose identityGeometryMsgsPose();

mt::PieceWisePol composePieceWisePol(const double t, const double dc, mt::PieceWisePol& p1, mt::PieceWisePol& p2);

std::vector<std::string> pieceWisePol2String(const mt::PieceWisePol& piecewisepol);

std_msgs::ColorRGBA getColorJet(double v, double vmin, double vmax);

std_msgs::ColorRGBA getColorInterpBetween2Colors(double v, double vmin, double vmax, std_msgs::ColorRGBA min_color,
                                                 std_msgs::ColorRGBA max_color);
std_msgs::ColorRGBA color(int id);

void quaternion2Euler(tf2::Quaternion q, double& roll, double& pitch, double& yaw);

void quaternion2Euler(geometry_msgs::Quaternion q, double& roll, double& pitch, double& yaw);

void quaternion2Euler(Eigen::Quaterniond q, double& roll, double& pitch, double& yaw);

visualization_msgs::Marker getMarkerSphere(double scale, int my_color);

geometry_msgs::Point pointOrigin();

Eigen::Vector3d vec2eigen(geometry_msgs::Vector3 vector);

geometry_msgs::Vector3 eigen2rosvector(Eigen::Vector3d vector);

geometry_msgs::Point eigen2point(Eigen::Vector3d vector);

geometry_msgs::Vector3 vectorNull();

geometry_msgs::Vector3 vectorUniform(double a);

visualization_msgs::MarkerArray trajectory2ColoredMarkerArray(const mt::trajectory& data, double max_value, int increm,
                                                              std::string ns, double scale, std::string color_type,
                                                              int id_agent, int n_agents, double min_cost = 0.0,
                                                              double max_cost = 0.0, double cost = 0.0,
                                                              bool collides = false);
