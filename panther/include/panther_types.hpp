/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <iostream>
#include <iomanip>  // std::setprecision
#include <deque>
#include <vector>
#include "termcolor.hpp"
#include <Eigen/Dense>
#include "timer.hpp"

#include "mparser.hpp"
// #include <any>
// #include <utility>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Polyhedron_Std;
typedef std::vector<Polyhedron_Std> ConvexHullsOfCurve_Std;
typedef std::vector<ConvexHullsOfCurve_Std> ConvexHullsOfCurves_Std;

// Same as above, but with different names
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> VertexesInterval;
typedef std::vector<Polyhedron_Std> VertexesObstacle;
typedef std::vector<ConvexHullsOfCurve_Std> VertexesObstacles;

enum DroneStatus
{
  YAWING = 0,
  TRAVELING = 1,
  GOAL_SEEN = 2,
  GOAL_REACHED = 3
};

enum PlannerStatus
{
  FIRST_PLAN = 0,
  START_REPLANNING = 1,
  REPLANNED = 2
};

namespace mt  // panther_types
{
typedef std::pair<Eigen::VectorXd, Eigen::VectorXd> Edge;
typedef std::vector<Edge> Edges;

struct log
{
  log(){};
  bool replanning_was_needed = false;

  PANTHER_timers::Timer tim_initial_setup;           //
  PANTHER_timers::Timer tim_convex_hulls;            //
  PANTHER_timers::Timer tim_opt;                     //
  PANTHER_timers::Timer tim_guess_pos;               //
  PANTHER_timers::Timer tim_guess_yaw_search_graph;  //
  PANTHER_timers::Timer tim_guess_yaw_fit_poly;      //
  PANTHER_timers::Timer tim_total_replan;            //

  double cost = 0.0;
  double obst_avoidance_violation = 0.0;
  double dyn_lim_violation = 0.0;

  Eigen::VectorXd tracking_now_pos;
  Eigen::VectorXd tracking_now_vel;

  Eigen::VectorXd pos;         // Current position of the UAV
  Eigen::VectorXd G_term_pos;  // Position of the terminal goal

  bool success_guess_pos = false;   //
  bool success_guess_yaw = false;   //
  bool success_opt = false;         //
  bool success_replanning = false;  //
  std::string info_replan = "";     //
  int drone_status;                 //
};

struct obstacleForOpt
{
  // casadi::DM bbox_inflated;
  // casadi::DM ctrl_pts;
  std::vector<Eigen::VectorXd> ctrl_pts;
  Eigen::VectorXd bbox_inflated;

  void printInfo()
  {
    std::cout << termcolor::yellow << "ctrl_pts=" << termcolor::reset << std::endl;
    for (auto& q : ctrl_pts)
    {
      std::cout << termcolor::yellow << q.transpose() << termcolor::reset << std::endl;
    }
    std::cout << termcolor::blue << "bbox_inflated=" << bbox_inflated.transpose() << termcolor::reset << std::endl;
  }
};

struct state
{
  Eigen::VectorXd pos;
  Eigen::VectorXd vel;
  Eigen::VectorXd accel;
  Eigen::VectorXd jerk;

  double yaw = 0;
  double dyaw = 0;
  double ddyaw = 0;

  state()
  {
  }

  void setPos(const double x, const double y)
  {
    pos.resize(2);
    pos << x, y;
  }

  void setPos(const double x, const double y, const double z)
  {
    pos.resize(3);
    pos << x, y, z;
  }
  
  void setVel(const double x, const double y)
  {
    vel.resize(2);
    vel << x, y;
  }

  void setVel(const double x, const double y, const double z)
  {
    vel.resize(3);
    vel << x, y, z;
  }

  void setAccel(const double x, const double y)
  {
    accel.resize(2);
    accel << x, y;
  }

  void setAccel(const double x, const double y, const double z)
  {
    accel.resize(3);
    accel << x, y, z;
  }

  void setJerk(const double x, const double y)
  {
    jerk.resize(2);
    jerk << x, y;
  }

  void setJerk(const double x, const double y, const double z)
  {
    jerk.resize(3);
    jerk << x, y, z;
  }

  void setPos(const Eigen::VectorXd& data)
  {
    assert(data.size() == 2 || data.size() == 3);
    pos.resize(data.size());
    if (data.size() == 2) {
      pos << data.x(), data.y();
    } else {
      pos << data.x(), data.y(), data.z();
    }
  }

  void setVel(const Eigen::VectorXd& data)
  {
    assert(data.size() == 2 || data.size() == 3);
    vel.resize(data.size());
    if (data.size() == 2) {
      vel << data.x(), data.y();
    } else {
      vel << data.x(), data.y(), data.z();
    }
  }

  void setAccel(const Eigen::VectorXd& data)
  {
    assert(data.size() == 2 || data.size() == 3);
    accel.resize(data.size());
    if (data.size() == 2) {
      accel << data.x(), data.y();
    } else {
      accel << data.x(), data.y(), data.z();
    }
  }

  void setJerk(const Eigen::VectorXd& data)
  {
    assert(data.size() == 2 || data.size() == 3);
    jerk.resize(data.size());
    if (data.size() == 2) {
      jerk << data.x(), data.y();
    } else {
      jerk << data.x(), data.y(), data.z();
    }
  }

  void setState(const Eigen::Matrix<double, 6, 1>& data)
  {
    pos.resize(2);
    pos << data(0, 0), data(1, 0);
    vel.resize(2);
    vel << data(2, 0), data(3, 0);
    accel.resize(2);
    accel << data(4, 0), data(5, 0);
  }

  void setState(const Eigen::Matrix<double, 9, 1>& data)
  {
    pos.resize(3);
    pos << data(0, 0), data(1, 0), data(2, 0);
    vel.resize(3);
    vel << data(3, 0), data(4, 0), data(5, 0);
    accel.resize(3);
    accel << data(6, 0), data(7, 0), data(8, 0);
  }

  void setYaw(const double& data)
  {
    yaw = data;
  }
  
  void setDYaw(const double& data)
  {
    dyaw = data;
  }
  
  void setDDYaw(const double& data)
  {
    ddyaw = data;
  }

  void setZero()
  {
    assert(pos.size() == vel.size() && vel.size() == accel.size() && accel.size() == jerk.size());
    assert(pos.size() > 0);
    setZero(pos.size());
  }

  void setZero(int dim)
  {
    pos.resize(dim);
    pos.setZero();
    vel.resize(dim);
    vel.setZero();
    accel.resize(dim);
    accel.setZero();
    jerk.resize(dim);
    jerk.setZero();
    yaw = 0;
    dyaw = 0;
    ddyaw = 0;
  }

  const void printPos()
  {
    std::cout << "Pos= " << pos.transpose() << std::endl;
  }

  const void print()
  {
    std::cout << std::setprecision(3) << "Pos= " << pos.transpose() << std::endl;
    std::cout << std::setprecision(3) << "Vel= " << vel.transpose() << std::endl;
    std::cout << std::setprecision(3) << "Accel= " << accel.transpose() << std::endl;
    std::cout << std::setprecision(3) << "Yaw= " << yaw << std::endl;
    std::cout << std::setprecision(3) << "DYaw= " << dyaw << std::endl;
  }

  const void printHorizontal()
  {
    using namespace termcolor;
    std::cout << std::setprecision(3) << red << "Pos" << reset << ", " << blue << "Vel" << reset << ", " << green
              << "Accel" << reset << ", " << white << "Jerk" << reset << ", " << yellow << "Yaw" << reset << ", "
              << magenta << "DYaw" << reset << "= " << red << pos.transpose() << reset;
    std::cout << " " << std::setprecision(3) << blue << vel.transpose() << reset;
    std::cout << " " << std::setprecision(3) << green << accel.transpose() << reset;
    std::cout << " " << std::setprecision(3) << jerk.transpose() << reset;
    std::cout << " " << std::setprecision(3) << yellow << yaw << reset;
    std::cout << " " << std::setprecision(3) << magenta << dyaw << reset << std::endl;
  }
};

// TODO: move this to a class (so that no one can modify these matrices)
struct basisConverter
{
  Eigen::Matrix<double, 2, 2> A_mv_deg1_rest;
  Eigen::Matrix<double, 2, 2> A_be_deg1_rest;
  Eigen::Matrix<double, 2, 2> A_bs_deg1_rest;

  Eigen::Matrix<double, 3, 3> A_mv_deg2_rest;
  Eigen::Matrix<double, 3, 3> A_be_deg2_rest;
  Eigen::Matrix<double, 3, 3> A_bs_deg2_rest;

  Eigen::Matrix<double, 4, 4> A_mv_deg3_rest;
  Eigen::Matrix<double, 4, 4> A_be_deg3_rest;
  Eigen::Matrix<double, 4, 4> A_bs_deg3_seg0, A_bs_deg3_seg1, A_bs_deg3_rest, A_bs_deg3_seg_last2, A_bs_deg3_seg_last;

  Eigen::Matrix<double, 4, 4> M_bs2mv_deg3_seg0, M_bs2mv_deg3_seg1, M_bs2mv_deg3_rest, M_bs2mv_deg3_seg_last2,
      M_bs2mv_deg3_seg_last;

  Eigen::Matrix<double, 4, 4> M_bs2be_deg3_seg0, M_bs2be_deg3_seg1, M_bs2be_deg3_rest, M_bs2be_deg3_seg_last2,
      M_bs2be_deg3_seg_last;

  Eigen::Matrix<double, 3, 3> M_bs2mv_deg2_seg0, M_bs2mv_deg2_rest, M_bs2mv_deg2_seg_last;
  Eigen::Matrix<double, 3, 3> M_bs2be_deg2_seg0, M_bs2be_deg2_rest, M_bs2be_deg2_seg_last;

  basisConverter()
  {
    // See matlab.
    // All these matrices are for t \in [0 1];

    // clang-format off

        //////MATRICES A FOR MINVO Degree 2 and 3///////// (there is only one per degree)
        A_mv_deg3_rest << 

     -3.4416308968564117698463178385282,  6.9895481477801393310755884158425, -4.4622887507045296828778191411402,                   0.91437149978080234369315348885721,
      6.6792587327074839365081970754545, -11.845989901556746914934592496138,  5.2523596690684613008670567069203,                                                    0,
     -6.6792587327074839365081970754545,  8.1917862965657040064115790301003, -1.5981560640774179482548333908198,                  0.085628500219197656306846511142794,
      3.4416308968564117698463178385282, -3.3353445427890959784633650997421, 0.80808514571348655231020075007109, -0.0000000000000000084567769453869345852581318467855;

      A_mv_deg2_rest <<   

     1.4999999992328318931811281800037, -2.3660254034601950756666610686807,  0.93301270211368159124276644433849,
    -2.9999999984656637863622563600074,  2.9999999984656637863622563600074,                                   0,
     1.4999999992328318931811281800037, -0.6339745950054685996732928288111, 0.066987297886318325490506708774774;

      A_mv_deg1_rest <<   
           -1.0, 1.0,
            1.0, 0.0;

        //////MATRICES A FOR Bezier Degree 2 and 3///////// (there is only one per degree)
        A_be_deg3_rest << 

           -1.0,  3.0, -3.0, 1.0,
            3.0, -6.0,  3.0,   0,
           -3.0,  3.0,    0,   0,
            1.0,    0,    0,   0;

        A_be_deg2_rest << 

             1.0, -2.0, 1.0,
            -2.0,  2.0,   0,
             1.0,    0,   0;

      A_be_deg1_rest <<   
           -1.0, 1.0,
            1.0, 0.0;

        //////MATRICES A FOR BSPLINE Degree 2 and 3/////////
        A_bs_deg3_seg0 <<

           -1.0000,    3.0000,   -3.0000,    1.0000,
            1.7500,   -4.5000,    3.0000,         0,
           -0.9167,    1.5000,         0,         0,
            0.1667,         0,         0,         0;

        A_bs_deg3_seg1 <<

           -0.2500,    0.7500,   -0.7500,    0.2500,
            0.5833,   -1.2500,    0.2500,    0.5833,
           -0.5000,    0.5000,    0.5000,    0.1667,
            0.1667,         0,         0,         0;

        A_bs_deg3_rest << 

           -0.1667,    0.5000,   -0.5000,    0.1667,
            0.5000,   -1.0000,         0,    0.6667,
           -0.5000,    0.5000,    0.5000,    0.1667,
            0.1667,         0,         0,         0;

        A_bs_deg3_seg_last2 <<
           -0.1667,    0.5000,   -0.5000,    0.1667,
            0.5000,   -1.0000,    0.0000,    0.6667,
           -0.5833,    0.5000,    0.5000,    0.1667,
            0.2500,         0,         0,         0;

        A_bs_deg3_seg_last <<

           -0.1667,    0.5000,   -0.5000,   0.1667,
            0.9167,   -1.2500,   -0.2500,   0.5833,
           -1.7500,    0.7500,    0.7500,   0.2500,
            1.0000,         0,         0,        0;

        A_bs_deg2_rest <<

           0.5, -1.0, 0.5,
          -1.0,  1.0, 0.5,
           0.5,    0,   0;

      A_bs_deg1_rest <<   
           -1.0, 1.0,
            1.0, 0.0;

        //TODO: Add also 2_seg_last,.... for degree = 2 (although I'm not using them right now)

        //////BSPLINE to MINVO Degree 3/////////

        M_bs2mv_deg3_seg0 <<

         1.1023313949144333268037598827505,   0.34205724556666972091534262290224, -0.092730934245582874453361910127569, -0.032032766697130621302846975595457,
      -0.049683556253749178166501110354147,   0.65780347324677179710050722860615,   0.53053863760186903419935333658941,   0.21181027098212013015654520131648,
      -0.047309044211162346038612724896666,  0.015594436894155586093013710069499,    0.5051827557159349613158383363043,   0.63650059656260427054519368539331,
     -0.0053387944495217444854096022766043, -0.015455155707597083292181849856206,  0.057009540927778303009976212933907,   0.18372189915240558222286892942066;

        M_bs2mv_deg3_seg1 <<

        0.27558284872860833170093997068761,  0.085514311391667430228835655725561, -0.023182733561395718613340477531892, -0.0080081916742826553257117438988644,
         0.6099042761975865811763242163579,   0.63806904207840509091198555324809,   0.29959938009132258684985572472215,    0.12252106674808682651445224109921,
        0.11985166952332682033244282138185,   0.29187180223752445806795208227413,   0.66657381254229419731416328431806,    0.70176522577378930289881964199594,
     -0.0053387944495217444854096022766043, -0.015455155707597083292181849856206,  0.057009540927778303009976212933907,    0.18372189915240558222286892942066;

        M_bs2mv_deg3_rest <<

        0.18372189915240555446729331379174,  0.057009540927778309948870116841135, -0.015455155707597117986651369392348, -0.0053387944495218164764338553140988,
        0.70176522577378919187651717948029,   0.66657381254229419731416328431806,   0.29187180223752384744528853843804,    0.11985166952332582113172065874096,
        0.11985166952332682033244282138185,   0.29187180223752445806795208227413,   0.66657381254229419731416328431806,    0.70176522577378930289881964199594,
     -0.0053387944495217444854096022766043, -0.015455155707597083292181849856206,  0.057009540927778303009976212933907,    0.18372189915240558222286892942066;


        M_bs2mv_deg3_seg_last2 <<

        0.18372189915240569324517139193631,  0.057009540927778309948870116841135, -0.015455155707597145742226985021261, -0.0053387944495218164764338553140988,
        0.70176522577378952494342456702725,   0.66657381254229453038107067186502,   0.29187180223752412500104469472717,    0.11985166952332593215402312125661,
         0.1225210667480875342816304396365,   0.29959938009132280889446064975346,   0.63806904207840497988968309073243,    0.60990427619758624810941682881094,
     -0.0080081916742826154270717964323012, -0.023182733561395621468825822830695,  0.085514311391667444106623463540018,    0.27558284872860833170093997068761;

        M_bs2mv_deg3_seg_last <<

       0.18372189915240555446729331379174, 0.057009540927778309948870116841135, -0.015455155707597117986651369392348, -0.0053387944495218164764338553140988,
       0.63650059656260415952289122287766,   0.5051827557159349613158383363043,  0.015594436894155294659469745965907,  -0.047309044211162887272337229660479,
       0.21181027098212068526805751389475,  0.53053863760186914522165579910506,   0.65780347324677146403359984105919,  -0.049683556253749622255710960416764,
     -0.032032766697130461708287185729205, -0.09273093424558248587530329132278,   0.34205724556666977642649385416007,     1.1023313949144333268037598827505;


        //////BSPLINE to BEZIER Degree 3/////////

        M_bs2be_deg3_seg0 <<

            1.0000,    0.0000,   -0.0000,         0,
                 0,    1.0000,    0.5000,    0.2500,
                 0,   -0.0000,    0.5000,    0.5833,
                 0,         0,         0,    0.1667;

        M_bs2be_deg3_seg1 <<

            0.2500,    0.0000,   -0.0000,         0,
            0.5833,    0.6667,    0.3333,    0.1667,
            0.1667,    0.3333,    0.6667,    0.6667,
                 0,         0,         0,    0.1667;

        M_bs2be_deg3_rest <<

            0.1667,    0.0000,         0,         0,
            0.6667,    0.6667,    0.3333,    0.1667,
            0.1667,    0.3333,    0.6667,    0.6667,
                 0,         0,         0,    0.1667;

        M_bs2be_deg3_seg_last2 <<

            0.1667,         0,   -0.0000,         0,
            0.6667,    0.6667,    0.3333,    0.1667,
            0.1667,    0.3333,    0.6667,    0.5833,
                 0,         0,         0,    0.2500;

        M_bs2be_deg3_seg_last <<

            0.1667,    0.0000,         0,         0,
            0.5833,    0.5000,         0,         0,
            0.2500,    0.5000,    1.0000,         0,
                 0,         0,         0,    1.0000;

        /////BSPLINE to MINVO Degree 2
        M_bs2mv_deg2_seg0 <<

    1.077349059083916,  0.1666702138890985, -0.07735049175615138,
 -0.03867488648729411,  0.7499977187062712,   0.5386802643920123,
 -0.03867417280506149, 0.08333206631563977,    0.538670227146185;

        M_bs2mv_deg2_rest <<

    0.538674529541958, 0.08333510694454926, -0.03867524587807569,
   0.4999996430546639,  0.8333328256508203,   0.5000050185139366,
 -0.03867417280506149, 0.08333206631563977,    0.538670227146185;

        M_bs2mv_deg2_seg_last <<

    0.538674529541958, 0.08333510694454926, -0.03867524587807569,
   0.5386738158597254,  0.7500007593351806, -0.03866520863224832,
 -0.07734834561012298,  0.1666641326312795,     1.07734045429237;

      /////BSPLINE to BEZIER Degree 2
        M_bs2be_deg2_seg0 <<

            1.0000,         0,         0,
                 0,    1.0000,    0.5000,
                 0,         0,    0.5000;

        M_bs2be_deg2_rest <<

            0.5000,         0,         0,
            0.5000,    1.0000,    0.5000,
                 0,         0,    0.5000;

        M_bs2be_deg2_seg_last <<

            0.5000,         0,         0,
            0.5000,    1.0000,         0,
                 0,         0,    1.0000;

    // clang-format on
  }

  //////MATRIX A FOR MINVO Deg 2 and 3/////////
  Eigen::Matrix<double, 2, 2> getArestMinvoDeg1()
  {
    return A_mv_deg1_rest;
  }

  Eigen::Matrix<double, 3, 3> getArestMinvoDeg2()
  {
    return A_mv_deg2_rest;
  }

  Eigen::Matrix<double, 4, 4> getArestMinvoDeg3()
  {
    return A_mv_deg3_rest;
  }
  //////MATRIX A FOR Bezier Deg 2 and 3/////////
  Eigen::Matrix<double, 2, 2> getArestBezierDeg1()
  {
    return A_be_deg1_rest;
  }

  Eigen::Matrix<double, 3, 3> getArestBezierDeg2()
  {
    return A_be_deg2_rest;
  }

  Eigen::Matrix<double, 4, 4> getArestBezierDeg3()
  {
    return A_be_deg3_rest;
  }

  //////MATRIX A FOR BSPLINE Deg 2 and 3/////////
  Eigen::Matrix<double, 2, 2> getArestBSplineDeg1()
  {
    return A_bs_deg1_rest;
  }

  Eigen::Matrix<double, 3, 3> getArestBSplineDeg2()
  {
    return A_bs_deg2_rest;
  }

  Eigen::Matrix<double, 4, 4> getArestBSplineDeg3()
  {
    return A_bs_deg3_rest;
  }
  //////MATRICES A FOR MINVO Deg3/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getAMinvoDeg3(int num_seg)
  {
    std::vector<Eigen::Matrix<double, 4, 4>> A_mv_deg3;  // will have as many elements as num_seg
    for (int i = 0; i < num_seg; i++)
    {
      A_mv_deg3.push_back(A_mv_deg3_rest);
    }
    return A_mv_deg3;
  }

  //////MATRICES A FOR Bezier POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getABezierDeg3(int num_seg)
  {
    std::vector<Eigen::Matrix<double, 4, 4>> A_be_deg3;  // will have as many elements as num_seg
    for (int i = 0; i < num_seg; i++)
    {
      A_be_deg3.push_back(A_be_deg3_rest);
    }
    return A_be_deg3;
  }

  //////MATRICES A FOR BSPLINE POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getABSplineDeg3(int num_seg)
  {
    std::vector<Eigen::Matrix<double, 4, 4>> A_bs_deg3;  // will have as many elements as num_seg
    A_bs_deg3.push_back(A_bs_deg3_seg0);
    A_bs_deg3.push_back(A_bs_deg3_seg1);
    for (int i = 0; i < (num_seg - 4); i++)
    {
      A_bs_deg3.push_back(A_bs_deg3_rest);
    }
    A_bs_deg3.push_back(A_bs_deg3_seg_last2);
    A_bs_deg3.push_back(A_bs_deg3_seg_last);
    return A_bs_deg3;
  }

  //////BSPLINE to MINVO POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getMinvoDeg3Converters(int num_seg)
  {
    std::vector<Eigen::Matrix<double, 4, 4>> M_bs2mv_deg3;  // will have as many elements as num_seg
    M_bs2mv_deg3.push_back(M_bs2mv_deg3_seg0);
    M_bs2mv_deg3.push_back(M_bs2mv_deg3_seg1);
    for (int i = 0; i < (num_seg - 4); i++)
    {
      M_bs2mv_deg3.push_back(M_bs2mv_deg3_rest);
    }
    M_bs2mv_deg3.push_back(M_bs2mv_deg3_seg_last2);
    M_bs2mv_deg3.push_back(M_bs2mv_deg3_seg_last);
    return M_bs2mv_deg3;
  }

  //////BSPLINE to BEZIER POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getBezierDeg3Converters(int num_seg)
  {
    std::vector<Eigen::Matrix<double, 4, 4>> M_bs2be_deg3;  // will have as many elements as num_seg
    M_bs2be_deg3.push_back(M_bs2be_deg3_seg0);
    M_bs2be_deg3.push_back(M_bs2be_deg3_seg1);
    for (int i = 0; i < (num_seg - 4); i++)
    {
      M_bs2be_deg3.push_back(M_bs2be_deg3_rest);
    }
    M_bs2be_deg3.push_back(M_bs2be_deg3_seg_last2);
    M_bs2be_deg3.push_back(M_bs2be_deg3_seg_last);
    return M_bs2be_deg3;
  }

  //////BSPLINE to BSPLINE POSITION/////////
  std::vector<Eigen::Matrix<double, 4, 4>> getBSplineDeg3Converters(int num_seg)
  {
    std::vector<Eigen::Matrix<double, 4, 4>> M_pos_bs2bs;  // will have as many elements as num_seg
    for (int i = 0; i < num_seg; i++)
    {
      M_pos_bs2bs.push_back(Eigen::Matrix<double, 4, 4>::Identity());
    }
    return M_pos_bs2bs;
  }

  //////BSPLINE to MINVO Velocity/////////
  std::vector<Eigen::Matrix<double, 3, 3>> getMinvoDeg2Converters(int num_seg)
  {
    std::vector<Eigen::Matrix<double, 3, 3>> M_bs2mv_deg2;  // will have as many elements as num_seg
    M_bs2mv_deg2.push_back(M_bs2mv_deg2_seg0);
    for (int i = 0; i < (num_seg - 2 - 1); i++)
    {
      M_bs2mv_deg2.push_back(M_bs2mv_deg2_rest);
    }
    M_bs2mv_deg2.push_back(M_bs2mv_deg2_seg_last);
    return M_bs2mv_deg2;
  }

  //////BSPLINE to BEZIER Velocity/////////
  std::vector<Eigen::Matrix<double, 3, 3>> getBezierDeg2Converters(int num_seg)
  {
    std::vector<Eigen::Matrix<double, 3, 3>> M_bs2be_deg2;  // will have as many elements as segments
    M_bs2be_deg2.push_back(M_bs2be_deg2_seg0);
    for (int i = 0; i < (num_seg - 2 - 1); i++)
    {
      M_bs2be_deg2.push_back(M_bs2be_deg2_rest);
    }
    M_bs2be_deg2.push_back(M_bs2be_deg2_seg_last);
    return M_bs2be_deg2;
  }

  //////BSPLINE to BSPLINE Velocity/////////
  std::vector<Eigen::Matrix<double, 3, 3>> getBSplineDeg2Converters(int num_seg)
  {
    std::vector<Eigen::Matrix<double, 3, 3>> M_vel_bs2bs;  // will have as many elements as num_seg
    for (int i = 0; i < num_seg; i++)
    {
      M_vel_bs2bs.push_back(Eigen::Matrix<double, 3, 3>::Identity());
    }
    return M_vel_bs2bs;
  }
};

struct polytope
{
  Eigen::MatrixXd A;
  Eigen::MatrixXd b;
};

struct PieceWisePol
{
  // Interval 0: t\in[t0, t1)
  // Interval 1: t\in[t1, t2)
  // Interval 2: t\in[t2, t3)
  //...
  // Interval n-1: t\in[tn, tn+1)

  // n intervals in total

  // times has n+1 elements
  std::vector<double> times;  // [t0,t1,t2,...,tn+1]
  int dim_;

  // coefficients has n elements
  // The coeffients are such that pol(t)=coeff_of_that_interval*[u^deg u^{deg-1} ... u 1]
  // with u=(t-t_min_that_interval)/(t_max_that_interval- t_min_that_interval)
  std::vector<Eigen::VectorXd> all_coeff_x;  // [a b c d ...]' of Int0 , [a b c d ...]' of Int1,...
  std::vector<Eigen::VectorXd> all_coeff_y;  // [a b c d ...]' of Int0 , [a b c d ...]' of Int1,...
  std::vector<Eigen::VectorXd> all_coeff_z;  // [a b c d ...]' of Int0 , [a b c d ...]' of Int1,...

  void setAllCoefficients(std::vector<std::vector<Eigen::VectorXd>> all_coeffs)
  {
    if (all_coeffs.size() == 2)
    {
      dim_ = 2;
      setAllCoefficients(all_coeffs[0], all_coeffs[1]);
    } else {
      dim_ = 3;
      setAllCoefficients(all_coeffs[0], all_coeffs[1], all_coeffs[2]);
    }
  }

  void setAllCoefficients(std::vector<Eigen::VectorXd> all_coeff_x_, std::vector<Eigen::VectorXd> all_coeff_y_)
  {
    dim_ = 2;
    all_coeff_x = all_coeff_x_;
    all_coeff_y = all_coeff_y_;
    all_coeff_z.clear();
  }

  void setAllCoefficients(std::vector<Eigen::VectorXd> all_coeff_x_, std::vector<Eigen::VectorXd> all_coeff_y_, std::vector<Eigen::VectorXd> all_coeff_z_)
  {
    dim_ = 3;
    all_coeff_x = all_coeff_x_;
    all_coeff_y = all_coeff_y_;
    all_coeff_z = all_coeff_z_;
  }

  void clear()
  {
    times.clear();
    all_coeff_x.clear();
    all_coeff_y.clear();
    all_coeff_z.clear();
  }

  int getDeg() const
  {
    return (all_coeff_x.back().rows() - 1);  // should be the same for y and z
  }

  int getDim() const
  {
    return dim_;
  }

  int getNumOfIntervals() const
  {
    return (all_coeff_x.size());  // should be the same for y and z
  }

  Eigen::VectorXd getU(double u) const
  {
    int degree = getDeg();
    Eigen::VectorXd U(degree + 1);

    for (int i = 0; i <= degree; i++)
    {
      U(i) = pow(u, degree - i);
    }

    return U;
  }

  int getInterval(double t) const
  {
    if (t >= times[times.size() - 1])
    {
      return (getNumOfIntervals() - 1);
    }
    if (t < times[0])
    {
      return 0;
    }
    for (int i = 0; i < (times.size() - 1); i++)
    {
      if (times[i] <= t && t < times[i + 1])
      {
        return i;
      }
    }
    std::cout << "[getInterval] THIS SHOULD NEVER HAPPEN" << std::endl;
    abort();
    return 0;
  }

  void saturateMinMax(double& var, const double min, const double max) const
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

  double t2u(double t) const
  {
    int j = getInterval(t);
    double u = (t - times[j]) / (times[j + 1] - times[j]);
    // if (u > 1.0 || u < 0.0)
    // {
    //   std::cout << "[t2u] THIS SHOULD NEVER HAPPENN, u=" << u << std::endl;
    //   std::cout << "[t2u] j=" << j << std::endl;
    //   std::cout << "t=" << t << std::endl;
    //   std::cout << "(times[j + 1] - times[j])=" << (times[j + 1] - times[j]) << std::endl;
    //   std::cout << "times[j]=" << times[j] << std::endl;
    //   print();
    //   // abort();
    // }

    // Saturate
    saturateMinMax(u, 0.0, 1.0);

    return u;
  }

  Eigen::Vector3d eval(double t) const
  {
    Eigen::Vector3d result;

    double tt = t;
    // Saturate
    saturateMinMax(tt, times[0], times[times.size() - 1]);

    double u = t2u(tt);
    int j = getInterval(tt);  // TODO [Improve efficiency]: note that t2u is already calling getInterval()

    // std::cout << "tt= " << tt << std::endl;
    // std::cout << "u= " << u << std::endl;
    // std::cout << "j= " << j << std::endl;

    Eigen::VectorXd tmp = getU(u);

    result.x() = all_coeff_x[j].transpose() * tmp;
    result.y() = all_coeff_y[j].transpose() * tmp;
    result.z() = all_coeff_z[j].transpose() * tmp;
    return result;
  }

  void print() const
  {
    std::cout << "all_coeff_x.size()= " << all_coeff_x.size() << std::endl;
    std::cout << "times.size()= " << times.size() << std::endl;

    for (int i = 0; i < (times.size() - 1); i++)
    {
      std::cout << "From " << times[i] << " to " << times[i + 1] << std::endl;
      std::cout << "  all_coeff_x= " << all_coeff_x[i].transpose() << std::endl;
      std::cout << "  all_coeff_y= " << all_coeff_y[i].transpose() << std::endl;
      std::cout << "  all_coeff_z= " << all_coeff_z[i].transpose() << std::endl;
    }
  }
};

struct dynTraj
{
  bool use_pwp_field;  // If true, pwp is used. If false, the string is used

  std::vector<std::string> s_mean;
  std::vector<std::string> s_var;
  mt::PieceWisePol pwp_mean;
  mt::PieceWisePol pwp_var;

  Eigen::Vector3d bbox;
  int id;
  double time_received;  // time at which this trajectory was received from an agent
  bool is_agent;         // true for a trajectory of an agent, false for an obstacle
};

struct dynTrajCompiled
{
  bool use_pwp_field;
  // std::vector<exprtk::expression<double>> s_mean;
  // std::vector<exprtk::expression<double>> s_var;
  std::vector<MathEvaluator> s_mean;
  std::vector<MathEvaluator> s_var;

  mt::PieceWisePol pwp_mean;
  mt::PieceWisePol pwp_var;

  Eigen::VectorXd bbox;
  int id;
  double time_received;  // time at which this trajectory was received from an agent
  bool is_agent;         // true for a trajectory of an agent, false for an obstacle
  bool is_static;
};

// struct mt::PieceWisePolWithInfo
// {
//   mt::PieceWisePol pwp;

//   Eigen::Vector3d bbox;
//   int id;
//   double time_received;  // time at which this trajectory was received from an agent
//   bool is_agent;         // true for a trajectory of an agent, false for an obstacle
//   bool is_static;
// };

// Could be improved using C++17:
// https://raymii.org/s/articles/Store_multiple_types_in_a_single_stdmap_in_cpp_just_like_a_python_dict.html
// (But the problem is that pybind11 doesn't have good compatibility with std::any() yet)
// Or doing the second part of this: https://stackoverflow.com/a/50871576/6057617

struct parameters
{
  //
  // clang-format off
  bool            use_ff;                             //void setVar_use_ff(const std::string& value) { use_ff = string2bool(value); };
  bool            visual;                             //void setVar_visual(const std::string& value) { visual = string2bool(value); };
  std::string     color_type_expert;                         //void setVar_color_type(const std::string& value) { color_type = value; };
  std::string     color_type_student;                         //void setVar_color_type(const std::string& value) { color_type = value; };
  int             n_agents;                           //void setVar_n_agents(const std::string& value) { n_agents = std::stoi(value); };
  int             num_of_trajs_per_replan;            //void setVar_num_of_trajs_per_replan(const std::string& value) { num_of_trajs_per_replan = std::stoi(value); };
  int             max_num_of_initial_guesses;            //void setVar_num_of_trajs_per_replan(const std::string& value) { num_of_trajs_per_replan = std::stoi(value); };
  double          dc;                                 //void setVar_dc(const std::string& value) { dc = std::stod(value); };
  double          goal_radius;                        //void setVar_goal_radius(const std::string& value) { goal_radius = std::stod(value); };
  double          drone_radius;                       //void setVar_drone_radius(const std::string& value) { drone_radius = std::stod(value); };//
  double          drone_extra_radius_for_NN;                       //void setVar_drone_radius(const std::string& value) { drone_radius = std::stod(value); };//
  double          Ra;                                 //void setVar_Ra(const std::string& value) { Ra = std::stod(value); };
  bool            impose_FOV_in_trajCB;               //void setVar_impose_FOV_in_trajCB(const std::string& value) { impose_FOV_in_trajCB = string2bool(value); };
  bool            pause_time_when_replanning;          //void setVar_pause_time_when_replanning(const std::string& value) { pause_time_when_replanning = string2bool(value); };
  double          replanning_trigger_time_student;    //void setVar_replanning_trigger_time(const std::string& value) { replanning_trigger_time = std::stod(value); };
  double          replanning_trigger_time_expert;     //void setVar_replanning_trigger_time(const std::string& value) { replanning_trigger_time = std::stod(value); };
  double          replanning_lookahead_time;          //void setVar_replanning_lookahead_time(const std::string& value) { replanning_lookahead_time = std::stod(value); };
  double          max_runtime_octopus_search;         //void setVar_max_runtime_octopus_search(const std::string& value) { max_runtime_octopus_search = std::stod(value); };
  double          fov_x_deg;                          //void setVar_fov_x_deg(const std::string& value) { fov_x_deg = std::stod(value); };
  double          fov_y_deg;                          //void setVar_fov_y_deg(const std::string& value) { fov_y_deg = std::stod(value); };
  double          fov_depth;                          //void setVar_fov_depth(const std::string& value) { fov_depth = std::stod(value); };
  double          angle_deg_focus_front;              //void setVar_angle_deg_focus_front(const std::string& value) { angle_deg_focus_front = std::stod(value); };
  double          x_min;                              //void setVar_x_min(const std::string& value) { x_min = std::stod(value); };
  double          x_max;                              //void setVar_x_max(const std::string& value) { x_max = std::stod(value); };
  double          y_min;                              //void setVar_y_min(const std::string& value) { y_min = std::stod(value); };
  double          y_max;                              //void setVar_y_max(const std::string& value) { y_max = std::stod(value); };
  double          z_min;                              //void setVar_z_min(const std::string& value) { z_min = std::stod(value); };
  double          z_max;                              //void setVar_z_max(const std::string& value) { z_max = std::stod(value); };
  double          ydot_max;                           //void setVar_ydot_max(const std::string& value) { ydot_max = std::stod(value); };
  Eigen::VectorXd v_max;                              
  Eigen::VectorXd a_max;
  Eigen::VectorXd j_max;
  double          factor_alpha;                       //void setVar_factor_alpha(const std::string& value) { factor_alpha = std::stod(value); };
  double          max_seconds_keeping_traj;           //void setVar_ydot_max(const std::string& value) { ydot_max = std::stod(value); };
  int             a_star_samp_x;                      //void setVar_a_star_samp_x(const std::string& value) { a_star_samp_x = std::stoi(value); };
  int             a_star_samp_y;                      //void setVar_a_star_samp_y(const std::string& value) { a_star_samp_y = std::stoi(value); };
  int             a_star_samp_z;                      //void setVar_a_star_samp_z(const std::string& value) { a_star_samp_z = std::stoi(value); };
  double          a_star_fraction_voxel_size;         //void setVar_a_star_fraction_voxel_size(const std::string& value) { a_star_fraction_voxel_size = std::stod(value); };
  double          a_star_bias;                        //void setVar_a_star_bias(const std::string& value) { a_star_bias = std::stod(value); };
  double          res_plot_traj;                      //void setVar_res_plot_traj(const std::string& value) { res_plot_traj = std::stod(value); };
  double          factor_alloc;                       //void setVar_factor_alloc(const std::string& value) { factor_alloc = std::stod(value); };
  double          alpha_shrink;                       //void setVar_alpha_shrink(const std::string& value) { alpha_shrink = std::stod(value); };
  double          norminv_prob;                       //void setVar_norminv_prob(const std::string& value) { norminv_prob = std::stod(value); };
  int             disc_pts_per_interval_oct_search;   //void setVar_disc_pts_per_interval_oct_search(const std::string& value) { disc_pts_per_interval_oct_search = std::stoi(value); };
  double          c_smooth_yaw_search;                //void setVar_c_smooth_yaw_search(const std::string& value) { c_smooth_yaw_search = std::stod(value); };
  double          c_visibility_yaw_search;            //void setVar_c_visibility_yaw_search(const std::string& value) { c_visibility_yaw_search = std::stod(value); };
  double          c_maxydot_yaw_search;               //void setVar_c_maxydot_yaw_search(const std::string& value) { c_maxydot_yaw_search = std::stod(value); }; 
  double          c_pos_smooth;                       //void setVar_c_pos_smooth(const std::string& value) { c_pos_smooth = std::stod(value); };
  double          c_yaw_smooth;                       //void setVar_c_yaw_smooth(const std::string& value) { c_yaw_smooth = std::stod(value); };
  double          c_fov;                              //void setVar_c_fov(const std::string& value) { c_fov = std::stod(value); };
  double          c_final_pos;                        //void setVar_c_final_pos(const std::string& value) { c_final_pos = std::stod(value); };
  double          c_final_yaw;                        //void setVar_c_final_yaw(const std::string& value) { c_final_yaw = std::stod(value); };
  double          c_total_time;                       //void setVar_c_total_time(const std::string& value) { c_total_time = std::stod(value); };
  bool            print_graph_yaw_info;               //void setVar_print_graph_yaw_info(const std::string& value) { print_graph_yaw_info = string2bool(value); };
  double          z_goal_when_using_rviz;                  
  std::string     mode;                               //void setVar_mode(const std::string& value) { mode = value; };
  Eigen::Matrix4d b_T_c;                              //Computed inside C++
  std::string     basis;                              //From Casadi                //void setVar_basis(const std::string& value) { basis = value; };
  int             num_max_of_obst;                    //From Casadi                //void setVar_num_max_of_obst(const std::string& value) { num_max_of_obst = std::stoi(value); };
  int             num_seg;                            //From Casadi                    //void setVar_num_seg(const std::string& value) { num_seg = std::stoi(value); };
  int             deg_pos;                            //From Casadi                    //void setVar_deg_pos(const std::string& value) { deg_pos = std::stoi(value); };
  int             deg_yaw;                            //From Casadi                   //void setVar_deg_yaw(const std::string& value) { deg_yaw = std::stoi(value); };
  int             num_of_yaw_per_layer;               //From Casadi               //void setVar_num_of_yaw_per_layer(const std::string& value) { num_of_yaw_per_layer = std::stoi(value); };
  double          fitter_total_time;                  //void setVar_fitter_total_time(const std::string& value) { fitter_total_time = std::stod(value); };
  int             fitter_num_samples;                 //From Casadi               //void setVar_fitter_num_samples(const std::string& value) { fitter_num_samples = std::stoi(value); }; 
  double          fitter_num_seg;                     //From Casadi               //void setVar_fitter_num_seg(const std::string& value) { fitter_num_seg = std::stod(value); };  
  double          fitter_deg_pos;                     //From Casadi               //void setVar_fitter_deg_pos(const std::string& value) { fitter_deg_pos = std::stod(value); };   
  int             sampler_num_samples;                //From Casadi              //void setVar_sampler_num_samples(const std::string& value) { sampler_num_samples = std::stoi(value); };
  double max_dist2goal;
  double max_dist2obs;
  double max_side_bbox_obs;
  double max_dist2BSPoscPoint;
  bool use_expert;
  bool use_student;
  std::string student_policy_path;
  bool static_planning;
  bool use_closed_form_yaw_student;
  double lambda_obst_avoidance_violation;
  double lambda_dyn_lim_violation;
  // clang-format on
};

struct committedTrajectory
{
  std::deque<mt::state> content;

  void print()
  {
    for (auto state_i : content)
    {
      state_i.printHorizontal();
    }
  }

  // now define the functions operating on the member of this struct directly
  int size()
  {
    return content.size();
  }

  void clear()
  {
    content.clear();
  }

  void push_back(mt::state tmp)
  {
    content.push_back(tmp);
  }

  void erase(std::deque<mt::state>::iterator a, std::deque<mt::state>::iterator b)
  {
    content.erase(a, b);
  }

  mt::state front()
  {
    return content.front();
  }

  mt::state back()
  {
    return content.back();
  }

  void pop_front()
  {
    content.pop_front();
  }

  std::deque<mt::state>::iterator end()
  {
    return content.end();
  }

  std::deque<mt::state>::iterator begin()
  {
    return content.begin();
  }

  mt::state get(int i)
  {
    return content[i];
  }

  std::vector<mt::state> toStdVector()
  {
    std::vector<mt::state> my_vector;
    std::copy(content.begin(), content.end(), std::back_inserter(my_vector));
    return my_vector;
  }
};

typedef std::vector<mt::state> trajectory;

}  // namespace mt
