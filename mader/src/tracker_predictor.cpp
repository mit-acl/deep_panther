#include <iostream>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <iterator>

#include <ros/ros.h>
// #include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "pcl_ros/point_cloud.h"
// #include <geometry_msgs/Point.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/geometry.h>
#include <pcl/filters/extract_indices.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
// #include <pcl/sample_consensus/method_types.h>
// #include <pcl/sample_consensus/model_types.h>
// #include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <limits>
#include <utility>

#include <pcl/point_types.h>
#include <pcl/common/common.h>

#include "mader_types.hpp"

#include "Hungarian.h"
#include "tracker_predictor.hpp"

#include <ros/package.h>  //TODO: remove this ros dependency

#include <tf2_eigen/tf2_eigen.h>

#include "utils.hpp"

#include "visualization_msgs/MarkerArray.h"

#include <mader_msgs/DynTraj.h>

using namespace termcolor;

TrackerPredictor::TrackerPredictor(ros::NodeHandle nh) : nh_(nh)
{
  safeGetParam(nh_, "num_seg_prediction", num_seg_prediction_);
  safeGetParam(nh_, "size_sliding_window", size_sliding_window_);
  safeGetParam(nh_, "meters_to_create_new_track", meters_to_create_new_track_);
  safeGetParam(nh_, "max_frames_skipped", max_frames_skipped_);
  safeGetParam(nh_, "cluster_tolerance", cluster_tolerance_);
  safeGetParam(nh_, "min_cluster_size", min_cluster_size_);
  safeGetParam(nh_, "max_cluster_size", max_cluster_size_);

  for (int j = 0; j < num_seg_prediction_; j++)
  {
    std::cout << "j= " << j << std::endl;
    cfs_kkt_Ab_.push_back(casadi::Function::load(ros::package::getPath("mader") + "/matlab/predictor_kkt_Ab_" +
                                                 std::to_string(j + 1) + ".casadi"));  // j+1 because Matlab uses
                                                                                       // 1-indexing
  }

  cf_coeff_predicted_ = casadi::Function::load(ros::package::getPath("mader") + "/matlab/predictor_coeff_"
                                                                                "predicted.casadi");

  tf_listener_ptr_ = std::unique_ptr<tf2_ros::TransformListener>(
      new tf2_ros::TransformListener(tf_buffer_));  // needed (although tf_listener_ptr_ is not used explicitly)

  pub_marker_predicted_traj_ = nh_.advertise<visualization_msgs::MarkerArray>("marker_predicted_traj", 1);
  pub_marker_bbox_obstacles_ = nh_.advertise<visualization_msgs::MarkerArray>("marker_bbox_obstacles", 1);
  pub_traj_ = nh_.advertise<mader_msgs::DynTraj>("trajs_predicted", 1, true);  // The last boolean is latched or not

  tree_ = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>);
}

double TrackerPredictor::getCostRowColum(cluster& a, track& b, double time)
{
  return (a.centroid - b.pwp.eval(time)).norm();
}

void TrackerPredictor::addNewTrack(const cluster& c)
{
  track tmp(size_sliding_window_, c);
  // mt::PieceWisePol pwp;  // will have only one interval
  // pwp.times.push_back(time);
  // pwp.times.push_back(std::numeric_limits<double>::max());  // infty

  // pwp.coeff_x.push_back(Eigen::Vector4d(0.0, 0.0, 0.0, c.centroid.x()));  // [a b c d]' of Int0 , [a b c d]' of
  // Int1,... pwp.coeff_y.push_back(Eigen::Vector4d(0.0, 0.0, 0.0, c.centroid.y()));  // [a b c d]' of Int0 , [a b c d]'
  // of Int1,... pwp.coeff_z.push_back(Eigen::Vector4d(0.0, 0.0, 0.0, c.centroid.z()));  // [a b c d]' of Int0 , [a b c
  // d]' of Int1,...

  // tmp.pwp = pwp;

  // unsigned int last_id = (all_tracks_.size() == 0) ? 0 : (all_tracks_[all_tracks_.rbegin()->.id] + 1);

  generatePredictedPwpForTrack(tmp);

  all_tracks_.push_back(tmp);

  // all_tracks_[last_id] = tmp;
}

void TrackerPredictor::cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input)
{
  std::cout << "-------------------------------" << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

  pcl::fromROSMsg(*input, *input_cloud);

  tree_->setInputCloud(input_cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(tree_);
  ec.setInputCloud(input_cloud);

  /* Extract the clusters out of pc and save indices in cluster_indices.*/
  ec.extract(cluster_indices);

  std::vector<cluster> clusters;

  double time_pcloud = input->header.stamp.toSec();

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
  {
    std::cout << "--- New cluster" << std::endl;
    std::cout << " " << std::endl;

    ///////////////////////
    // Compute bounding box
    ///////////////////////

    // First option (slow):

    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    // for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
    // {
    //   cloud_cluster->points.push_back(input_cloud->points[*pit]);  // TODO: I think I can avoid doing this
    // }

    // cloud_cluster->width = cloud_cluster->points.size();
    // cloud_cluster->height = 1;
    // cloud_cluster->is_dense = true;

    // pcl::PointXYZ minPt, maxPt;

    // pcl::getMinMax3D(*cloud_cluster, minPt, maxPt);

    ////////////////////////
    ////////////////////////

    // Second option, it's faster (taken from
    // https://stackoverflow.com/questions/35669182/this-predefined-function-slowing-down-my-programs-performance   )
    // But note that I've deleted the else() (I think they are wrong)
    double min_x = std::numeric_limits<double>::max();
    double max_x = -std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_y = -std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_z = -std::numeric_limits<double>::max();

    for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
    {
      // std::cout << "input_cloud->points[*pit]= " << input_cloud->points[*pit] << std::endl;
      if (input_cloud->points[*pit].x <= min_x)
      {
        min_x = input_cloud->points[*pit].x;
      }
      if (input_cloud->points[*pit].y <= min_y)
      {
        min_y = input_cloud->points[*pit].y;
      }
      if (input_cloud->points[*pit].z <= min_z)
      {
        min_z = input_cloud->points[*pit].z;
      }
      if (input_cloud->points[*pit].x >= max_x)
      {
        // std::cout << "assigning max_x" << std::endl;
        max_x = input_cloud->points[*pit].x;
      }
      if (input_cloud->points[*pit].y >= max_y)
      {
        // std::cout << "assigning max_y" << std::endl;
        max_y = input_cloud->points[*pit].y;
      }
      if (input_cloud->points[*pit].z >= max_z)
      {
        // std::cout << "assigning max_z" << std::endl;
        max_z = input_cloud->points[*pit].z;
      }
    }

    // std::cout << "min_x= " << min_x << std::endl;
    // std::cout << "min_y= " << min_y << std::endl;
    // std::cout << "min_z= " << min_z << std::endl;
    // std::cout << "max_x= " << max_x << std::endl;
    // std::cout << "max_y= " << max_y << std::endl;
    // std::cout << "max_z= " << max_z << std::endl;

    ////////////////////////
    ////////////////////////

    std::vector<Eigen::Vector4d> vertexes_bbox(8);
    vertexes_bbox[0] = Eigen::Vector4d(max_x, max_y, max_z, 1.0);
    vertexes_bbox[1] = Eigen::Vector4d(max_x, max_y, min_z, 1.0);
    vertexes_bbox[2] = Eigen::Vector4d(max_x, min_y, max_z, 1.0);
    vertexes_bbox[3] = Eigen::Vector4d(max_x, min_y, min_z, 1.0);
    vertexes_bbox[4] = Eigen::Vector4d(min_x, max_y, max_z, 1.0);
    vertexes_bbox[5] = Eigen::Vector4d(min_x, max_y, min_z, 1.0);
    vertexes_bbox[6] = Eigen::Vector4d(min_x, min_y, max_z, 1.0);
    vertexes_bbox[7] = Eigen::Vector4d(min_x, min_y, min_z, 1.0);

    // for (size_t i = 0; i < vertexes_bbox.size(); i++)
    // {
    //   // std::cout << "vertex " << i << " = " << vertexes_bbox[i].transpose() << std::endl;
    // }
    /////Listen to the transform
    geometry_msgs::TransformStamped transform_stamped;
    Eigen::Vector3d transform;

    Eigen::Affine3d w_T_b;

    try
    {
      transform_stamped = tf_buffer_.lookupTransform("world", input->header.frame_id, input->header.stamp,
                                                     ros::Duration(0.02));  // TODO: change this duration time?

      // std::cout << "transform_stamped= " << transform_stamped << std::endl;
      // std::cout << "input->header.frame_id= " << input->header.frame_id << std::endl;

      w_T_b = tf2::transformToEigen(transform_stamped);
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("[world_database_master_ros] OnGetTransform failed with %s", ex.what());
      std::cout << red << "ERROR in TransformException" << reset << std::endl;
      return;
    }

    // std::cout << "w_T_b.translation()= " << w_T_b.translation() << std::endl;
    // std::cout << "w_T_b.rotation()= " << w_T_b.rotation() << std::endl;

    // apply r_T_w to the vertexes of the bbox (this is much faster than transforming the whole point cloud, although a
    // little bit conservative (because it's the AABB of a AABB))

    for (size_t i = 0; i < vertexes_bbox.size(); i++)
    {
      vertexes_bbox[i] = w_T_b * vertexes_bbox[i];
    }
    // https://stackoverflow.com/questions/9070752/getting-the-bounding-box-of-a-vector-of-points
    auto xExtremes =
        std::minmax_element(vertexes_bbox.begin(), vertexes_bbox.end(),
                            [](const Eigen::Vector4d& lhs, const Eigen::Vector4d& rhs) { return lhs.x() < rhs.x(); });

    auto yExtremes =
        std::minmax_element(vertexes_bbox.begin(), vertexes_bbox.end(),
                            [](const Eigen::Vector4d& lhs, const Eigen::Vector4d& rhs) { return lhs.y() < rhs.y(); });

    auto zExtremes =
        std::minmax_element(vertexes_bbox.begin(), vertexes_bbox.end(),
                            [](const Eigen::Vector4d& lhs, const Eigen::Vector4d& rhs) { return lhs.z() < rhs.z(); });

    max_x = xExtremes.second->x();
    max_y = yExtremes.second->y();
    max_z = zExtremes.second->z();

    min_x = xExtremes.first->x();
    min_y = yExtremes.first->y();
    min_z = zExtremes.first->z();

    std::cout << std::endl;

    std::cout << "min_x= " << min_x << std::endl;
    std::cout << "min_y= " << min_y << std::endl;
    std::cout << "min_z= " << min_z << std::endl;
    std::cout << "max_x= " << max_x << std::endl;
    std::cout << "max_y= " << max_y << std::endl;
    std::cout << "max_z= " << max_z << std::endl;

    // min_x = std::numeric_limits<double>::max();
    // max_x = -std::numeric_limits<double>::max();
    // min_y = std::numeric_limits<double>::max();
    // max_y = -std::numeric_limits<double>::max();
    // min_z = std::numeric_limits<double>::max();
    // max_z = -std::numeric_limits<double>::max();

    // for (auto& vertex : vertexes_bbox)
    // {
    //   std::cout << "vertex before= " << vertex.transpose() << std::endl;

    //   if (vertex.x() <= min_x)
    //     min_x = vertex.x();
    //   if (vertex.y() <= min_y)
    //     min_y = vertex.y();
    //   if (vertex.z() <= min_z)
    //     min_z = vertex.z();
    //   if (vertex.x() >= max_x)
    //     max_x = vertex.x();
    //   if (vertex.y() >= max_y)
    //     max_y = vertex.y();
    //   if (vertex.z() >= max_z)
    //     max_z = vertex.z();
    // }

    cluster tmp;
    tmp.bbox = Eigen::Vector3d(max_x - min_x, max_y - min_y, max_z - min_z);

    assert(tmp.bbox.x() >= 0 && "Must hold: tmp.bbox.x() >= 0");
    assert(tmp.bbox.y() >= 0 && "Must hold: tmp.bbox.y() >= 0");
    assert(tmp.bbox.z() >= 0 && "Must hold: tmp.bbox.z() >= 0");

    std::cout << bold << magenta << "bbox= " << tmp.bbox << reset << std::endl;

    tmp.centroid = Eigen::Vector3d((max_x + min_x) / 2.0, (max_y + min_y) / 2.0,
                                   (max_z + min_z) / 2.0);  // This is the centroid of the bbox, not the
                                                            // centroid of the point cloud

    std::cout << red << "tmp.centroid= " << tmp.centroid.transpose() << reset << std::endl;

    tmp.time = time_pcloud;

    // std::cout << red << tmp.centroid.transpose() << reset << std::endl;
    clusters.push_back(tmp);
  }

  // rows = clusters
  // colums = tracks

  std::vector<unsigned int> indexes_costs_too_big;

  // Compute costs for each of the clusters detected
  // std::cout << "Computing costs for each of the clusters detected" << std::endl;
  std::cout << "Num of clusters detected= " << clusters.size() << std::endl;
  std::cout << "Num of current tracks   = " << all_tracks_.size() << std::endl;
  for (unsigned int i = 0; i < clusters.size(); i++)
  {
    double min_cost = std::numeric_limits<double>::max();
    for (auto& track_j : all_tracks_)
    {
      min_cost = std::min(min_cost, getCostRowColum(clusters[i], track_j, time_pcloud));
    }

    std::cout << "min_cost= " << min_cost << std::endl;

    if (min_cost > meters_to_create_new_track_)
    {
      indexes_costs_too_big.push_back(i);
    }
  }

  std::cout << "Creating= " << indexes_costs_too_big.size() << " new clusters because cost row is too big" << std::endl;
  // If the minimum of each of the rows is too large --> create new track
  for (auto i : indexes_costs_too_big)
  {
    // clusters[i].print();
    addNewTrack(clusters[i]);
  }

  //////////////////////////
  // Run Hungarian Algorithm
  //////////////////////////

  // Create the cost matrix
  std::vector<std::vector<double>> costMatrix;

  for (auto cluster_i : clusters)  // for each of the rows
  {
    std::vector<double> costs_cluster_i;
    for (auto& track_j : all_tracks_)  // for each of the columns
    {
      costs_cluster_i.push_back(getCostRowColum(cluster_i, track_j, time_pcloud));
    }

    costMatrix.push_back(costs_cluster_i);  // Add row to the matrix
  }

  // Run the Hungarian Algorithm;
  HungarianAlgorithm HungAlgo;
  std::vector<int> track_assigned_to_cluster;
  double cost = HungAlgo.Solve(costMatrix, track_assigned_to_cluster);

  //////////////////////////
  //////////////////////////

  ///////////////////////////

  // Increase by one the frames skipped on all the tracks
  std::for_each(all_tracks_.begin(), all_tracks_.end(), [](track& x) { x.num_frames_skipped++; });

  for (unsigned int i = 0; i < costMatrix.size(); i++)  // for each of the rows
  {
    std::cout << i << "," << track_assigned_to_cluster[i] << "\t";

    all_tracks_[track_assigned_to_cluster[i]].num_frames_skipped--;

    // If a cluster has been unassigned (can happen if rows>columns), then create a new track for it
    if (track_assigned_to_cluster[i] == -1)
    {
      std::cout << "cluster " << i << " unassigned, creating new track for it" << std::endl;
      std::cout << clusters[i].centroid.transpose() << std::endl;
      std::cout << "vamos a ello" << std::endl;
      addNewTrack(clusters[i]);
    }
    else
    {  // add an element to the history of the track
      all_tracks_[track_assigned_to_cluster[i]].addToHistory(clusters[i]);
    }
  }
  std::cout << "\n";

  //////////////////////////
  //////////////////////////

  // int max_frames_skipped = 10;  // TODO (as a param)

  int tracks_removed = 0;
  // Erase the tracks that haven't been detected in many frames
  all_tracks_.erase(std::remove_if(all_tracks_.begin(), all_tracks_.end(),
                                   [this, tracks_removed](const track& x) mutable {
                                     tracks_removed++;
                                     return x.num_frames_skipped > max_frames_skipped_;
                                   }),
                    all_tracks_.end());

  std::cout << "Removed " << tracks_removed << " tracks because too many frames skipped" << std::endl;

  printAllTracks();

  ////////////////////////////////////
  // Now fit a spline to past history
  ////////////////////////////////////

  for (auto& track_j : all_tracks_)
  {
    generatePredictedPwpForTrack(track_j);
  }

  ////////////////////////////////////
  // publish the stuff
  ////////////////////////////////////

  int samples = 20;

  int novale = 0;
  for (auto& track_j : all_tracks_)
  {
    std::cout << "--" << std::endl;

    track_j.printHistory();
    track_j.printPrediction(3.0, 5);

    // track_j.pwp.print();

    std::string ns = "predicted_traj_" + std::to_string(novale);
    pub_marker_predicted_traj_.publish(
        pwp2ColoredMarkerArray(track_j.pwp, time_pcloud, time_pcloud + 1.0, samples, ns, track_j.color));

    /////////////////// construct a DynTraj msg. //TODO: use the pwp instead (this will require modifications in the
    /// mader code, for when it's not an agent)

    mader_msgs::DynTraj dynTraj_msg;
    dynTraj_msg.header.frame_id = "world";
    dynTraj_msg.header.stamp = ros::Time::now();

    dynTraj_msg.function = pieceWisePol2String(track_j.pwp);

    std::vector<double> tmp = eigen2std(track_j.getLatestBbox());

    dynTraj_msg.bbox = std::vector<float>(tmp.begin(), tmp.end());  // TODO: Here I'm using the latest Bbox. Should I
                                                                    // use the biggest one of the whole history?
    dynTraj_msg.pos = eigen2rosvector(track_j.pwp.eval(ros::Time::now().toSec()));
    dynTraj_msg.id = track_j.id_int;
    dynTraj_msg.is_agent = false;

    pub_traj_.publish(dynTraj_msg);
    //////////////////

    novale++;
  }

  std::cout << "Going to get bboxes" << std::endl;

  pub_marker_bbox_obstacles_.publish(getBBoxesAsMarkerArray());

  std::cout << "End of cloud_cb" << std::endl;
}

void TrackerPredictor::printAllTracks()
{
  std::cout << green << "All tracks: " << reset;
  for (int i = 0; i < all_tracks_.size(); i++)
  {
    std::cout << "Track " << i << " (" << all_tracks_[i].num_frames_skipped << "), ";
  }
  std::cout << std::endl;
}

void TrackerPredictor::generatePredictedPwpForTrack(track& track_j)
{
  // Conversion DM <--> Eigen:  https://github.com/casadi/casadi/issues/2563

  std::map<std::string, casadi::DM> map_arguments;

  // double t0 = track_j.getOldestTimeSW();
  // double tf = track_j.getLatestTimeSW();

  double t0_r = track_j.getRelativeOldestTimeSW();
  double tf_r = track_j.getRelativeLatestTimeSW();
  double total_time = tf_r - t0_r;
  double time_per_segment = total_time / num_seg_prediction_;

  casadi::DM A, b;

  for (int i = 0; i < track_j.getSizeSW(); i++)
  {
    // std::cout << "going to get centroid" << std::endl;
    map_arguments["pos"] = eigen2std(track_j.getCentroidHistory(i));

    double time = track_j.getRelativeTimeHistory(i);

    // std::cout << "time= " << time << std::endl;

    int j = (time == tf_r) ? (num_seg_prediction_ - 1) :
                             floor((time - t0_r) / time_per_segment);  // interval of the clampled uniform bspline

    assert((j >= 0) && "(j >= 0 must hold");
    assert((j <= (num_seg_prediction_ - 1)) && "(j <= (num_seg_prediction_ - 1) must hold");

    // std::cout << blue << "time_per_segment= " << time_per_segment << reset << std::endl;

    double u = (time - (t0_r + j * time_per_segment)) / time_per_segment;

    map_arguments["u"] = u;

    assert((u >= 0) && "u>=0 must hold");
    assert((u <= 1) && "u<=1 must hold");

    // std::cout << "time-t0_r=" << time - t0_r << std::endl;
    // std::cout << green << "j=" << j << ", u= " << u << reset << std::endl;

    std::map<std::string, casadi::DM> result = cfs_kkt_Ab_[j](map_arguments);

    A = (i == 0) ? result["A"] : (A + result["A"]);
    b = (i == 0) ? result["b"] : (b + result["b"]);
  }

  std::cout << "Going to solve the kkt equations" << std::endl;

  // std::cout << std::endl;

  // std::cout << "Determinant of A= " << casadi::Determinant(A) << std::endl;

  std::cout << "A= \n" << A << std::endl;
  std::cout << "b= \n" << b << std::endl;

  // A = casadi::full(A);

  // Only for debugging
  // auto vector_x = static_cast<std::vector<double>>(A);
  // Eigen::MatrixXd A_eigen(vector_x.data());

  // Only for debugging
  // auto vector_x = static_cast<std::vector<float>>(A);
  // Eigen::Matrix<float, 6, 2> A_eigen(vector_x.data());

  /////

  casadi::DM invA_b = solve(A, b);  // Equivalent to Matlab A\b, see
                                    // https://web.casadi.org/docs/#id2-sub:~:text=Linear%20system%20solve
                                    // TODO: use Schur complement to solve only for the last segment of the spline?
  std::cout << "invA_b= " << invA_b << std::endl;

  std::map<std::string, casadi::DM> map_arguments2;
  map_arguments2["t0"] = t0_r;
  map_arguments2["tf"] = tf_r;
  map_arguments2["invA_b"] = invA_b;

  std::map<std::string, casadi::DM> result = cf_coeff_predicted_(map_arguments2);

  casadi::DM coeffs = result["coeff_predicted"];
  std::cout << "Coeffs: " << std::endl;
  std::cout << coeffs << std::endl;

  //////////////////////////

  Eigen::Matrix<double, 4, 1> coeff_old_x =
      Eigen::Vector4d(0.0, double(coeffs(0, 0)), double(coeffs(0, 1)), double(coeffs(0, 2)));
  Eigen::Matrix<double, 4, 1> coeff_old_y =
      Eigen::Vector4d(0.0, double(coeffs(1, 0)), double(coeffs(1, 1)), double(coeffs(1, 2)));
  Eigen::Matrix<double, 4, 1> coeff_old_z =
      Eigen::Vector4d(0.0, double(coeffs(2, 0)), double(coeffs(2, 1)), double(coeffs(2, 2)));

  double time_pcloud = track_j.getLatestTimeSW() - track_j.getOldestTimeSW();
  Eigen::Vector4d T =
      Eigen::Vector4d(pow(time_pcloud, 2), pow(time_pcloud, 2), pow(time_pcloud, 1), pow(time_pcloud, 0));
  std::cout << magenta << "predicted before= " << coeff_old_x.transpose() * T <<  //////
      ", " << coeff_old_y.transpose() * T <<                                      /////
      ", " << coeff_old_z.transpose() * T << reset << std::endl;

  // Fill pwp;
  double prediction_seconds = 1e6;  // infty TODO

  Eigen::Matrix<double, 4, 1> coeff_new_x, coeff_new_y, coeff_new_z;

  rescaleCoeffPol(coeff_old_x, coeff_new_x, tf_r, tf_r + prediction_seconds);
  rescaleCoeffPol(coeff_old_y, coeff_new_y, tf_r, tf_r + prediction_seconds);
  rescaleCoeffPol(coeff_old_z, coeff_new_z, tf_r, tf_r + prediction_seconds);

  mt::PieceWisePol pwp;  // will have only one interval
  pwp.times.push_back(track_j.getLatestTimeSW());
  pwp.times.push_back(track_j.getLatestTimeSW() + prediction_seconds);

  pwp.coeff_x.push_back(coeff_new_x);
  pwp.coeff_y.push_back(coeff_new_y);
  pwp.coeff_z.push_back(coeff_new_z);

  track_j.pwp = pwp;

  std::cout << magenta << "predicted after= " << track_j.pwp.eval(time_pcloud + track_j.getOldestTimeSW()).transpose()
            << reset << std::endl;
  std::cout << magenta << "real= " << track_j.getLatestCentroid().transpose() << reset << std::endl;
  std::cout << std::endl;
}

visualization_msgs::MarkerArray TrackerPredictor::getBBoxesAsMarkerArray()
{
  visualization_msgs::MarkerArray marker_array;

  int j = 0;
  for (auto& track_j : all_tracks_)
  {
    visualization_msgs::Marker m;
    m.type = visualization_msgs::Marker::CUBE;
    m.header.frame_id = "world";
    m.header.stamp = ros::Time::now();
    m.ns = "predictor";
    m.action = visualization_msgs::Marker::ADD;
    m.id = j;

    std_msgs::ColorRGBA color;
    color.r = track_j.color.x();
    color.g = track_j.color.y();
    color.b = track_j.color.z();
    color.a = 0.6;

    m.color = color;  // color(BLUE_TRANS_TRANS);

    Eigen::Vector3d centroid = track_j.getLatestCentroid();

    m.pose.position.x = centroid.x();
    m.pose.position.y = centroid.y();
    m.pose.position.z = centroid.z();

    Eigen::Vector3d bbox = track_j.getLatestBbox();

    m.scale.x = bbox.x();
    m.scale.y = bbox.y();
    m.scale.z = bbox.z();

    m.pose.orientation.w = 1.0;

    marker_array.markers.push_back(m);

    j = j + 1;

    // add its text
    m.id = j;
    m.scale.x = 0.1;
    m.scale.y = 0.1;
    m.scale.z = 0.1;

    m.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    m.text = track_j.id_string;
    m.color.a = 1.0;

    m.pose.position.z = m.pose.position.z + bbox.z();

    marker_array.markers.push_back(m);

    j = j + 1;
  }

  return marker_array;
}