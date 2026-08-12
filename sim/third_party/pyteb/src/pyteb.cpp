// pyteb -- a THIN pybind11 bridge to the upstream teb_local_planner
// (rst-tu-dortmund/teb_local_planner, Roesmann et al.), version 0.9.1, as
// shipped by the RoboStack conda package `ros-noetic-teb-local-planner`.
//
// This file contains NO planning algorithm.  Every line of Timed-Elastic-Band
// maths -- the g2o hyper-graph, the pose AND time-difference vertices, the
// velocity / acceleration / obstacle / time-optimality edges, the sparse
// Levenberg-Marquardt solve, the autoResize of the band -- lives inside the
// upstream teb_local_planner.dll.  Here we only:
//   * fill a teb_local_planner::TebConfig from a {name: value} map,
//   * build a teb_local_planner::ObstContainer of PointObstacle / LineObstacle,
//   * call TebOptimalPlanner::plan(...) and ::getVelocityCommand(...),
//   * copy the resulting band out for inspection.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>
#include <vector>
#include <cmath>

#include <teb_local_planner/optimal_planner.h>
#include <teb_local_planner/homotopy_class_planner.h>
#include <teb_local_planner/obstacles.h>
#include <teb_local_planner/robot_footprint_model.h>
#include <teb_local_planner/teb_config.h>
#include <teb_local_planner/pose_se2.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>

namespace py = pybind11;
namespace teb = teb_local_planner;

class TebBridge {
public:
  TebBridge(const std::map<std::string, double>& params,
            double footprint_radius,
            bool use_homotopy)
      : use_homotopy_(use_homotopy) {
    apply(params);
    if (footprint_radius > 0.0)
      robot_model_ = boost::make_shared<teb::CircularRobotFootprint>(footprint_radius);
    else
      robot_model_ = boost::make_shared<teb::PointRobotFootprint>();
    reset();
  }

  // Re-create the planner, discarding any hot-started band.
  void reset() {
    if (use_homotopy_)
      planner_ = teb::PlannerInterfacePtr(new teb::HomotopyClassPlanner(
          cfg_, &obstacles_, robot_model_, teb::TebVisualizationPtr(), &via_points_));
    else
      planner_ = teb::PlannerInterfacePtr(new teb::TebOptimalPlanner(
          cfg_, &obstacles_, robot_model_, teb::TebVisualizationPtr(), &via_points_));
  }

  void set_point_obstacles(const std::vector<std::array<double, 4>>& obs) {
    obstacles_.clear();
    for (const auto& o : obs) {
      teb::ObstaclePtr p(new teb::PointObstacle(o[0], o[1]));
      if (o[2] != 0.0 || o[3] != 0.0)
        p->setCentroidVelocity(Eigen::Vector2d(o[2], o[3]));
      obstacles_.push_back(p);
    }
    for (const auto& l : walls_) {
      obstacles_.push_back(teb::ObstaclePtr(new teb::LineObstacle(l[0], l[1], l[2], l[3])));
    }
  }

  // Static line obstacles (the sidewalk kerbs).  In the real ROS stack these
  // come out of the local costmap; here they are supplied explicitly.
  void set_walls(const std::vector<std::array<double, 4>>& walls) { walls_ = walls; }

  // initial_plan: [[x, y, theta], ...] -- the reference path, exactly what
  // TebLocalPlannerROS hands to plan() after pruning the global plan.
  bool plan(const std::vector<std::array<double, 3>>& initial_plan,
            double vx, double vy, double omega, bool free_goal_vel) {
    std::vector<geometry_msgs::PoseStamped> plan_msg;
    plan_msg.reserve(initial_plan.size());
    for (const auto& p : initial_plan) {
      geometry_msgs::PoseStamped ps;
      ps.pose.position.x = p[0];
      ps.pose.position.y = p[1];
      ps.pose.position.z = 0.0;
      const double h = 0.5 * p[2];
      ps.pose.orientation.x = 0.0;
      ps.pose.orientation.y = 0.0;
      ps.pose.orientation.z = std::sin(h);
      ps.pose.orientation.w = std::cos(h);
      plan_msg.push_back(ps);
    }
    geometry_msgs::Twist v0;
    v0.linear.x = vx;
    v0.linear.y = vy;
    v0.angular.z = omega;
    return planner_->plan(plan_msg, &v0, free_goal_vel);
  }

  std::array<double, 4> velocity_command(int look_ahead_poses) {
    double vx = 0, vy = 0, om = 0;
    bool ok = planner_->getVelocityCommand(vx, vy, om, look_ahead_poses);
    return {ok ? 1.0 : 0.0, vx, vy, om};
  }

  // Full band: [[x, y, theta, dt_to_next], ...].  dt_to_next IS the TEB time
  // difference decision variable -- present here, absent from the in-repo
  // reimplementation.
  std::vector<std::array<double, 4>> band() {
    std::vector<std::array<double, 4>> out;
    const teb::TebOptimalPlanner* op = nullptr;
    if (use_homotopy_) {
      auto* hcp = dynamic_cast<teb::HomotopyClassPlanner*>(planner_.get());
      if (hcp) op = hcp->bestTeb().get();
    } else {
      op = dynamic_cast<teb::TebOptimalPlanner*>(planner_.get());
    }
    if (!op) return out;
    const teb::TimedElasticBand& t = op->teb();
    for (int i = 0; i < t.sizePoses(); ++i) {
      double dt = (i < t.sizeTimeDiffs()) ? t.TimeDiff(i) : 0.0;
      out.push_back({t.Pose(i).x(), t.Pose(i).y(), t.Pose(i).theta(), dt});
    }
    return out;
  }

  void clear_planner() { planner_->clearPlanner(); }

  double config(const std::string& name) const {
    auto* p = const_cast<TebBridge*>(this)->slot(name);
    return p ? *p : std::nan("");
  }

  std::vector<std::string> config_names() const {
    std::vector<std::string> out;
    for (const auto& kv : table()) out.push_back(kv);
    return out;
  }

private:
  teb::TebConfig cfg_;
  teb::ObstContainer obstacles_;
  teb::ViaPointContainer via_points_;
  teb::PlannerInterfacePtr planner_;
  std::vector<std::array<double, 4>> walls_;
  bool use_homotopy_;
  teb::RobotFootprintModelPtr robot_model_;

  void apply(const std::map<std::string, double>& params) {
    for (const auto& kv : params) {
      double* d = slot(kv.first);
      int* i = islot(kv.first);
      bool* b = bslot(kv.first);
      if (d) { *d = kv.second; continue; }
      if (i) { *i = static_cast<int>(kv.second); continue; }
      if (b) { *b = (kv.second != 0.0); continue; }
      throw std::runtime_error("pyteb: unknown TebConfig parameter '" + kv.first + "'");
    }
  }

#define D(name, expr) if (n == name) return &(expr);
  double* slot(const std::string& n) {
    D("trajectory.teb_autosize", cfg_.trajectory.teb_autosize)
    D("trajectory.dt_ref", cfg_.trajectory.dt_ref)
    D("trajectory.dt_hysteresis", cfg_.trajectory.dt_hysteresis)
    D("trajectory.global_plan_viapoint_sep", cfg_.trajectory.global_plan_viapoint_sep)
    D("trajectory.max_global_plan_lookahead_dist", cfg_.trajectory.max_global_plan_lookahead_dist)
    D("trajectory.global_plan_prune_distance", cfg_.trajectory.global_plan_prune_distance)
    D("trajectory.force_reinit_new_goal_dist", cfg_.trajectory.force_reinit_new_goal_dist)
    D("trajectory.force_reinit_new_goal_angular", cfg_.trajectory.force_reinit_new_goal_angular)
    D("robot.max_vel_x", cfg_.robot.max_vel_x)
    D("robot.max_vel_x_backwards", cfg_.robot.max_vel_x_backwards)
    D("robot.max_vel_y", cfg_.robot.max_vel_y)
    D("robot.max_vel_theta", cfg_.robot.max_vel_theta)
    D("robot.acc_lim_x", cfg_.robot.acc_lim_x)
    D("robot.acc_lim_y", cfg_.robot.acc_lim_y)
    D("robot.acc_lim_theta", cfg_.robot.acc_lim_theta)
    D("robot.min_turning_radius", cfg_.robot.min_turning_radius)
    D("goal_tolerance.xy_goal_tolerance", cfg_.goal_tolerance.xy_goal_tolerance)
    D("goal_tolerance.yaw_goal_tolerance", cfg_.goal_tolerance.yaw_goal_tolerance)
    D("obstacles.min_obstacle_dist", cfg_.obstacles.min_obstacle_dist)
    D("obstacles.inflation_dist", cfg_.obstacles.inflation_dist)
    D("obstacles.dynamic_obstacle_inflation_dist", cfg_.obstacles.dynamic_obstacle_inflation_dist)
    D("obstacles.obstacle_association_force_inclusion_factor", cfg_.obstacles.obstacle_association_force_inclusion_factor)
    D("obstacles.obstacle_association_cutoff_factor", cfg_.obstacles.obstacle_association_cutoff_factor)
    D("obstacles.costmap_obstacles_behind_robot_dist", cfg_.obstacles.costmap_obstacles_behind_robot_dist)
    D("optim.penalty_epsilon", cfg_.optim.penalty_epsilon)
    D("optim.weight_max_vel_x", cfg_.optim.weight_max_vel_x)
    D("optim.weight_max_vel_y", cfg_.optim.weight_max_vel_y)
    D("optim.weight_max_vel_theta", cfg_.optim.weight_max_vel_theta)
    D("optim.weight_acc_lim_x", cfg_.optim.weight_acc_lim_x)
    D("optim.weight_acc_lim_y", cfg_.optim.weight_acc_lim_y)
    D("optim.weight_acc_lim_theta", cfg_.optim.weight_acc_lim_theta)
    D("optim.weight_kinematics_nh", cfg_.optim.weight_kinematics_nh)
    D("optim.weight_kinematics_forward_drive", cfg_.optim.weight_kinematics_forward_drive)
    D("optim.weight_kinematics_turning_radius", cfg_.optim.weight_kinematics_turning_radius)
    D("optim.weight_optimaltime", cfg_.optim.weight_optimaltime)
    D("optim.weight_shortest_path", cfg_.optim.weight_shortest_path)
    D("optim.weight_obstacle", cfg_.optim.weight_obstacle)
    D("optim.weight_inflation", cfg_.optim.weight_inflation)
    D("optim.weight_dynamic_obstacle", cfg_.optim.weight_dynamic_obstacle)
    D("optim.weight_dynamic_obstacle_inflation", cfg_.optim.weight_dynamic_obstacle_inflation)
    D("optim.weight_viapoint", cfg_.optim.weight_viapoint)
    D("optim.weight_adapt_factor", cfg_.optim.weight_adapt_factor)
    D("optim.obstacle_cost_exponent", cfg_.optim.obstacle_cost_exponent)
    D("optim.weight_prefer_rotdir", cfg_.optim.weight_prefer_rotdir)
    D("recovery.oscillation_v_eps", cfg_.recovery.oscillation_v_eps)
    D("recovery.oscillation_omega_eps", cfg_.recovery.oscillation_omega_eps)
    D("hcp.selection_cost_hysteresis", cfg_.hcp.selection_cost_hysteresis)
    D("hcp.selection_obst_cost_scale", cfg_.hcp.selection_obst_cost_scale)
    D("hcp.selection_viapoint_cost_scale", cfg_.hcp.selection_viapoint_cost_scale)
    D("hcp.selection_prefer_initial_plan", cfg_.hcp.selection_prefer_initial_plan)
    D("hcp.obstacle_heading_threshold", cfg_.hcp.obstacle_heading_threshold)
    D("hcp.roadmap_graph_area_width", cfg_.hcp.roadmap_graph_area_width)
    D("hcp.roadmap_graph_area_length_scale", cfg_.hcp.roadmap_graph_area_length_scale)
    D("hcp.h_signature_prescaler", cfg_.hcp.h_signature_prescaler)
    D("hcp.h_signature_threshold", cfg_.hcp.h_signature_threshold)
    D("hcp.switching_blocking_period", cfg_.hcp.switching_blocking_period)
    D("hcp.obstacle_keypoint_offset", cfg_.hcp.obstacle_keypoint_offset)
    return nullptr;
  }

  int* islot(const std::string& n) {
    D("trajectory.min_samples", cfg_.trajectory.min_samples)
    D("trajectory.max_samples", cfg_.trajectory.max_samples)
    D("trajectory.feasibility_check_no_poses", cfg_.trajectory.feasibility_check_no_poses)
    D("trajectory.control_look_ahead_poses", cfg_.trajectory.control_look_ahead_poses)
    D("obstacles.obstacle_poses_affected", cfg_.obstacles.obstacle_poses_affected)
    D("optim.no_inner_iterations", cfg_.optim.no_inner_iterations)
    D("optim.no_outer_iterations", cfg_.optim.no_outer_iterations)
    D("hcp.max_number_classes", cfg_.hcp.max_number_classes)
    D("hcp.roadmap_graph_no_samples", cfg_.hcp.roadmap_graph_no_samples)
    return nullptr;
  }

  bool* bslot(const std::string& n) {
    D("trajectory.global_plan_overwrite_orientation", cfg_.trajectory.global_plan_overwrite_orientation)
    D("trajectory.allow_init_with_backwards_motion", cfg_.trajectory.allow_init_with_backwards_motion)
    D("trajectory.via_points_ordered", cfg_.trajectory.via_points_ordered)
    D("trajectory.exact_arc_length", cfg_.trajectory.exact_arc_length)
    D("trajectory.publish_feedback", cfg_.trajectory.publish_feedback)
    D("robot.cmd_angle_instead_rotvel", cfg_.robot.cmd_angle_instead_rotvel)
    D("robot.is_footprint_dynamic", cfg_.robot.is_footprint_dynamic)
    D("goal_tolerance.free_goal_vel", cfg_.goal_tolerance.free_goal_vel)
    D("goal_tolerance.complete_global_plan", cfg_.goal_tolerance.complete_global_plan)
    D("hcp.selection_alternative_time_cost", cfg_.hcp.selection_alternative_time_cost)
    D("recovery.shrink_horizon_backup", cfg_.recovery.shrink_horizon_backup)
    D("recovery.oscillation_recovery", cfg_.recovery.oscillation_recovery)
    D("obstacles.include_dynamic_obstacles", cfg_.obstacles.include_dynamic_obstacles)
    D("obstacles.include_costmap_obstacles", cfg_.obstacles.include_costmap_obstacles)
    D("obstacles.legacy_obstacle_association", cfg_.obstacles.legacy_obstacle_association)
    D("optim.optimization_activate", cfg_.optim.optimization_activate)
    D("optim.optimization_verbose", cfg_.optim.optimization_verbose)
    D("hcp.enable_homotopy_class_planning", cfg_.hcp.enable_homotopy_class_planning)
    D("hcp.enable_multithreading", cfg_.hcp.enable_multithreading)
    D("hcp.simple_exploration", cfg_.hcp.simple_exploration)
    D("hcp.viapoints_all_candidates", cfg_.hcp.viapoints_all_candidates)
    D("hcp.visualize_hc_graph", cfg_.hcp.visualize_hc_graph)
    D("hcp.delete_detours_backwards", cfg_.hcp.delete_detours_backwards)
    return nullptr;
  }
#undef D

  std::vector<std::string> table() const { return {}; }
};

PYBIND11_MODULE(pyteb, m) {
  m.doc() = "Thin pybind11 bridge to upstream teb_local_planner 0.9.1 "
            "(rst-tu-dortmund, Roesmann et al.). No algorithm is implemented here.";
  m.attr("teb_version") = "0.9.1";
  py::class_<TebBridge>(m, "TebBridge")
      .def(py::init<const std::map<std::string, double>&, double, bool>(),
           py::arg("params"), py::arg("footprint_radius") = 0.25,
           py::arg("use_homotopy") = false)
      .def("reset", &TebBridge::reset)
      .def("set_walls", &TebBridge::set_walls)
      .def("set_point_obstacles", &TebBridge::set_point_obstacles)
      .def("plan", &TebBridge::plan, py::arg("initial_plan"), py::arg("vx"),
           py::arg("vy"), py::arg("omega"), py::arg("free_goal_vel") = false)
      .def("velocity_command", &TebBridge::velocity_command,
           py::arg("look_ahead_poses") = 1)
      .def("band", &TebBridge::band)
      .def("clear_planner", &TebBridge::clear_planner)
      .def("config", &TebBridge::config);
}
