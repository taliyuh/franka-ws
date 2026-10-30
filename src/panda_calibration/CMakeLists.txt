cmake_minimum_required(VERSION 3.0.2)
project(panda_calibration)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/run_calibration_capture
  scripts/run_taught_capture
  scripts/capture_stereo_rgbd
  scripts/solve_handeye
  scripts/solve_joint_handeye
  scripts/probe_board
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
