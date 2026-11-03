FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# System + ROS Noetic + MoveIt + ros_control + camera driver dependencies
# ---------------------------------------------------------------------------
# osrf/ros:noetic-desktop-full already carries rviz, Gazebo 11 + gazebo_ros,
# cv_bridge, tf2, xacro, robot_state_publisher and its apt source for the
# (end-of-life) Noetic packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # General development tools
    git \
    curl \
    wget \
    vim \
    nano \
    bash-completion \
    build-essential \
    cmake \
    pkg-config \
    gnupg \
    ca-certificates \
    python3-pip \
    python3-rosdep \
    python3-numpy \
    python3-scipy \
    python3-opencv \
    python3-yaml \
    # MoveIt (move_group for /compute_ik) and ros_control
    ros-noetic-moveit \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    # libfranka 0.9.2 build deps
    libpoco-dev \
    libeigen3-dev \
    # franka_ros build deps not in desktop-full (franka_gazebo: boost_sml)
    ros-noetic-boost-sml \
    ros-noetic-combined-robot-hw \
    # OrbbecSDK_ROS1 (Femto Mega / Bolt)
    ros-noetic-backward-ros \
    ros-noetic-image-geometry \
    libgflags-dev \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    libdw-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    && rm -rf /var/lib/apt/lists/*

# colcon from PyPI: builds the catkin packages of the workspace and keeps the
# `colcon build` / install/setup.bash workflow (colcon_defaults.yaml).
RUN pip3 install --no-cache-dir colcon-common-extensions==0.3.0

# ---------------------------------------------------------------------------
# Azure Kinect Sensor SDK
# ---------------------------------------------------------------------------
# Microsoft publishes the Linux SDK packages in its Ubuntu 18.04 repository;
# the 1.4.2 packages install on this Ubuntu 20.04 image and include the
# proprietary depth engine required at runtime. k4a-tools is not installed:
# the ROS driver only needs the runtime and development packages below.
RUN install -d -m 0755 /usr/share/keyrings \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
       | gpg --dearmor --yes -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/18.04/prod bionic main" \
       > /etc/apt/sources.list.d/microsoft-prod.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
       libk4a1.4 \
       libk4a1.4-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# libfranka 0.9.2
# ---------------------------------------------------------------------------
# The last libfranka release for the Panda: 0.9.x requires Panda system
# version >= 4.2.1 (this robot runs 4.2.2); 0.10.0 and later only talk to the
# FR3. Installed to /usr/local; franka_ros below is built against it.
ARG LIBFRANKA_VERSION=0.9.2
RUN git clone --recursive --depth 1 -b ${LIBFRANKA_VERSION} \
       https://github.com/frankaemika/libfranka.git /tmp/libfranka \
    && cmake -S /tmp/libfranka -B /tmp/libfranka/build \
       -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    && cmake --build /tmp/libfranka/build -j"$(nproc)" \
    && cmake --install /tmp/libfranka/build \
    && ldconfig \
    && rm -rf /tmp/libfranka

# ---------------------------------------------------------------------------
# Underlay: franka_ros, panda_moveit_config and the camera drivers
# ---------------------------------------------------------------------------
# Built into the image rather than kept as checkouts in src/: none of them
# needs local changes for the Panda (the 4K camera settings live in
# panda_calibration/launch). The workspace overlay is sourced on top.
#   franka_ros 0.10.2           franka_control, franka_gripper, franka_gazebo, ...
#   panda_moveit_config 0.8.1   move_group for /compute_ik (needs franka_description >= 0.10)
#   Azure_Kinect_ROS_Driver     melodic branch (the ROS 1 driver; builds on Noetic)
#   OrbbecSDK_ROS1 v2.10.6      Femto Mega / Femto Bolt
ARG FRANKA_ROS_VERSION=0.10.2
ARG PANDA_MOVEIT_CONFIG_VERSION=0.8.1
ARG AZURE_DRIVER_COMMIT=62c740603bedbdaf373dc81ec521f791bc0bad0d
ARG ORBBEC_ROS1_VERSION=v2.10.6
WORKDIR /opt/franka_underlay
RUN mkdir -p src \
    && git clone --depth 1 -b ${FRANKA_ROS_VERSION} https://github.com/frankaemika/franka_ros.git src/franka_ros \
    && git clone --depth 1 -b ${PANDA_MOVEIT_CONFIG_VERSION} https://github.com/moveit/panda_moveit_config.git src/panda_moveit_config \
    && git clone -b melodic https://github.com/microsoft/Azure_Kinect_ROS_Driver.git src/Azure_Kinect_ROS_Driver \
    && git -C src/Azure_Kinect_ROS_Driver checkout ${AZURE_DRIVER_COMMIT} \
    && git clone --depth 1 -b ${ORBBEC_ROS1_VERSION} https://github.com/orbbec/OrbbecSDK_ROS1.git src/OrbbecSDK_ROS1 \
    && rm -rf src/*/.git
# rosdep fills in whatever the list above missed. libfranka is skipped: the
# apt ros-noetic-libfranka would shadow the 0.9.2 build above.
RUN apt-get update \
    && (rosdep init 2>/dev/null || true) && rosdep update --include-eol-distros \
    && rosdep install --from-paths src --ignore-src -r -y --rosdistro noetic --skip-keys libfranka \
    && rm -rf /var/lib/apt/lists/*
RUN source /opt/ros/noetic/setup.bash \
    && colcon build --merge-install --install-base /opt/franka_underlay/install \
       --cmake-args -DCMAKE_BUILD_TYPE=Release -DFranka_DIR=/usr/local/lib/cmake/Franka \
    && rm -rf build log

# ---------------------------------------------------------------------------
# Mesa 25 for the Azure Kinect depth engine
# ---------------------------------------------------------------------------
# The depth engine computes depth with OpenGL. Ubuntu 20.04's Mesa 21.2 does not
# know this PC's Intel GPU ("Driver does not support the 0xa7ab PCI ID") and
# falls back to llvmpipe, software rendering on every CPU core, which starves
# libfranka's 1 kHz loop (communication_constraints_violation reflexes). kisak's
# turtle PPA still builds current Mesa for focal (kisak-mesa no longer does);
# with it the iris driver renders on the GPU. Kept last so that changing it does
# not rebuild the underlay.
RUN curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xEB8B81E14DA65431D7504EA8F63F0F2B90935439" \
       | gpg --dearmor --yes -o /usr/share/keyrings/kisak-turtle.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kisak-turtle.gpg] https://ppa.launchpadcontent.net/kisak/turtle/ubuntu focal main" \
       > /etc/apt/sources.list.d/kisak-turtle.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade \
       libgl1-mesa-dri \
       libglx-mesa0 \
       libegl-mesa0 \
       libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Workspace Setup
# ---------------------------------------------------------------------------
WORKDIR /franka_ws

# Source ROS, the underlay and (once built) the workspace in interactive shells.
RUN echo "source /opt/franka_underlay/install/setup.bash" >> ~/.bashrc && \
    echo "if [ -f /franka_ws/install/setup.bash ]; then source /franka_ws/install/setup.bash; fi" >> ~/.bashrc

CMD ["/bin/bash"]
