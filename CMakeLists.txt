cmake_minimum_required(VERSION 2.8)

project(object_detection)

find_package(PCL 1.7 REQUIRED PATHS /usr NO_DEFAULT_PATH)
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
set(PROJ_INCLUDE_DIRS src)
include_directories(${PCL_INCLUDE_DIRS} ${PROJ_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})


set(SRC_LIST
    src/utility/pcd_viewer.cpp
    src/utility/ply2cloud.cpp
    src/utility/xyzreader.cpp
    src/app/object_detection_app.cpp)


add_executable(object_detect_demo ${SRC_LIST})
target_link_libraries(object_detect_demo ${PCL_LIBRARIES} ${VTK_LIBRARIES})
