cmake_minimum_required(VERSION 3.15)
project(MyProject)

set(CMAKE_CXX_STANDARD 20)
set(OpenCV_DIR C:/OpenCV-MinGW-Build)
find_package(OpenCV REQUIRED)
add_executable(MyProject main.cpp TSRA.h TSRA.cpp)
target_link_libraries(MyProject ${OpenCV_LIBS})
