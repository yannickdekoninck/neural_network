cmake_minimum_required (VERSION 3.8)
project(neural_network VERSION 0.1.0)


# Include sub-projects.
add_subdirectory ("src")


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
