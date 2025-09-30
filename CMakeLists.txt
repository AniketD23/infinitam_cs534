##########################################
# Top-level CMakeLists.txt for InfiniTAM #
##########################################

cmake_minimum_required(VERSION 3.22)
message("Current CMake version: ${CMAKE_VERSION}")

PROJECT(InfiniTAM)

#################################
# Specify the CMake module path #
#################################

SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(Threads REQUIRED)

find_package(Eigen3 REQUIRED CONFIG)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "EIGEN Location: ${EIGEN3_INCLUDE_DIR}")
find_package(OpenCV 4 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

find_package(draco_illixr REQUIRED CONFIG)
# Prefer GLVND
if (POLICY CMP0072)
  cmake_policy (SET CMP0072 NEW)
endif(POLICY CMP0072)

find_program(CLANG_EXE clang REQUIRED)
find_program(CLANG_CXX_EXE REQUIRED NAMES clang-cpp clang++)
if(NOT CLANG_EXE OR NOT CLANG_CXX_EXE)
    message(FATAL_ERROR "This project requires the clang compiler. ${CLA}")
endif()
set(CMAKE_C_COMPILER ${CLANG_EXE} CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER ${CLANG_CXX_EXE} CACHE FILEPATH "" FORCE)

#################################
# Add additional compiler flags #
#################################

#this needs to be updated
IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -std=c++17 -Wno-deprecated-declarations -Wno-unused-function -I /usr/local/cuda/include")
ENDIF()

IF(NOT MSVC_IDE)
  SET(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "-fPIC -O3 -std=c++17 ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
  #SET(CMAKE_CXX_FLAGS "-fPIC -g ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF()

#########################
# Suppress auto-linking #
#########################

IF(MSVC_IDE)
  ADD_DEFINITIONS(-DUSING_CMAKE=1)
ENDIF()

# Pose format
# ADD_DEFINITIONS(-DVCU)

# Verbose debug prints
# ADD_DEFINITIONS(-DDEBUG)

# Instrumentation
# ADD_DEFINITIONS(-DINSTRUMENTATION)

#message(STATUS "main CUDA Include Directory: ${CUDA_INCLUDE_DIRS}")
#message(STATUS "main CUDA Library Directory: ${CUDA_LIBRARY_DIRS}")


######################
# Add subdirectories #
######################

#ADD_SUBDIRECTORY(Apps)
ADD_SUBDIRECTORY(FernRelocLib)
#ADD_SUBDIRECTORY(InputSource)
ADD_SUBDIRECTORY(ITMLib)
ADD_SUBDIRECTORY(MiniSlamGraphLib)
ADD_SUBDIRECTORY(ORUtils)
set(PLUGIN_NAME plugin.ada.infinitam${ILLIXR_BUILD_SUFFIX})

add_library(${PLUGIN_NAME} SHARED plugin.cpp
            plugin.hpp
            ${ILLIXR_ROOT}/illixr/plugin.hpp
            ${ILLIXR_ROOT}/illixr/switchboard.hpp
            ${ILLIXR_ROOT}/illixr/data_format/scene_reconstruction.hpp
            ${ILLIXR_ROOT}/illixr/data_format/mesh.hpp
            ${ILLIXR_ROOT}/illixr/relative_clock.hpp
            ${ILLIXR_ROOT}/illixr/phonebook.hpp
)
find_package(spdlog REQUIRED)
#TARGET_LINK_LIBRARIES(plugin ${DRACO_LIB} CLIEngine InputSource ITMLib MiniSlamGraphLib ORUtils FernRelocLib)
#TARGET_LINK_LIBRARIES(${PLUGIN_NAME} PUBLIC draco_illixr::draco InputSource ITMLib MiniSlamGraphLib ORUtils FernRelocLib spdlog::spdlog)
TARGET_LINK_LIBRARIES(${PLUGIN_NAME} PUBLIC draco_illixr::draco ITMLib MiniSlamGraphLib ORUtils FernRelocLib spdlog::spdlog)
target_include_directories(${PLUGIN_NAME} PUBLIC ${ILLIXR_ROOT})
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseBoost.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenMP.cmake)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${Boost_LIBRARIES} ${OpenCV_LIBS} Eigen3::Eigen)

install(TARGETS ${PLUGIN_NAME} DESTINATION lib)
