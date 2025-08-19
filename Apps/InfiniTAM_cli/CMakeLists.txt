#########################################
# CMakeLists.txt for Apps/InfiniTAM_cli #
#########################################

###########################
# Specify the target name #
###########################

SET(targetname InfiniTAM_cli)

################################
# Specify the libraries to use #
################################
#set(CMAKE_CXX_STANDARD 17)

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseBoost.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenMP.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenNI.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UsePNG.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseRealSense.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseRealSense2.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseUVC.cmake)


#############################
# Specify the project files #
#############################

SET(sources
CLIEngine.cpp
InfiniTAM_cli.cpp
)

SET(headers
CLIEngine.h
)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${sources} ${headers})

##########################################
# Specify the target and where to put it #
##########################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/SetCUDAAppTarget.cmake)

message(STATUS "CUDA Include Directory: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Library Directory: ${CUDA_LIBRARY_DIRS}")

#################################
# Specify the libraries to link #
#################################

#TARGET_LINK_LIBRARIES(${targetname} /media/yihan/Hy_SSD/Efficient_mesh_streaming/libdraco.a InputSource ITMLib MiniSlamGraphLib ORUtils FernRelocLib)
ADD_LIBRARY(CLIEngine STATIC CLIEngine.cpp)

TARGET_LINK_LIBRARIES(${targetname} ${DRACO_LIB} InputSource ITMLib MiniSlamGraphLib ORUtils FernRelocLib) 
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkBoost.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkOpenNI.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkPNG.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkRealSense.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkRealSense2.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkUVC.cmake)
