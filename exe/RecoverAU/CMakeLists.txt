# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
include_directories(${FaceAnalyser_SOURCE_DIR}/include)
include_directories(${Utilities_SOURCE_DIR}/include)

add_executable(RecoverAU RecoverAU.cpp)
target_link_libraries(RecoverAU LandmarkDetector)
target_link_libraries(RecoverAU FaceAnalyser)
# target_link_libraries(RecoverAU GazeAnalyser) # Not needed for AU only / AUのみなら不要
target_link_libraries(RecoverAU Utilities)

install (TARGETS RecoverAU DESTINATION bin)
