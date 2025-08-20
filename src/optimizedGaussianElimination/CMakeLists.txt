set(sources optimizedGaussianElimination.cpp)
set(headers ../../include/JC/util.h ../../include/JC/openCLUtil.hpp)
set(resources ../optimizedKernels.ocl)

set(my_include_dirs ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

add_executable(optimizedGaussianElimination ${sources} ${headers} ${resources})


# this does not work
#set(RESOURCE_FILES
#  simpleKernels.ocl
#)

target_include_directories(optimizedGaussianElimination PRIVATE 
    ${OpenCL_INCLUDE_DIRS}
    ${my_include_dirs})

target_link_libraries(optimizedGaussianElimination ${OpenCL_LIBRARIES})

			
add_custom_command(TARGET optimizedGaussianElimination
	COMMAND POST_BUILD COMMAND  ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../optimizedKernels.ocl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/optimizedKernels.ocl
	COMMENT "Copying OpenCL files to output/runtime directory")

add_custom_command(TARGET optimizedGaussianElimination
	COMMAND POST_BUILD COMMAND  ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../optimizedKernels.ocl ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Copying OpenCL files to build directory")