# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr

set(SIBR_PROJECT "gaussian")
project(sibr_${SIBR_PROJECT} LANGUAGES CXX)

sibr_gitlibrary(TARGET CudaRasterizer
    GIT_REPOSITORY 	"https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
    GIT_TAG			"255c532d5fd0c3d83afec0e8e49574b65ae381bd"
)

include(FetchContent)
if (WIN32)
	FetchContent_Declare(
			onnxruntime
			URL https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-win-x64-1.16.3.zip
			# For Windows: onnxruntime-win-x64-1.16.3.zip
			# For macOS: onnxruntime-osx-x64-1.16.3.tgz
	)
else ()
	FetchContent_Declare(
			onnxruntime
			URL https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
			# For Windows:
			# For macOS: onnxruntime-osx-x64-1.16.3.tgz
	)
endif()

FetchContent_MakeAvailable(onnxruntime)

find_package(CUDAToolkit REQUIRED)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

file(GLOB SHADERS "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")
source_group("Source Files\\shaders" FILES ${SHADERS})

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")

## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_SOURCE_DIR}/lib)

if (WIN32)
	set(ONNXRUNTIME_LIB ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib)
else()
	set(ONNXRUNTIME_LIB ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so)
endif()

set(ONNXRUNTIME_INCLUDE_DIRS)
if(EXISTS "${onnxruntime_SOURCE_DIR}/include")
	list(APPEND ONNXRUNTIME_INCLUDE_DIRS "${onnxruntime_SOURCE_DIR}/include")
endif()

# Print found include directories
foreach(dir ${ONNXRUNTIME_INCLUDE_DIRS})
	message(STATUS "Found ONNX Runtime include directory: ${dir}")
endforeach()

# Verify the library exists
if(NOT EXISTS ${ONNXRUNTIME_LIB})
	message(FATAL_ERROR "ONNX Runtime library not found at: ${ONNXRUNTIME_LIB}")
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
		${Boost_INCLUDE_DIRS}
		${ONNXRUNTIME_INCLUDE_DIRS}
		.
)

if (WIN32)
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	glfw3
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CudaRasterizer
	${ONNXRUNTIME_LIB}
)
else()
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	glfw
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CudaRasterizer
	${ONNXRUNTIME_LIB}
)
endif()

add_definitions( -DSIBR_EXP_ULR_EXPORTS -DBOOST_ALL_DYN_LINK  )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/${SIBR_PROJECT}/renderer")

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
	SHADERS ${SHADERS}
	RSC_FOLDER ${SIBR_PROJECT}

    #STANDALONE  ${INSTALL_STANDALONE}   ## mean call install_runtime with bundle dependencies resolution
    COMPONENT   ${PROJECT_NAME}_install ## will create custom target to install only this project
)
