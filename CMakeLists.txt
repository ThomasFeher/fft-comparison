project(fft-comparison)

cmake_minimum_required(VERSION 3.2)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(fft-comparison
	src/main.cpp
	)

set_property(TARGET fft-comparison PROPERTY CXX_STANDARD 11)

target_link_libraries(fft-comparison
	${ARMADILLO_LIBRARIES}
	)
