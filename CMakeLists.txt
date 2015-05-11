project(fft-comparison)

cmake_minimum_required(VERSION 3.2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDE_DIRS})
add_definitions(-DAUDIOFFT_FFTW3)

add_executable(fft-comparison
	src/main.cpp
	src/AudioFFT.cpp
	)

set_property(TARGET fft-comparison PROPERTY CXX_STANDARD 11)

target_link_libraries(fft-comparison
	${ARMADILLO_LIBRARIES}
	${FFTW_LIBRARIES}
	)
