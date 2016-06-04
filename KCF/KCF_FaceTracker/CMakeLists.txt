cmake_minimum_required(VERSION 2.8)
project( PV_Tracking )
find_package( OpenCV REQUIRED )

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")  

add_executable( VIDEO Video.cpp ${sourcefiles})
target_link_libraries( VIDEO ${OpenCV_LIBS})


