cmake_minimum_required(VERSION 2.8)
project( MS_Tracking )
find_package( OpenCV REQUIRED )
add_executable( Do Main.cpp ${sourcefiles})
target_link_libraries( Do ${OpenCV_LIBS})


