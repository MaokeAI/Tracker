cmake_minimum_required(VERSION 2.8)
project( MS_Tracking )
find_package( OpenCV REQUIRED )
add_executable( Video Main.cpp ${sourcefiles})
target_link_libraries( Video ${OpenCV_LIBS})


