cmake_minimum_required(VERSION 3.1.0)

project(simpleYoloObjectDetection)

list(APPEND CMAKE_PREFIX_PATH "D:\\opencv\\buildWhatEver\\install")

find_package(OpenCV REQUIRED)

add_executable(simpleYoloObjectDetection
	main.cpp
)

target_link_libraries(simpleYoloObjectDetection ${OpenCV_LIBS})

set(YOLO_RES_PATH "${PROJECT_BINARY_DIR}/")
if (WIN32)
	# where ever the flying fuck this stupid build config extension in the path comes from...
	set(YOLO_RES_PATH "${PROJECT_BINARY_DIR}/Debug/")
endif (WIN32)

configure_file("yolo3/coco.names" ${YOLO_RES_PATH} COPYONLY)
configure_file("yolo3/yolov3.cfg" ${YOLO_RES_PATH} COPYONLY)
configure_file("yolo3/yolov3.weights" ${YOLO_RES_PATH} COPYONLY)