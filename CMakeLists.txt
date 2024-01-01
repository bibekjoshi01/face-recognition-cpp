cmake_minimum_required(VERSION 3.10)

project(ImageRecognition)

# Find OpenCV
set(OPENCV_DIR C:/opencv/build)
find_package(OPENCV REQUIRED)

# Add ImGui and its backends
add_subdirectory(imgui)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(glfw)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} imgui glfw ${GLFW_LIBRARIES} OpenGL32.lib)
