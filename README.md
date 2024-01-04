# Image Recognition Project in C++

## Project Overview
- This C++ project leverages OpenCV, GLFW, and IMGUI to implement various image recognition features. The project includes the following functionalities:   

    1. Face Detection
    2. Feature (Eyes, Mouth) Detection
    3. Image Capture
    4. Face Recognition
    5. Facial Emotion Detection (Additional)

## Prerequisites
Before running the project, ensure you have the following dependencies installed:

1. **C++ Compiler: MinGW**
    - Add the MinGW bin directory to the system's environment variables (e.g., C:\MinGW\bin).
    - MINGW: https://sourceforge.net/projects/mingw/

2. **Install or Extract OpenCV to the C:\ directory.**
    - Add the OpenCV bin directory to the system's environment variables (e.g., C:\opencv\build\x64\vc16\bin).
    - CMake: https://cmake.org/download/

3. **Install CMake to generate build files.**
    - GLFW: https://www.glfw.org/
    - GLFW is used for window creation.

4. **IMGUI GitHub:** https://github.com/ocornut/imgui

## Setting Up the Project

1. Configure C++ Compiler:
    - Install MinGW and add its bin directory to the system's PATH.

2. Configure OpenCV:
    - Install or extract OpenCV to the C:\ directory.
    - Add the OpenCV bin directory to the system's PATH.

3. Download CMake and CMakeLists Extension for VSCode:
    - Install CMake and the CMake extension in Visual Studio Code for easy project configuration.

4. Download Other Dependencise CURL and JSON
    - To install libcurl and nlohmann/json (json.hpp), you have a few options depending on your development environment and how you prefer to manage dependencies.

    On Ubuntu, you can install libcurl with apt:
    - sudo apt-get install libcurl4-openssl-dev
    On Windows, you can use vcpkg, a package manager for C++ libraries:
    - vcpkg install curl

    On Ubuntu, you can install it with apt:
    - sudo apt-get install nlohmann-json3-dev

    On Windows, with vcpkg:
    - vcpkg install nlohmann-json


## Installing vcpkg on Windows:

1. Clone vcpkg: You need to clone vcpkg from its GitHub repository
    - Open a command prompt and run:
        git clone https://github.com/Microsoft/vcpkg.git
        cd vcpkg

2. Bootstrap vcpkg: Run the bootstrap script:
    - .\bootstrap-vcpkg.bat

3. Integrate vcpkg with Visual Studio:
    - After bootstrapping, you can integrate vcpkg with Visual Studio to make the installed libraries available globally:
    - .\vcpkg integrate install

4. Add vcpkg to PATH (Optional):
    - For convenience, you can add the vcpkg executable to your system's PATH environment variable. 

5. Build the project: Then you can build the project with CMake as usual.
    -  you are using VS Code with CMake Tools extension, you might need to configure the settings.json for your workspace to include the toolchain file:

    {
        "cmake.configureSettings": {
            "CMAKE_TOOLCHAIN_FILE": "C:/Users/Bibek Joshi/Desktop/vcpkg/scripts/buildsystems/vcpkg.cmake"
        }
    }

## Building and Running the Project
    - Build the Project: Use CMake to generate build files.

# Explore Image Recognition:

The application window will open, showcasing the various image recognition features.
Follow the on-screen instructions to capture images, detect faces, and explore additional functionalities.

Happy coding! 🚀