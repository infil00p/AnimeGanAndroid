# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("animegangallerydemo")


set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
set(OpenCV_Android ${CMAKE_SOURCE_DIR}/../../../../../external/OpenCV-android-sdk)

file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")


find_library(PYTORCH_LIBRARY pytorch_jni
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)

find_library(FBJNI_LIBRARY fbjni
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)

# If someone would find a good AAR that included headers, that would be awesome
# either that, or something that uses prefab
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(
        lib_opencv
        PROPERTIES
        IMPORTED_LOCATION
        ${OpenCV_Android}/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so)

# Set all the include directories
# We create an include directory of headers that we want to use independent of
# the Android project, like OpenCV (We are using OpenCV 4.1 here)

include_directories(
        ${PYTORCH_INCLUDE_DIRS}
        ${OpenCV_Android}/sdk/native/jni/include
)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        animegangallerydemo

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        AnimeGanPyTorch.cpp
        AnimeGan.cpp
        native-lib.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        animegangallerydemo

        # Links the target library to the log library
        # included in the NDK.
        -ljnigraphics
        ${FBJNI_LIBRARY}
        ${PYTORCH_LIBRARY}
        lib_opencv
        ${log-lib})