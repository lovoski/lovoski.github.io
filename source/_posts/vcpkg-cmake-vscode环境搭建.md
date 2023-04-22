---
title: vcpkg+cmake+vscode环境搭建
date: 2023-04-20 16:19:07
comments: false
tags:
- environment
categories:
- [Environment]
---
因为需要用到的工具大多只有`windows`平台的，我的代码环境只有少量运行在`wsl2`上，大部分需要在`windows`编译运行。为了维护c++用到的大量第三方库，我用[vcpkg](https://vcpkg.io/)和[cmake](https://cmake.org/)搭建出c++的编译工具链。我下面直接用Q&A的方式快速表明解决方案。

## vcpkg 国内下载源很慢，镜像网站资源不全怎么处理？
- 可以参考[这里的解决方案](https://zhuanlan.zhihu.com/p/383683670)，通过再环境变量里面添加一项
```
X_VCPKG_ASSET_SOURCES
```
让这项的值为
```
x-azurl,http://106.15.181.5/
```
这样可以让大部分的资源都从镜像网站下载，但是有的镜像网站值镜像了部分常用的包，还有的仍然是从`github`上下载的。
- 一个办法是`git`加上代理，通过命令
```powershell
git config --global http.proxy "<ip>:<port>"
git config --global https.proxy "<ip>:<port>"
```
让`git`发送`http`和`https`请求的时候走代理端口，`clash`的默认端口号是7890。
- 另外一个比较痛苦的方法是手动下载需要的包，按照vcpkg下载时的一则信息
```
... https://... -> ... 
```
手动去前面的网址下载需要的包，下载完后重命名为箭头后面的名字，放到vcpkg根目录下面的download文件夹里面，下一次重新运行`vcpkg install ...`的时候会找到这个下载好的文件。

## libigl 的一些组件 [cgal, glfw, imgui] 很难完整下载下来安装

- 如果给git加上代理也没有作用，可是采用下面的办法。
- 因为这种小组件依赖了一些找不到下载地址的`submodule`，但是这些组件的源代码都在`libigl`的源码包里面，直接把`libigl`的源代码覆盖到`vcpkg/installed/x64-windows/include`里面去，下一次重新安装的时候会提示覆盖已有文件，然后能成功编译了。

## vscode 怎么结合 cmake 编译调试程序？

- `vscode`有一个微软的官方插件`CMake Tools`，集成了`cmake`的功能，确保安装了`cmake`并添加到环境变量之后可以直接用`ctrl`+`shift`+`p`调出命令盘，输入`cmake:`，接下来选择其中的`configure`，`build`就可以了。下面是`vscode`调出的操作盘。
![fig](vcpkg-cmake-vscode环境搭建/vscode_cmake_tools.png)
- 如果需要给程序添加断点，逐步debug调试，可以在当前的项目根目录的`.vscode`文件夹里面创建或者修改`settings.json`为下面的样式
```json
{
    "cmake.configureSettings": {
        "CMAKE_TOOLCHAIN_FILE": "...\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake"
    },
    "cmake.debugConfig": {
        "args": ["a", "b", "c"]
    }
}
```
上面的配置文件指定了`cmake`编译使用`vcpkg`的工具链，路径不全为本机上vcpkg的安装位置下的`scripts/buildsystems/vcpkg.cmake`文件。下面的是在`debug`的时候需要给二进制文件传递的命令行参数，上面的参数等价于在命令行里面执行
```powershell
.\some_binary.exe a b c
```

## 我该怎么组织一个自己的 cmake 项目？

- 首先要确认需要的项目的大致结构，比较常见的是这样的
- root folder
  - lib_a
  - lib_b
  - sub folder
    - exe_a
    - lib_c
  - exe_b
- 在上面的例子中，我们假设`exe_a`是一个用于测试的可执行程序，依赖于`lib_a`，`lib_a`和`lib_b`是最基本的库文件，`lib_c`是一个依赖于`lib_a`的库文件，`exe_b`是我们的主程序，主程序同时依赖于`lib_a`，`lib_b`和`lib_c`。
- 我们首先要熟悉库文件和可执行文件的`CMakeLists.txt`的写法下面是一个库文件的`CMakeLists.txt`的例子
```cmake
# 要求 cmake 的最低版本
cmake_minimum_required(VERSION 3.20)
# 这个子程序的项目名称
# 在这里会生成一个名为 Model3D.lib 的库文件
project(Model3D)
set(CMAKE_CXX_STANDARD 17)
# 设置产生二进制文件的位置
# 这里需要提一下，PROJECT_SOURCE_DIR是不会随着当前的项目变化的
# 在整个项目里面不管有多少子项目都是同一个宏
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# 把当前的相对路径下的 src 文件夹下的所有 .cpp 文件记录到变量 SRC_FILES 里面
file(GLOB SRC_FILES src/*.cpp)
# 指定当前的 include 路径为相对路径下的 include 文件夹
include_directories(include)
# 用于找到需要的第三方依赖库
find_package(CGAL CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
find_package(libigl CONFIG REQUIRED)

# 生成库文件
add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
# 为生成的库文件链接第三方库
target_link_libraries(${PROJECT_NAME} PRIVATE CGAL::CGAL Eigen3::Eigen igl::core igl::common)
```
- 下面的是一个可执行文件的`CMakeLists.txt`的例子
```cmake
cmake_minimum_required(VERSION 3.10)
project(CoveringForGeodesic)
set(CMAKE_CXX_STANDARD 17)

# 设置是否要编译下面的这些子项目
option(BUILD_MODEL3D "Build subproject model3d statically" ON)
option(BUILD_GEODESIC "Build subproject geodesic statically" ON)
option(BUILD_RIDGE_STRUCTURE "Build subproject ridge_structure statically" ON)

file(GLOB MAIN_SRC_FILES *.cpp *.hpp)
# 这里还要添加自己依赖的子项目库的头文件
include_directories(
  ${CMAKE_SOURCE_DIR}/Geodesic/include
  ${CMAKE_SOURCE_DIR}/Model3D/include
  ${CMAKE_SOURCE_DIR}/CommonAlgorithms
  ${CMAKE_SOURCE_DIR}/RidgeStructure/include
)
link_directories(${PROJECT_SOURCE_DIR}/build)
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
find_package(libigl CONFIG REQUIRED)

if (${BUILD_MODEL3D})
  add_subdirectory(Model3D)
endif()

if (${BUILD_GEODESIC})
  add_subdirectory(Geodesic)
endif()

if (${BUILD_RIDGE_STRUCTURE})
  add_subdirectory(RidgeStructure)
endif()

# 生成二进制文件
add_executable(${PROJECT_NAME} ${MAIN_SRC_FILES})
# 链接库文件
# 这里直接用子项目的 PROJECT_NAME 作为链接的标识
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen igl::core igl::common Model3D Geodesic RidgeStructure)
```
- 从上面的两个例子已经可以看出，如果我需要把自己写的本地库项目连接到项目内的可执行文件里，需要把库项目的头文件用`include_directories`设置到`include`路径上，在之后通过`target_link_libraries`用库项目的`PROJECT_NAME`直接链接到可执行文件。

## find_package 做了些什么？