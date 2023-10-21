---
title: cmake编译和相对路径
category:
  - []
comments: false
date: 2023-10-21 17:51:44
tags:
---


## 编译中相对路径的问题

我们在编写 c++ 代码的时候经常会在文件里面写如下的代码：

```c++
bool loadout = Loader.LoadFile("./resources/bunny.obj");
```

这样的代码从相对路径中读取需要的文件，但是这里的相对路径指的是编译完成后二进制文件的相对路径，如果我们的二进制文件和源文件不是在同一路径下，是没办法通过这种办法找到读取文件的。

也就是说，只有在编译后文件的结构如下二进制文件才能正常访问资源文件：

```
base_folder
|
--main.cpp
--main.exe
--resources
	      |
	      --bunny.obj
```

不过大部分情况下都不是这样，特别是我们在用 `cmake` 作为编译工具时，二进制文件往往在一个单独的文件夹下，在 `CMakeLists.txt` 中我们可以用 `CMAKE_BINARY_DIR` 来表示这个文件夹，默认情况下，这个文件夹在 `CMAKE_SOURCE_DIR/build` ，也就是项目最底层的 `CMakeLists.txt` 同一级的 `build` 文件夹，项目的结构类似于：

``` 
base_folder
|
--main.cpp
--CMakeLists.txt
--build
      |
      --main.exe
--resources
	      |
	      --bunny.obj
```

实际情况可能会更复杂，因为我们可能在一个主要的项目下通过 `add_subdirectory` 添加了其他的子项目，每一个子项目也会有自己编译后应该在的文件夹，可以用 `CMAKE_CURRENT_BINARY_DIR` 或者是 `PROJECT_BINARY_DIR`。

>在 `cmake ` 中，由于可以通过 `add_subdirectory` 添加一些文件夹作为子项目，为了方便我们确认当前项目的路径，`cmake` 提供了两种模式的变量，分别是 `CMAKE_**` 和 `PROJECT_**` 。
>
>其中 `CMAKE_**` 表示的是最底层的 `CMakeLists.txt` 的属性；`PROJECT_**` 表示的是当前 `CMakeLists.txt` 所在项目的属性。

 比如说我们有下面结构的项目：

```
base_folder
|
--CMakeLists.txt (a)
--main.cpp
--Lib1
     |
     --CMakeLists.txt (b)
     --lib1.hpp
     --lib1.cpp
     --resources
               |
               --bunny.obj
```

为了方便接下来的表述，我们称底层的 `CMakeLists.txt` 为 `(a)` ，在 `Lib1` 文件夹下的为 `(b)` 。

我们在 `(a)` 中所写的 `CMAKE_SOURCE_DIR` 和 `PROJECT_SOURCE_DIR` 指的都是同一个文件夹，`base_folder/` 。但是我们在 `(b)` 中写的两个变量，第一个指的是 `base_folder/` ，但是第二个指的是 `base_folder/Lib1/` 。

也就是 `PROJECT_**` 类型的变量是针对当前项目的，但是 `CMAKE_**` 是对于最底层的项目的。有了这样的分层，之后能够大大方便我们正确找到位于不同位置的资源文件。

比如说我们希望在 `(b)` 中表示 `./resources/bunny.obj` ，我们可以在 `(b)` 中写 `CMAKE_SOURCE_DIR/Lib1/resources/bunny.obj` 或者是 `PROJECT_SOURCE_DIR/resources/bunny.obj` ，也就是写成下面的 `cmake` 代码：

```cmake
// (b)

set(BUNNY_MODEL_PATH_1 ${PROJECT_SOURCE_DIR}/resources/bunny.obj)
set(BUNNY_MODEL_PATH_2 ${CMAKE_SOURCE_DIR}/Lib1/resources/bunny.obj)
```

## 解决问题

在知道了上面的知识之后，我们可以开始着手解决问题了，我们在 `c++` 文件中有一些相对路径，我们希望找到正确访问这些文件的方法，同时希望能够保留写相对路径的便利，如果之后我们打算分发我们的二进制文件，我们希望能够做一些很简单的修改，也能确保 `c++` 中的文件不需要改动。

主要的解决方案有两种，第一种是利用函数 `configure_file` 让 `cmake` 帮我们自动生成一些表示了路径的文件，第二种比较直接，我们直接让 `cmake` 帮我们把所有涉及到的资源文件自动复制到二进制文件的相对路径上去。

### configure_file

这种方法要求我们自己写一个模板文件，一般命名为 `config.h.in` ，我们不直接 `include` 这个文件，而是让 `cmake` 基于这个文件帮我们生成一个 `config.h` 文件，我们代码中直接 `#include "config.h"` 。当然，这个文件只有在我们运行了 `cmake config` 命令后才会生成。

这种方式的原理可以理解成，我们让 `cmake` 告诉 `c++` 文件自己一些变量的值，以此确保编译后能够正确运行。

比如我们在 `CMakeLists.txt` 中已经正确定义了模型文件的路径，并把路径保存到变量 `BUNNY_MODEL_PATH` 里面，我们可以让 `cmake` 通过一个宏变量告诉 `c++` ，`BUNNY_MODEL_PATH` 是什么。

我们首先要创建一个 `config.h.in` 模板文件，这部分可以参考 `cmake` 提供的[官方文档](https://cmake.org/cmake/help/latest/command/configure_file.html)：

```c++
// config.h.in

#define BUNNY_MODEL_PATH "@BUNNY_MODEL_PATH@"
```

这个模板文件还有很多其他的用法，这里只用到最简单的一种。随后我们在 `CMakeLists.txt (b)` 里面：

```cmake
# CMakeLists.txt (b)

# ...
set(BUNNY_MODEL_PATH ${PROJECT_SOUCE_DIR}/resources/bunny.obj)

configure_file(${PROJECT_SOURCE_DIR}/config/config.h.in config.h)
# ...
```

这里要注意，设置 `BUNNY_MODEL_PATH` 一定要在 `configure_file` 之前。

我们修改了这些文件后运行 `cmake config` ，可以看到 当前项目对应的二进制文件夹下生成了一个 `config.h` 文件，我们在 `c++` 文件中 `#include "config.h"`  ，就可以把最开始的代码替换成：

```c++
#include "config.h"

// ...
bool loadout = Loader.LoadFile(BUNNY_MODEL_PATH);
```

现在我们的项目结构为：

```
base_folder
|
--CMakeLists.txt (a)
--main.cpp
--Lib1
     |
     --CMakeLists.txt (b)
     --lib1.hpp
     --lib1.cpp
     --config
            |
            --config.h.in
     --resources
               |
               --bunny.obj
```



当然，我们现在可以举一反三，如果我希望载入的文件非常多，我们可以在 `CMakeLists.txt` 里面设置一个变量为一整个文件夹的路径，再在 `c++` 文件中把这两个字符串拼起来：

```c++
// c++ file
#include "config.h"

// ...
bool loadout = Loader.LoadFile(MODEL_PATH "/bunny.obj");
```

```cmake
# CMakeLists.txt (b)

# ...
set(MODEL_PATH ${PROJECT_SOUCE_DIR}/resources)

configure_file(${PROJECT_SOURCE_DIR}/config/config.h.in config.h)
# ...
```

```c++
// config.h.in

#define MODEL_PATH "@MODEL_PATH@"
```

还有个要提的地方，我们不需要把 `config.h` 手动复制到任何地方，我们只要在 `CMakeLists.txt` 中加上：

```cmake
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_BINARY_DIR})
```

就可以在 `cmake config` 后自动找到 `config.h` 。

### 复制资源文件到相对路径

这部分其实大部分只要复制代码就可以了，可以找到很多现成的代码：

```cmake
# CMakeLists.txt

# copy resources to relative path of binaries
file(GLOB_RECURSE RESOURCE_FILES ${PROJECT_SOURCE_DIR}/resources/**)
foreach(RESOURCE_FILE ${RESOURCE_FILES})
  file(RELATIVE_PATH RESOURCE_REL_PATH ${PROJECT_SOURCE_DIR} ${RESOURCE_FILE})
  add_custom_command(TARGET ${MAIN_NAME} POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E copy_if_different ${RESOURCE_FILE}
  $<TARGET_FILE_DIR:${PROJECT_NAME}>/${RESOURCE_REL_PATH})
endforeach()
```

上面的代码不仅能够复制资源文件到二进制文件的相对路径，还会检查这个资源文件相比于上一次编译有没有发生变化 `copy_if_different` ，如果变化了就会把更新同步过去。
