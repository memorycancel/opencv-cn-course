https://www.bilibili.com/video/BV14s4y1g7Zj



folder: 

```
add.cpp
div.cpp
head.h
main.cpp
mult.cpp
sub.cpp
```

1 g++:

```bash
g++ *.cpp -o app
```

2 cmake

```bash
touch CMakeLists.txt
```

input `cmake_minimum_required() ` `project()`  `add_executable()`

```shell
cmake_minimum_required(VERSION 3.15)
project(test)
add_executable(app add.cpp div.cpp mult.cpp main.cpp sub.cpp)
```

make folder:

```bash
mkdir build
cd build
cmake ..
```

run executable file

```bash
./app
```

3 set `set()`

```shell
set(SRC add.c;div.c;main.c;mult.c;sub.c)
add_executable(app ${SRC})
```

 C++ version

```shell
# -std=c++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
```

path

```shell
set(HOME /home/memorycancel/Desktop)
set(EXECUTABLE_OUTPUT_PATH ${HOME}/bin)
```

search file

```shell
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
```

4 指定头文件`include_directories()`

 ```shell
 mkdir include
 mkdir src
 mv head.h
 mv *.cpp src
 tree -L 1
 ```

```text
.
|--CMakeLists.txt
|--build
|--include
|--src
```

CMakeLists.txt 改一行

```shell
# file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
```

CMakeLists.txt 加上头文件

```
include_directories(${PROJECT_SOURCE_DIR}/include)
```

5 制作库 `add_library()`

```shell
cp -r v1 v2
cd v2
mv src/main.cpp .
```

lib+库名+后缀：

+ 动态库后缀：linux 是 .so ；windows 是 dll

+ 静态库后缀：linux 是 .a ；windows 是 lib



5.1 制作动态库

CMakeLists.txt 先注释掉可执行命令,加上`add_library(calc SHARED ${SRC})`

```shell
# add_executable(app #SRC)
add_library(calc SHARED ${SRC})
```

生成 `libcalc.so` 具备可执行权限



5.2 制作静态库 SHARE ->STATIC `add_library(calc STATIC ${SRC})`

生成 `libcalc.a` 不具备可执行权限



5.3 如何使用

需要发布给使用者，需要：

+ 库文件 .a .so
+ 头文件 .h

5.4 制定库生成路径 `set(EXECUTABLE_OUTPUT_PATH)` `LIBRARY_OUTPUT_PATH`

```shell
set(LIBRARY_OUTPUT_PATH /home/memorycancel/Desktop)
```



6 测试调用静态库文件 `link_libraries()` `link_directories()`

```shell
cp -r v2 v3
# src 没用了 因为已经编译到了库文件
rm -rf src
mkdir lib_share
mkdir lib_static
cp ~/livcalc.a lib_static
cp ~/livcalc.so lib_share
```

CMakeLists.txt 

```shell
cmake_minimum_required(VERSION 3.15)
project(test)
# src 不需要了,只需要main.cpp
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
#add_library(calc SHARED ${SRC})
# 包含头文件
include_directories(${PROJECT_SOURCE_DIR}/include)
# 引入静态库文件(放在add_executable前面)
link_libraries(calc)
# 给出库文件路径
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/static)
add_executable(app ${SRC})
```

PS：静态库会打包到可执行程序里面，可执行文件大小会包含静态库。而动态库不会。



7 测试调用动态库文件 `target_link_libraries()`

 ```she
 cp -r v3 v4
 cd v4
 ```

CMakeLists.txt 

```shell
cmake_minimum_required(VERSION 3.15)
project(test)
# src 不需要了,只需要main.cpp
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
#add_library(calc SHARED ${SRC})
# 包含头文件
#include_directories(${PROJECT_SOURCE_DIR}/include)
# 给出库文件路径
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/share)
add_executable(app ${SRC})

# 引入动态库文件(放在add_executable后面)
target_link_libraries(app calc)
```





8 实战fmt库

```shell
git clone https://github.com/fmtlib/fmt
cd fmt
cmake
make
# 文件夹会生成 libfmt.so libfmt.a
touch test123.cpp
vi test123.cpp
```

`test123.cpp`:

```cpp
#include <vector>
#include <fmt/ranges.h>

int main() {
  std::vector<int> v = {1, 2, 3};
  fmt::print("{}\n", v);
}
```

run

```shell
g++ test123.cpp libfmt.a -o test123
./test123
```

[1, 2, 3]





