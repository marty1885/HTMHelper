project(iris)

add_executable(iris iris.cpp)
target_link_libraries(iris nupic_core)