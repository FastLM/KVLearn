CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS ?=

LIB_SRCS = \
  src/cost_model.cpp \
  src/features.cpp \
  src/prp.cpp \
  src/cars.cpp \
  src/atc.cpp \
  src/pool.cpp \
  src/radix_tree.cpp \
  src/kvlearn.cpp

LIB_OBJS = $(LIB_SRCS:src/%.cpp=build/%.o)

.PHONY: all clean test sim

all: build/libkvlearn.a build/kvlearn_tests build/kvlearn_sim

build:
	mkdir -p build

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/libkvlearn.a: $(LIB_OBJS)
	ar rcs $@ $^

build/kvlearn_tests: tests/test_kvlearn.cpp build/libkvlearn.a | build
	$(CXX) $(CXXFLAGS) $< -o $@ -Lbuild -lkvlearn $(LDFLAGS)

build/kvlearn_sim: apps/simulate.cpp build/libkvlearn.a | build
	$(CXX) $(CXXFLAGS) $< -o $@ -Lbuild -lkvlearn $(LDFLAGS)

test: build/kvlearn_tests
	./build/kvlearn_tests

sim: build/kvlearn_sim
	./build/kvlearn_sim 2000 1.1 64

clean:
	rm -rf build
