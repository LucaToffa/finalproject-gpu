CC = g++
CXXFLAGS = -std=c++11 -g

BUILD := build
SRC := src
INCLUDE := include

all: main

main: $(SRC)/*.cpp main.cpp
	$(CC) $(CXXFLAGS) -I$(INCLUDE) $^ -o $@

run: main 
	./main $(ARGS)
setup:
	mkdir -p src build include \
	& touch readme.md main.cpp \

clean:
	rm -rf build/* \
	rm *.o main cachegrind.*