CC = nvcc
CXXFLAGS = -std=c++11

BUILD := build
SRC := src
INCLUDE := include

all: main

debug: $(SRC)/*.cpp main.cu
	$(CC) $(CXXFLAGS) -g -DDEBUG -I$(INCLUDE) $^ -o $(BUILD)/$@
	./$(BUILD)/$@
	
main: $(SRC)/*.cpp main.cu
	$(CC) $(CXXFLAGS) -I$(INCLUDE) $^ -o $@

run: main 
	./main $(ARGS)
setup:
	mkdir -p src build include \
	& touch readme.md main.cpp \

clean:
	rm -rf build/* \
	rm *.o main cachegrind.*