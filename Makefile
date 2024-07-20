CC = nvcc
CXXFLAGS = -std=c++11 -ccbin /home/linuxbrew/.linuxbrew/bin/g++-12

BUILD := build
SRC := src
INCLUDE := include
T = 32 ## T >= B
B = 8 ## works up to 16

.PHONY: all clean run setup
all: run
debug: $(SRC)/* main.cu
	$(CC) $(CXXFLAGS) -g -DDEBUG -DTILE_SIZE=$(T) -DBLOCK_ROWS=$(B) -I$(INCLUDE) $^ -o $(BUILD)/$@
	./$(BUILD)/$@
main: $(SRC)/* main.cu
	$(CC) $(CXXFLAGS) -DTILE_SIZE=$(T) -DBLOCK_ROWS=$(B) -I$(INCLUDE) $^ -o $@ 
run: main 
	./main $(N)
setup:
	mkdir -p src build include \
	& touch README.md main.cpp \
clean:
	rm -rf build/* \
	rm *.o main app cachegrind.*
