NVCC = nvcc
CC = g++
CXXFLAGS = -std=c++17 -lcusparse
## uncomment to run on remote machine
# CXXFLAGS += -ccbin /home/linuxbrew/.linuxbrew/bin/g++-12
CC = $(NVCC)

BUILD := build
SRC := src
INCLUDE := include

#source files
CPPSRC := $(wildcard $(SRC)/*.cpp)
CUSRC := $(wildcard $(SRC)/*.cu)
#object files
OBJ := $(patsubst $(SRC)/%.cpp, $(BUILD)/%.o, $(CPPSRC))
OBJ += $(patsubst $(SRC)/%.cu, $(BUILD)/%.o, $(CUSRC))

.PHONY: all clean run setup debug

all: run
debug: $(CPPSRC) $(CUSRC) main.cu
	$(NVCC) $(CXXFLAGS) -g -DDEBUG -I$(INCLUDE) $^ -o $(BUILD)/$@
	@./$(BUILD)/$@
main: $(OBJ) $(BUILD)/main.o
	$(NVCC) $(CXXFLAGS) -I$(INCLUDE) $^ -o $@

$(BUILD)/main.o: main.cu
	@$(NVCC) $(CXXFLAGS) -I$(INCLUDE) -c $< -o $@
$(BUILD)/%.o: $(SRC)/%.cpp
	@$(CC) $(CXXFLAGS) -I$(INCLUDE) -c $< -o $@
$(BUILD)/%.o: $(SRC)/%.cu
	@$(NVCC) $(CXXFLAGS) -I$(INCLUDE) -c $< -o $@

valgrind: main
	valgrind --leak-check=full --show-leak-kinds=all -s --track-origins=yes --log-fd=9 9>valgrind.txt ./main
run: main
	./main $(N)
setup:
	mkdir -p src build include logs
clean:
	rm -rf build/* \
	rm *.o main cachegrind.*
