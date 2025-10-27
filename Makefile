CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -I./NVmesh/core -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcuda
TARGET = success_example
SOURCE = example/success_example.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

.PHONY: all clean
