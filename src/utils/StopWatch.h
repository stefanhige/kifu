#pragma once

#include <chrono>

class StopWatch
{
public:
    ~StopWatch()
    {
         auto end_time = std::chrono::high_resolution_clock::now();
         auto time = end_time - start_time;
               std::cout << " took " <<
                 time/std::chrono::milliseconds(1) << "ms to run.\n" << std::endl;
    }

private:
 std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
};
