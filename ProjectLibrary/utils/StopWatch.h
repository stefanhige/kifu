#pragma once

#include <chrono>

class StopWatch
{
public:
    StopWatch(){}
    StopWatch(std::string info)
        :m_info(info){}
    ~StopWatch()
    {
         auto end_time = std::chrono::high_resolution_clock::now();
         auto time = end_time - start_time;
               std::cout << m_info << " took " <<
                 time/std::chrono::milliseconds(1) << "ms to run." << std::endl;
    }

private:
 std::string m_info = "";
 std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
};
