#pragma once

#include <CL/opencl.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
namespace Utils {
    struct EventProfile {
        std::string m_name;
        cl_ulong m_queued;
        cl_ulong m_submit;
        cl_ulong m_start;
        cl_ulong m_end;
    };

    class EventProfiler {
    public:
        static EventProfile profileEvent(const cl::Event& p_event, const std::string& p_name);

        static void printTimeline(const std::vector<EventProfile>& p_events);
    };
}