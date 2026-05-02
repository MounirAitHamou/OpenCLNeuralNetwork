#include "Utils/EventProfiler.hpp"

namespace Utils
{
    EventProfile EventProfiler::profileEvent(const cl::Event &p_event, const std::string &p_name)
    {
        EventProfile profile;
        profile.m_name = p_name;
        p_event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &profile.m_queued);
        p_event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &profile.m_submit);
        p_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &profile.m_start);
        p_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &profile.m_end);
        return profile;
    }

    void EventProfiler::printTimeline(const std::vector<EventProfile> &p_events)
    {
        std::vector<EventProfile> sorted = p_events;
        std::ranges::sort(sorted,
                          [](const EventProfile &eventA, const EventProfile &eventB)
                          {
                              return eventA.m_start < eventB.m_start;
                          });

        std::cout << "=== GPU Event Timeline (ns → ms) ===\n";
        for (const auto &event : sorted)
        {
            double startMs = event.m_start * 1e-6;
            double endMs = event.m_end * 1e-6;
            double durMs = (event.m_end - event.m_start) * 1e-6;

            std::cout << event.m_name
                      << " | start: " << startMs << " ms"
                      << " | end: " << endMs << " ms"
                      << " | dur: " << durMs << " ms\n";
        }
        std::cout << "===================================\n";
    }
}