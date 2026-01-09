#include "Utils/EventProfiler.hpp"

namespace Utils {
    EventProfile EventProfiler::profileEvent(const cl::Event& p_event, const std::string& p_name) {
        EventProfile profile;
        profile.m_name = p_name;
        p_event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &profile.m_queued);
        p_event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &profile.m_submit);
        p_event.getProfilingInfo(CL_PROFILING_COMMAND_START,  &profile.m_start);
        p_event.getProfilingInfo(CL_PROFILING_COMMAND_END,    &profile.m_end);
        return profile;
    }

    void EventProfiler::printTimeline(const std::vector<EventProfile>& p_events) {
        std::vector<EventProfile> sorted = p_events;
        std::sort(sorted.begin(), sorted.end(),
                [](const EventProfile& a, const EventProfile& b) {
                    return a.m_start < b.m_start;
                });

        std::cout << "=== GPU Event Timeline (ns → ms) ===\n";
        for (const auto& e : sorted) {
            double startMs = e.m_start * 1e-6;
            double endMs   = e.m_end   * 1e-6;
            double durMs   = (e.m_end - e.m_start) * 1e-6;

            std::cout << e.m_name
                    << " | start: " << startMs << " ms"
                    << " | end: "   << endMs   << " ms"
                    << " | dur: "   << durMs   << " ms\n";
        }
        std::cout << "===================================\n";
    }
}