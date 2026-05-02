#pragma once

#include <iostream>
#include <cstdlib>

#define CLNN_FATAL(msg)           \
    do                            \
    {                             \
        std::cerr << msg << '\n'; \
        std::abort();             \
    } while (0)
