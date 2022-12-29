#pragma once

#include <array>
#include <random>

namespace
{
constexpr uint32_t N_DISTINCT_COLORS = 100;
}

struct ColorPalette
{
    ColorPalette()
    {
        for (auto &color : colorPalette_)
        {
            // Generate uniform random number between [0,1]
            double randR{static_cast<double>(rand()) / static_cast<double>(RAND_MAX)};
            double randG{static_cast<double>(rand()) / static_cast<double>(RAND_MAX)};
            double randB{static_cast<double>(rand()) / static_cast<double>(RAND_MAX)};

            color[0] = randR;
            color[1] = randG;
            color[2] = randB;
        }
    }

    const std::array<double, 3> &GetColorForId(uint32_t id) const
    {
        return colorPalette_.at(id);
    }

    std::array<std::array<double, 3>, N_DISTINCT_COLORS> colorPalette_;
};