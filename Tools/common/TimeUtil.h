#pragma once

#include <chrono>
#include <iostream>

namespace time_util
{

using time_point                 = decltype(std::chrono::high_resolution_clock::now());
inline constexpr auto &chronoNow = std::chrono::high_resolution_clock::now;

void  showTimeDuration(const time_point &t2, const time_point &t1, const std::string &message);
float getTimeDuration(const time_point &t2, const time_point &t1);

} // namespace time_util