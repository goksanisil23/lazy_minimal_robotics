#include "TimeUtil.h"

namespace time_util
{

float getTimeDuration(const time_point &t2, const time_point &t1)
{
    return std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();
}

void showTimeDuration(const time_point &t2, const time_point &t1, const std::string &message)
{
    std::cout << message << std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count() << std::endl;
}

} // namespace time_util