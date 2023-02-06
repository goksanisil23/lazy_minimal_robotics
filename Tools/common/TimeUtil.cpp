#include "TimeUtil.h"

namespace time_util
{

double getTimeDuration(const time_point &t2, const time_point &t1)
{
    return std::chrono::duration<double, std::chrono::seconds::period>(t2 - t1).count();
}

void showTimeDuration(const time_point &t2, const time_point &t1, const std::string &message)
{
    std::cout << message << std::chrono::duration<double, std::chrono::seconds::period>(t2 - t1).count() << std::endl;
}

} // namespace time_util