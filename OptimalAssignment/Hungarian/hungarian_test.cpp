#include <algorithm>
#include <iostream>
#include <vector>

#include "HungarianOptimizer.hpp"

// first: row, second: col
void ShowAssignments(const std::vector<std::pair<size_t, size_t>> &assignments)
{
    for (const auto &assignment : assignments)
    {
        std::cout << assignment.first << " -> " << assignment.second << std::endl;
    }
}

int main()
{
    // clang-format off
    // Row-major order
    std::vector<double> costMtx = {
        82,	83,	69,	92, 
        77, 37,	49,	92,
        11, 69,  5, 86, 
         8,  9, 98, 23};
    // clang-format on

    HungarianOptimizer<double>             hungSolver;
    std::vector<std::pair<size_t, size_t>> optAssignments;

    hungSolver.costs()->Reserve(1000, 1000);

    hungSolver.costs()->Resize(4, 4);

    hungSolver.costs()->AssignFromVec(costMtx);

    hungSolver.Minimize(&optAssignments);

    ShowAssignments(optAssignments);

    return 0;
}