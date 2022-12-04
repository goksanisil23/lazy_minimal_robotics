#include "VisNode.h"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<landmarkSim2D::VisNode>());

    rclcpp::shutdown();
    return 0;
}