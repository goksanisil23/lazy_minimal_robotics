#include "MultiRobotEnv.h"

// ----- Robot Class ----- //
MultiRobotEnv::Robot::Robot(const raylib::Vector2 &pos0,
                            const double          &heading0,
                            const double          &v0,
                            const double          &radius,
                            const size_t          &id)
    : position{pos0}, heading{heading0}, velocity{v0}, radius{radius}, id{id}
{
}
void MultiRobotEnv::Robot::Draw() const
{
    DrawCircle(position.x, position.y, radius, BLUE);
    DrawBoundingBox(bbox, GREEN);
    DrawText(std::to_string(id).c_str(), bbox.min.x - 5, bbox.min.y - 5, 15, GREEN);
}

// ----- MultiRobotEnv Class ----- //

MultiRobotEnv::MultiRobotEnv() : simWindow_(SCREENWIDTH, SCREENHEIGHT, "Hungarian Assignment")
{
    SetTargetFPS(FPS);
    areaWidth_  = static_cast<double>(SCREENWIDTH);
    areaHeight_ = static_cast<double>(SCREENHEIGHT);
}

void MultiRobotEnv::GenerateRobots(const size_t &areaWidth, const size_t &areaHeight, const size_t &numRobots)
{
    for (size_t robotIdx = 0; robotIdx < numRobots; robotIdx++)
    {
        double xCoord   = static_cast<double>(rand()) / static_cast<double>(RAND_MAX / (areaWidth));
        double yCoord   = static_cast<double>(rand()) / static_cast<double>(RAND_MAX / (areaHeight));
        double heading  = static_cast<double>(rand()) / static_cast<double>(RAND_MAX / (2 * M_PI));
        double velocity = static_cast<double>(rand()) / static_cast<double>(RAND_MAX / (V_MAX));
        robots_.emplace_back(Robot(raylib::Vector2(xCoord, yCoord), heading, velocity, ROBOT_RADIUS, robotIdx));
    }
}

void MultiRobotEnv::DrawRobots()
{
    std::for_each(robots_.begin(), robots_.end(), [](const MultiRobotEnv::Robot &robot) { robot.Draw(); });
}

void MultiRobotEnv::MoveRobots(const double &dt)
{
    std::for_each(robots_.begin(),
                  robots_.end(),
                  [&dt, this](MultiRobotEnv::Robot &robot)
                  {
                      // Handle the bouncing-off the walls
                      double tmpPosX = robot.position.x + std::cos(robot.heading) * robot.velocity;
                      double tmpPosY = robot.position.y - std::sin(robot.heading) * robot.velocity;
                      if (tmpPosX >= this->areaWidth_)
                      {
                          double diffX     = tmpPosX - this->areaWidth_;
                          robot.position.x = this->areaWidth_ - diffX;
                          robot.heading    = M_PI - robot.heading;
                      }
                      else if (tmpPosX <= 0)
                      {
                          double diffX     = std::abs(tmpPosX);
                          robot.position.x = diffX;
                          robot.heading    = M_PI - robot.heading;
                      }
                      else
                      {
                          robot.position.x = tmpPosX;
                      }
                      if (tmpPosY >= this->areaHeight_)
                      {
                          double diffY     = tmpPosY - this->areaHeight_;
                          robot.position.y = this->areaHeight_ - diffY;
                          robot.heading    = -robot.heading;
                      }
                      else if (tmpPosY <= 0)
                      {
                          double diffY     = std::abs(tmpPosY);
                          robot.position.y = diffY;
                          robot.heading    = -robot.heading;
                      }
                      else
                      {
                          robot.position.y = tmpPosY;
                      }
                      // Update the bounding boxes
                      UpdateRobotBbox(robot);
                  });
}

void MultiRobotEnv::UpdateRobotBbox(Robot &robot)
{
    robot.bbox = raylib::BoundingBox(raylib::Vector3(robot.position.x - robot.radius * ROBOT_BBOX_SIZE_RATIO,
                                                     robot.position.y - robot.radius * ROBOT_BBOX_SIZE_RATIO,
                                                     0),
                                     raylib::Vector3(robot.position.x + robot.radius * ROBOT_BBOX_SIZE_RATIO,
                                                     robot.position.y + robot.radius * ROBOT_BBOX_SIZE_RATIO,
                                                     0));
}

// [c_x, c_y, w, h]
std::vector<Eigen::VectorXd> MultiRobotEnv::GetAllRobotBboxs()
{
    std::vector<Eigen::VectorXd> allBboxes;
    for (const auto &robot : robots_)
    {
        double cx     = (robot.bbox.min.x + robot.bbox.max.x) / 2.0;
        double cy     = (robot.bbox.min.y + robot.bbox.max.y) / 2.0;
        double width  = (robot.bbox.max.x - robot.bbox.min.x);
        double height = (robot.bbox.max.y - robot.bbox.min.y);
        allBboxes.emplace_back(Eigen::Vector4d(cx, cy, width, height));
    }
    return allBboxes;
}