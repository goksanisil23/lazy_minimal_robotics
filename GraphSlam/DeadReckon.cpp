#include "DeadReckon.h"

// Accumlates delta_pose from robot frame into global frame
void DeadReckon::update(const raylib::Vector2 delta_pos, const float delta_rot)
{
    float global_dx = delta_pos.x * cos(heading) - delta_pos.y * sin(heading);
    float global_dy = delta_pos.x * sin(heading) + delta_pos.y * cos(heading);

    // Update the global pose
    position.x += global_dx;
    position.y += global_dy;
    heading += delta_rot;

    // Normalize the angle to remain within -PI to PI
    heading = fmod(heading + M_PI, 2 * M_PI);
    if (heading < 0)
        heading += 2 * M_PI;
    heading -= M_PI;
}