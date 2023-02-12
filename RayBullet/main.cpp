#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <btBulletDynamicsCommon.h>

#include <raylib-cpp.hpp>

namespace rl = raylib;

constexpr int screenWidth  = 1200;
constexpr int screenHeight = 1000;

struct Robot
{
    Robot(const btVector3 &position0, const double &yaw0, const double &radius)
    {
        // Create a sphere shape for the robots
        btCollisionShape *robotShape = new btSphereShape(radius);
        btVector3         inertia(0, 0, 0);
        btScalar          mass = 1.0;
        robotShape->calculateLocalInertia(mass, inertia);

        // TODO: use yaw

        // Create a motion state for each robot
        btDefaultMotionState *robotMotionState =
            new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), position0));

        // Create rigid bodies for each robot
        btRigidBody::btRigidBodyConstructionInfo robotConstructionInfo1(mass, robotMotionState, robotShape, inertia);
        rigidBody = std::make_shared<btRigidBody>(robotConstructionInfo1);
    }

    raylib::Vector3              rlPos;
    std::shared_ptr<btRigidBody> rigidBody;
    btVector3                    velocity;
};

int main()
{
    // --------------- Physics setup --------------- //
    // Set up the physics world
    btDefaultCollisionConfiguration     *collisionConfiguration = new btDefaultCollisionConfiguration();
    btCollisionDispatcher               *dispatcher             = new btCollisionDispatcher(collisionConfiguration);
    btDbvtBroadphase                    *broadphase             = new btDbvtBroadphase();
    btSequentialImpulseConstraintSolver *solver                 = new btSequentialImpulseConstraintSolver();
    btDiscreteDynamicsWorld             *dynamicsWorld =
        new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
    dynamicsWorld->setGravity(btVector3(0, 0, 0));

    // Generate the robots
    Robot rob1(btVector3(0, -10, 0), 0, 1.0);
    Robot rob2(btVector3(0, 10, 0), 0, 1.0);

    dynamicsWorld->addRigidBody(rob1.rigidBody.get());
    dynamicsWorld->addRigidBody(rob2.rigidBody.get());

    rob1.velocity = btVector3(0, 1, 0);
    rob1.rigidBody->setLinearVelocity(rob1.velocity);

    // --------------- Graphics setup --------------- //
    rl::Window   w(screenWidth, screenHeight, "Multiple Robots");
    rl::Camera3D camera({5.0f, 5.0f, 5.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 45.0f, CAMERA_PERSPECTIVE);
    camera.SetMode(CAMERA_FREE);
    SetTargetFPS(60);

    // Simulate the robots
    while (!w.ShouldClose())
    {
        auto rayTo   = rob1.rigidBody->getWorldTransform().getOrigin();
        auto rayFrom = rob1.rigidBody->getWorldTransform().getOrigin() + rob1.velocity.normalized() * 100;
        btCollisionWorld::ClosestRayResultCallback rayCallback(rayTo, rayFrom);
        dynamicsWorld->rayTest(rayTo, rayFrom, rayCallback);

        auto pos = rob1.rigidBody->getCenterOfMassPosition();
        std::cout << pos.x() << " " << pos.y() << " " << pos.z() << std::endl;

        // Step the simulation
        dynamicsWorld->stepSimulation(1.0 / 60.0, 10);

        // Draw robots and environment
        camera.Update();
        w.BeginDrawing();
        {
            w.ClearBackground(BLACK);
            camera.BeginMode();
            {
                DrawGrid(10, 1.0f);

                rl::Vector3 posRob1(rob1.rigidBody->getCenterOfMassPosition().x(),
                                    rob1.rigidBody->getCenterOfMassPosition().y(),
                                    rob1.rigidBody->getCenterOfMassPosition().z());
                DrawSphere(posRob1, 1.0, RED);

                rl::Vector3 posRob2(rob2.rigidBody->getCenterOfMassPosition().x(),
                                    rob2.rigidBody->getCenterOfMassPosition().y(),
                                    rob2.rigidBody->getCenterOfMassPosition().z());
                DrawSphere(posRob2, 1.0, GREEN);

                DrawLine3D(posRob1, posRob2, MAGENTA);
                if (rayCallback.hasHit())
                {
                    DrawPoint3D(rl::Vector3(rayCallback.m_hitPointWorld.getX(),
                                            rayCallback.m_hitPointWorld.getY(),
                                            rayCallback.m_hitPointWorld.getZ()),
                                BLUE);
                }
            }
            camera.EndMode();
        }
        w.EndDrawing();
    }

    // Clean up
    dynamicsWorld->removeRigidBody(rob1.rigidBody.get());
    dynamicsWorld->removeRigidBody(rob2.rigidBody.get());
    delete dynamicsWorld;
    delete solver;
    delete broadphase;
    delete dispatcher;
    delete collisionConfiguration;

    return 0;
}