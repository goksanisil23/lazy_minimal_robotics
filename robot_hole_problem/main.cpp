#include "raylib.h"

int main() {

    // ----- Initialization -----
    const int screenWidth = 1600;
    const int screenHeight = 900;

    InitWindow(screenWidth, screenHeight, "Robot-Hole problem execution visualized with Raylib");

    // Define the camera observing the 3D world
    Camera3D camera = {0}; // partial initialization of all struct members to 0
    camera.position = (Vector3){100.0f, 100.0f, 100.0f}; // camera position
    camera.target = (Vector3){0.0f, 0.0f, 0.0f}; // camera target it looks at
    camera.up = (Vector3){0.0f, 1.0f, 0.0f}; //camera up vector (rotation axis)
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    SetCameraMode(camera, CAMERA_FREE); // set a free camera mode which allows control over cursor

    SetTargetFPS(50);

    // setup the objects in the scene
    Vector3 cubePosition = {0.0f,0.0f,0.0f}; // center coordinate of the cube
    Vector3 spherePosition = {30.0f,5.0f,40.0f};
    Ray ray = {0}; // ray to be drawn from robot towards the designated hole

    // ----- Main Game Loop -----
    while(!WindowShouldClose()) {
        // ----- Update ----- 
        // (Update the variables and implement the game logic here)
        UpdateCamera(&camera); // updates the camera based on cursor and keyboard inputs

        ray.position = cubePosition;
        ray.direction = spherePosition;

        // ----- Draw -----
        // Draw everything that requires to be drawn here
        BeginDrawing();
            ClearBackground(RAYWHITE);
            BeginMode3D(camera);

                // DrawText("Sample Text gisil", 190, 200, 20, LIGHTGRAY);
                DrawCube(cubePosition, 2.0f, 2.0f, 2.0f, RED); // size of dimensions + color
                DrawCubeWires(cubePosition, 2.0f, 2.0f, 2.0f, MAROON);
                DrawSphere(spherePosition, 2.0f, BLUE);
                // DrawSphereWires(spherePosition, 2.0f, 20, 20, MAROON);

                DrawGrid(100, 1.0f);
            
            EndMode3D();

            DrawFPS(10,10); // cannot draw this in 3d mode


        EndDrawing();
    }

    // De-Initialization
    // -----------------------------------------------
    // (Unload all loaded resources at this point)
    CloseWindow();

    return 0;
}