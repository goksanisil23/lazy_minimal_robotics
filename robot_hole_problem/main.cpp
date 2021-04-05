#include "raylib.h"

int main() {

    // Initialization
    // ----------------------------------------------
    const int screenWidth = 800;
    const int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "Robot-Hole problem execution visualized with Raylib");

    SetTargetFPS(50);

    // Main Game Loop
    while(!WindowShouldClose()) {
        // Update 
        // (Update the variables and implement the game logic here)

        // Draw
        BeginDrawing();
            ClearBackground(RAYWHITE);

            // Draw everything that requires to be drawn here
            DrawText("Sample Text gisil", 190, 200, 20, LIGHTGRAY);

        EndDrawing();
    }

    // De-Initialization
    // -----------------------------------------------
    // (Unload all loaded resources at this point)
    CloseWindow();

    return 0;
}