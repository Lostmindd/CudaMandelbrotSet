#include <Windows.h>
#include <SDL.h>
#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int screen_width = 1024;
int screen_height = 640;

SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;
SDL_Texture* texture = nullptr;

// Initializes the window, renderer and texture
void initWindow()
{
    window = SDL_CreateWindow("CudaMandelbrot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, screen_width, screen_height, SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, screen_width, screen_height);
}

// Returns the color in the Mandelbrot set for x and y
__device__ Uint8 getMandelbrotColor(double x, double y)
{
    double z1 = 0, z2 = 0;
    #pragma unroll 16
    for (Uint8 step = 0; step < 100; ++step)
    {
        double new_z1 = z1 * z1 - z2 * z2 + x;
        double new_z2 = 2 * z1 * z2 + y;
        if ((new_z1 * new_z1 + new_z2 * new_z2) > 4)
            return step * 2.5;
        z1 = new_z1;
        z2 = new_z2;
    }
    return 255;
}

// Compute the colors of each pixel for the Mandelbrot set on the GPU
__global__ void computeMandelbrot(Uint8* __restrict__ colors, double min_x, double min_y, 
    const double x_diff, const double y_diff, const int SCREEN_WIDTH, const int SCREEN_HEIGHT)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < SCREEN_WIDTH && j < SCREEN_HEIGHT)
    {
        double x = min_x + i * x_diff;
        double y = min_y + j * y_diff;
        colors[j * SCREEN_WIDTH + i] = getMandelbrotColor(x, y);
    }
}

// Renders the Mandelbrot set on the screen
void renderMandelbrotSet(double min_x, double max_x, double min_y, double max_y)
{
    const double x_diff = (max_x - min_x) / screen_width;
    const double y_diff = (max_y - min_y) / screen_height;

    Uint8* colors_on_gpu;
    cudaMalloc((void**)&colors_on_gpu, screen_width * screen_height * sizeof(Uint8));

    dim3 blockSize(32, 32);
    dim3 gridSize((screen_width + 31) / 32, (screen_height + 31) / 32);
    computeMandelbrot <<<gridSize, blockSize>>> (colors_on_gpu, min_x, min_y, x_diff, y_diff, screen_width, screen_height);

    Uint8* colors_on_cpu = new Uint8[screen_width * screen_height];
    //copies to cpu only when all blocks complete their work
    cudaMemcpy(colors_on_cpu, colors_on_gpu, screen_width * screen_height * sizeof(Uint8), cudaMemcpyDeviceToHost);

    Uint32* pixels = nullptr;
    int pitch; //length of one texture line in bytes
    SDL_LockTexture(texture, nullptr, (void**)&pixels, &pitch);

    #pragma omp parallel for colapse(2)
    for (int i = 0; i < screen_height; i++) {
        for (int j = 0; j < screen_width; j++) {
            Uint8 color = colors_on_cpu[i * screen_width + j];
            Uint32 pixelColor = 0xFF000000 | ((color * 5) % 256 << 16) | ((color * 7) % 256 << 8) | ((color * 11) % 256);
            pixels[i * (pitch / 4) + j] = pixelColor;
        }
    }

    SDL_UnlockTexture(texture);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);

    cudaFree(colors_on_gpu);
    delete[] colors_on_cpu;
}

// Ends the program
void quit()
{
    SDL_DestroyTexture(texture);
    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();
    exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[])
{
    ::ShowWindow(::GetConsoleWindow(), SW_HIDE);
    if (argc == 3) 
    {
        screen_width = atoi(argv[1]);  
        screen_height = atoi(argv[2]); 
    }

    SDL_Init(SDL_INIT_EVERYTHING);
    initWindow();

    double curent_min_x = -2, curent_max_x = 1, curent_min_y = -1.5, curent_max_y = 1.5;
    double curent_move_step = (curent_max_x - curent_min_x) / 10;

    renderMandelbrotSet(curent_min_x, curent_max_x, curent_min_y, curent_max_y);

    SDL_Event window_event;
    while (true)
    {
        if (SDL_PollEvent(&window_event))
        {
            bool rerender_needed = false;

            switch (window_event.type)
            {
            case SDL_MOUSEWHEEL:
                if (window_event.wheel.y == -1)
                {
                    curent_min_x -= curent_move_step;
                    curent_max_x += curent_move_step;
                    curent_min_y -= curent_move_step;
                    curent_max_y += curent_move_step;
                }
                else if (window_event.wheel.y == 1)
                {
                    curent_min_x += curent_move_step;
                    curent_max_x -= curent_move_step;
                    curent_min_y += curent_move_step;
                    curent_max_y -= curent_move_step;
                }
                rerender_needed = true;
                curent_move_step = (curent_max_x - curent_min_x) / 8;
                break;

            case SDL_KEYDOWN:
                switch (window_event.key.keysym.sym)
                {
                case SDLK_UP:
                    curent_min_y -= curent_move_step;
                    curent_max_y -= curent_move_step;
                    break;
                case SDLK_DOWN:
                    curent_min_y += curent_move_step;
                    curent_max_y += curent_move_step;
                    break;
                case SDLK_LEFT:
                    curent_min_x -= curent_move_step;
                    curent_max_x -= curent_move_step;
                    break;
                case SDLK_RIGHT:
                    curent_min_x += curent_move_step;
                    curent_max_x += curent_move_step;
                    break;
                case SDLK_ESCAPE:
                    quit();
                }
                rerender_needed = true;
                curent_move_step = (curent_max_x - curent_min_x) / 8;
                break;

            case SDL_QUIT:
                quit();
            }
            if (rerender_needed)
                renderMandelbrotSet(curent_min_x, curent_max_x, curent_min_y, curent_max_y);
        }
    }
}