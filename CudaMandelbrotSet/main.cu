#include <Windows.h>
#include <SDL.h>
#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 768

SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;
SDL_Texture* texture = nullptr;

// Инициализирует окно, рендерер и текстуру
bool initWindow()
{
    bool ok = true;

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        ok = false;
    }

    window = SDL_CreateWindow("CudaMandelbrot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window)
    {
        ok = false;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        ok = false;
    }

    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);
    if (!texture)
    {
        ok = false;
    }

    return ok;
}

// Возвращает цвет в множестве Мандельброта
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

__global__ void computeMandelbrot(Uint8* __restrict__ colors, double min_x, double min_y, const double x_diff, const double y_diff)
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

// Генерирует множество Мандельброта на экране
void renderMandelbrotSet(double min_x, double max_x, double min_y, double max_y)
{
    const double x_diff = (max_x - min_x) / SCREEN_WIDTH;
    const double y_diff = (max_y - min_y) / SCREEN_HEIGHT;

    Uint8* device_colors;
    cudaMalloc((void**)&device_colors, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint8));

    dim3 blockSize(16, 16);
    dim3 gridSize((SCREEN_WIDTH + 15) / 16, (SCREEN_HEIGHT + 15) / 16);
    computeMandelbrot <<<gridSize, blockSize >>> (device_colors, min_x, min_y, x_diff, y_diff);

    Uint32* pixels = nullptr;
    int pitch;
    SDL_LockTexture(texture, nullptr, (void**)&pixels, &pitch);

    Uint8* host_colors = new Uint8[SCREEN_WIDTH * SCREEN_HEIGHT];
    cudaMemcpy(host_colors, device_colors, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint8), cudaMemcpyDeviceToHost);

    #pragma omp parallel for colapse(2)
    for (int i = 0; i < SCREEN_HEIGHT; i++) {
        for (int j = 0; j < SCREEN_WIDTH; j++) {
            Uint8 color = host_colors[i * SCREEN_WIDTH + j];
            Uint32 pixelColor = 0xFF000000 | ((color * 5) % 256 << 16) | ((color * 7) % 256 << 8) | ((color * 11) % 256);
            pixels[i * (pitch / 4) + j] = pixelColor;
        }
    }

    SDL_UnlockTexture(texture);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);

    cudaFree(device_colors);
    delete[] host_colors;
}

int main(int argc, char* argv[])
{
    ::ShowWindow(::GetConsoleWindow(), SW_HIDE);
    SDL_Init(SDL_INIT_EVERYTHING);
    initWindow();

    double curent_min_x = -1, curent_max_x = 1, curent_min_y = -1, curent_max_y = 1;
    double curent_move_step = (curent_max_x - curent_min_x) / 10;

    renderMandelbrotSet(curent_min_x, curent_max_x, curent_min_y, curent_max_y);

    SDL_Event window_event;
    while (true)
    {
        if (SDL_PollEvent(&window_event))
        {
            bool rerender_needed = false;

            if (SDL_MOUSEWHEEL == window_event.type)
            {
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
                curent_move_step = (curent_max_x - curent_min_x) / 10;
            }

            if (SDL_KEYDOWN == window_event.type)
            {
                if (window_event.key.keysym.sym == SDLK_UP) {
                    curent_min_y -= curent_move_step;
                    curent_max_y -= curent_move_step;
                }
                else if (window_event.key.keysym.sym == SDLK_DOWN) {
                    curent_min_y += curent_move_step;
                    curent_max_y += curent_move_step;
                }
                else if (window_event.key.keysym.sym == SDLK_LEFT) {
                    curent_min_x -= curent_move_step;
                    curent_max_x -= curent_move_step;
                }
                else if (window_event.key.keysym.sym == SDLK_RIGHT) {
                    curent_min_x += curent_move_step;
                    curent_max_x += curent_move_step;
                }
                rerender_needed = true;
                curent_move_step = (curent_max_x - curent_min_x) / 10;
            }

            if (rerender_needed)
                renderMandelbrotSet(curent_min_x, curent_max_x, curent_min_y, curent_max_y);

            if (SDL_QUIT == window_event.type)
                break;
        }
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();
    return EXIT_SUCCESS;
}
