## Rendering the Mandelbrot set on GPU
the **SDL2** library is used for rendering, and **CUDA** is used to speed up the calculation of points included in the Mandelbrot set.
You can specify the window size for rendering by running the program with the width and height arguments:
```bash
CudaMandelbrotSet WIDTH HEIGHT
```
For example:
```bash
CudaMandelbrotSet 1920 1080
```
