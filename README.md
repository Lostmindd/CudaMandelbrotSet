## Rendering the Mandelbrot Set on GPU

The **CUDA** and **SDL2** libraries are used to render interactive visualization of the Mandelbrot set. By leveraging the power of the GPU, the program achieves significantly faster calculations of the set's intricate fractal structures, enabling smooth zooming.

---


### Example Render
*Here's an example of what the Mandelbrot set might look like after rendering.*

![Mandelbrot Set Example1](./images/1.png)
![Mandelbrot Set Example2](./images/2.png)
![Mandelbrot Set Example3](./images/3.png)

---

### How to Run
To run the program, you need to specify the window size (width and height) as command-line arguments. The format is as follows:

```bash
CudaMandelbrotSet WIDTH HEIGHT
```

#### Example:
```bash
CudaMandelbrotSet 1920 1080
```
This command will render the Mandelbrot set in a 1920x1080 window.

---

### Prerequisites
Ensure that the following software and dependencies are installed on your system:
- **NVIDIA CUDA Toolkit** (version 12.6 or higher recommended)
- A CUDA-capable NVIDIA GPU

---

### How It Works
1. **CUDA Kernel**: The kernel computes the color of each pixel in parallel. Each thread corresponds to a pixel, and the escape-time algorithm is used to determine if the pixel is part of the Mandelbrot set and its color.
2. **Rendering**: The result is rendered in the SDL2 window in real-time, with potential for smooth zooming and navigation.

---

## Controls
- **Arrow keys**: Move the view in the corresponding direction.
- **Mouse wheel**: Zoom in and out of the fractal.

---
