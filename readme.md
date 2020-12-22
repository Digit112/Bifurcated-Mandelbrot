# Crude Engine for rendering the bifurcated Mandelbrot Set.

Generates videos of the Bifurcated Mandelbrot Set. An example is given [here](https://www.youtube.com/watch?v=OOiBGJ0QATs).
The path the camera takes, orbiting the set, is currently the only path. Maybe if someone wants I'll have it follow bezier paths optionally.

The Bifurcated Mandelbrot Set is a visualization of the cycles adopted by points on the complex plane under the iterative formula ![equation](https://latex.codecogs.com/gif.latex?z_%7Bn&plus;1%7D%3Dz_n%5E2&plus;c) for which the Mandelbrot set is defined.

## Programs

#### bifurcate.cpp
Creates a file called out.bin containing cyclic data for a set of points in the mandelbrot set. This data is effectively the 3D model of the object being rendered, although the data is not at all like a 3D model but rather is more analogous to topological data. It's like a heightmap where each pixel can have any number of values associated with it.

#### render.cpp
Using the data in out.bin, this files calls an OpenCL kernel (Whose source code is in render.cl) that runs on your GPU to render an image of the data. It is a ray marching engine that functions by casting rays from each pixel in the output image and advancing them forward a small amount, and checking the pixel in the 3D data that corresponds to their x, y location to see if their z value is close to any of the listed values for that pixel in the data. It then outputs, as the value for that pixel, a single byte corresponding to the z value that ray. This data is used to color the image later when these "images" are converted to actual PNG files. For each frame, one of these bin files is placed into ./out

render.cpp must be compiled with OpenCL installed, and linked with -lOpenCL

#### render_CPU.cpp
The same as render.cpp, but does not use OpenCL and instead runs entirely on CPU. It is also about 40 times slower on my machine.

#### convert.py
This converts the bin files in ./out to PNG images in ./anim. Each byte in a bin file corresponds to the z value of the cell that was being rendered, and is used to modify the hue of the color of that pixel. This gives the effect that a pixel's color depends on its z value. The byte value of 0, however, is reserved for black pixels. It also flips the image vertically (which is necessary because of reasons) and scales it down for anti-aliasing reasons.

#### animate.py
Takes the number of frames specified from ./anim and combines them into a gif file, although using ffmpeg for this one is probably prefferred.

#### vec.hpp / vec.cpp
My personal vector library.

## Theory

The mandelbort set is defined as the set of complex numbers such that z remains bounded under repeated iteration of the equation ![equation](https://latex.codecogs.com/gif.latex?z_%7Bn&plus;1%7D%3Dz_n%5E2&plus;c) where z is initialized to 0 and c is the point being tested.

In other words, you start with z at 0 + 0i on the complex plane, and c at some point. If the point is (1, 2) then c would equal 1 + 2i. You then repeatedly apply the function, assigning the result back to z after each iteration. If z approaches some constant or enters a cycle, then c belongs to the mandelbrot set. If, however, it grows towards infinity, it is not.

For simplicity, we will use real numbers with no imaginary component to simplify the math. (These points lie along the x, or "real" axis)
![equation](https://latex.codecogs.com/gif.latex?z%3D0&plus;0i%3D0)

![equation](https://latex.codecogs.com/gif.latex?c%3D1&plus;0i%3D1)

![equation](https://latex.codecogs.com/gif.latex?0%5E2&plus;1%3D1)

![equation](https://latex.codecogs.com/gif.latex?1%5E2&plus;1%3D2)

![equation](https://latex.codecogs.com/gif.latex?2%5E2&plus;1%3D5)

![equation](https://latex.codecogs.com/gif.latex?5%5E2&plus;1%3D26)

![equation](https://latex.codecogs.com/gif.latex?26%5E2&plus;1%3D677)

Because the number approaches infinity, (1, 0) does NOT belong to the Mandelbrot set.

Sidenote, squaring a complex number a + bi looks like 
![equation](https://latex.codecogs.com/gif.latex?%28a%20&plus;%20bi%29%5E2%20%3D%20a%5E2-b%5E2%20&plus;%202abi)
, so in code, that would just be:
```
new_a = a*a - b*b;
new_b = 2*a*b;
```

Here is another example, which demonstrates one of two ways z may remain bounded.
![equation](https://latex.codecogs.com/gif.latex?z%3D0&plus;0i%3D0)

![equation](https://latex.codecogs.com/gif.latex?c%3D-1&plus;0i%3D-1)

![equation](https://latex.codecogs.com/gif.latex?0%5E2&plus;-1%3D-1)

![equation](https://latex.codecogs.com/gif.latex?-1%5E2&plus;-1%3D0)

![equation](https://latex.codecogs.com/gif.latex?0%5E2&plus;-1%3D-1)

![equation](https://latex.codecogs.com/gif.latex?-1%5E2&plus;-1%3D0)

![equation](https://latex.codecogs.com/gif.latex?0%5E2&plus;-1%3D-1)

Clearly, this results in a cycle. Normally, simply seeing the mandelbrot set can be interesting, but what about also seeing the cycles that various points adopt?

In my program, the xy axes compose the complex plane, as usual, but an additional z axis is added in which the real component of the values visited by the repeated iteration formula are plotted. For instance, in my data, you will find points (-1, 0, 0) and (-1, 0, -1) Representing that point (-1, 0) visits real values 0 and -1 in its cycle. The imaginary values visited are ignored.

These values are found by performing the iterative function many times, each time searching the data for a cycle.
If the function approaches a fixed point, this will be found and treated as a cycle of length 1. If the point is not a member of the Mandelbrot set, it is treated as a cycle of length 0, which is a special case.

With this data, we render the final image via ray marching. Each pixel gets a ray cast out from it which moves forward until it gets close to an entry in the data, then it gets colored based on its z value.
