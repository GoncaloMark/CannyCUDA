# Canny Edge Filter CUDA

## Introduction

This project implements a CUDA-accelerated version of the **Canny Edge Detector** algorithm for grayscale images. The Canny Edge Detector operates in four main steps:

1. **Gaussian Filter** – Smooths the image to reduce noise.
2. **Gradient Computation** – Calculates intensity gradients on horizontal and vertical axis.
3. **Non-Maximum Suppression** – Retains only the local maxima in the gradient directions.
4. **Hysteresis Thresholding** – Traces edges by connecting strong and weak edge pixels.

The original sequential implementation is provided in C. This project introduces several CUDA kernels to parallelize each of the major steps, improving performance by utilizing GPU resources efficiently.

---

## Setup and Usage

### Compilation

To compile the project:

```bash
make
```

### Execution Options

The program supports different execution modes depending on the level of optimization. The available modes are:

| Command               | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| `./canny`              | Default execution using basic CUDA device implementation           |
| `./canny -m`           | Uses shared memory optimized kernels                               |
| `./canny -t in.txt`    | Uses a tensor inspired variant (reads input from a file)           |
| `./canny -f`           | Runs the performance-focused optimized version of ./canny          |

### Running Tests and Collecting Results

To benchmark performance across all modes, run:

```bash
python3 tests.py
```

The script performs the following actions:

- Executes each mode 5 times  
- Extracts timing information for both host and device  
- Calculates average, median, and standard deviation  
- Saves results to `results.txt`

---

## Approaches

### `canny-device.cu`

This implementation follows the classical stages of the Canny Edge Detector, mapping each step to one or more CUDA kernels. Below we describe each step and how it is implemented in parallel.

#### 1. Convolution kernel

Performs generic 2D convolution:

- Each thread computes a single pixel using a sliding kernel.
- Supports arbitrary odd-sized kernels.
- Used for Gaussian filtering and Sobel gradient calculation (Gx, Gy).

#### 2. Generate Gaussian Filter

Generates and applies a Gaussian filter:

- Dynamically generates a Gaussian kernel on the device using `generate_gaussian_kernel`.
- Uses `convolution_kernel` to apply the Gaussian blur.
- Normalizes the result using `min_max_kernel` (shared memory + `atomicMin/Max`) and `normalize_kernel`.

#### 3. Non Maximum Supression

Thins out edges by suppressing non-maximum pixels:

- Computes gradient direction using `atan2f(Gy, Gx)`.
- Compares each pixel to its neighbors in the gradient direction.
- Keeps only local maxima; others are set to 0.

#### 4. First Edges

Applies strong thresholding:

- Marks pixels above `tmax` as strong edges (`255`).
- Others remain at zero.

#### 5. Hysteresis Edges

Performs edge tracking by hysteresis:

- Promotes weak pixels (above `tmin`) if connected to strong edges.
- Uses 8-connected neighborhood.
- Uses `atomicOr` to track convergence across iterations.

---

### `canny-device-sm.cu`

In this implementation, the main difference lies in the use of **shared memory** within the CUDA kernels for the convolution and hysteresis edge detection steps. This optimization improves performance by reducing global memory access latency and increasing data reuse.

#### 1. Shared Memory in Convolution Kernel

#### How Shared Memory is Used:

- Instead of each thread reading pixel data directly from **global memory** multiple times, the kernel first loads a **tile** (a block of pixels) from global memory into **shared memory**.
- Shared memory is a small, fast memory located on the GPU chip, accessible by all threads within the same block.
- Threads then perform convolution operations on the pixels in shared memory, which is faster than accessing global memory repeatedly.
- Border handling is done by loading extra pixels (the halo region) around the tile into shared memory, ensuring correct convolution at tile edges.

#### Benefits:

- Reduces the number of slow global memory reads.
- Increases memory access efficiency by reusing pixel data across threads.

#### 2. Shared Memory in Hysteresis Edges Kernel

#### How Shared Memory is used:

- Similar to the convolution kernel, the hysteresis edge detection kernel loads a tile of the edge map into shared memory.
- This tile includes the strong and weak edge pixels that will be examined for connectivity.
- Threads then perform the hysteresis thresholding within shared memory, checking neighbors efficiently without repeated global memory reads.
- The shared memory tile includes surrounding pixels to correctly evaluate edge connectivity across block boundaries.

#### Benefits:

- Speeds up the hysteresis process by minimizing global memory accesses.
- Enables more efficient neighbor checks when determining if a weak edge pixel should be retained.
- Improves overall kernel throughput, contributing to faster edge detection.

---

### `canny-device-speed-opt.cu`

In this version we basically picked up the `canny-device.cu` version and just optimized away certain computations regarding the use of double precision (changing most computing functions to float like versions fmod to fmodf), as GPU's are optimized for single precision floating point math, and also stripped away redundant computations or divisions by using bit shifting and storing single results in variables for reuse instead of redoing the operation. Also we applied some branchless programming techniques.

---

### `canny-tensor.cu`

For the tensor version of the **Canny Edge Detector** we first operated on the loadPGM function provided by the professor as we wanted to store the images in a 1D pre-allocated array. Then we used the same basis for the code but the kernels now had a z axis to work with. Instead of a 3D approach we applied the kernels on a layer basis which still prooved to be more efficient. The host code processes each image on a different thread as we used the 1st project ThreadPool for managing and distributing the work 1 core per image.

---

## Parallel Reduction Algorithm

We employed a Parallel Sequential Addressing Reduction algorithm for the minmax kernel used in the gaussian filter as it is conflict free considering the memory bunks and it has better locality than other interleaved addressing strategies. It can be further optimized as on the first loop half of the threads are idle, this could be fixed by performing a first level reduction from global memory into shared memory.

## Benchmark Results (With `banana.ua.pt`)

GPU: NVIDIA GeForce GTX 1660 Ti (Compute 7.5)

| Approach             | Host Average (ms) | Device Average (ms) | Speedup (Host / Device) |
|----------------------|-------------------|----------------------|--------------------------|
| `./canny`            | 33.199            | 1.315                | 25.24x                   |
| `./canny -m`         | 33.671            | 1.390                | 24.23x                   |
| `./canny -t in.txt`  | 48.563            | 8.309                | 5.84x                    |
| `./canny -f`         | 33.578            | 1.354                | 24.80x                   |

This provides a clear comparison between the host CPU implementation and the various CUDA approaches.

---

## Benchmark Results (With `HP ZBook Fury 16 Personal Laptop`)

GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU (compute 8.9)

| Approach             | Host Average (ms) | Device Average (ms) | Speedup (Host / Device) |
|----------------------|-------------------|----------------------|--------------------------|
| `./canny`            | 31.167            | 1.854                | 16.81x                   |
| `./canny -m`         | 32.074            | 1.866                | 17.19x                   |
| `./canny -t in.txt`  | 44.611            | 7.786                | 5.73x                    |
| `./canny -f`         | 35.364            | 1.212                | 25.76x                   |

With the laptop GPU we can verify that the optimized version is faster than the banana.ua.pt GPU and it has a greater speedup rate too.

---

### Formulas Used

The following formulas were used to compute the performance metrics:

- **Speedup**  
    $$
    \text{Speedup} = \frac{T_1}{T_n}
    $$
    Where $T_1$ is the execution time using only 1 process, and $T_n$ is the execution time using $n$ processes.
