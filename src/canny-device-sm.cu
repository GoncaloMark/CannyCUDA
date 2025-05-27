
// CLE 24'25

#include <cuda_runtime.h>
#define MAX_BRIGHTNESS 255

typedef int pixel_t;

__global__ void convolution_kernel_SM(const pixel_t *in, pixel_t *out, const float *kernel, const int nx, const int ny, const int kn, const int khalf){
    extern __shared__ pixel_t shared[];

    // Global indices
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Local thread indices in block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Shared memory width and height (block size + halo * 2)
    const int shared_w = blockDim.x + 2 * khalf;
    //const int shared_h = blockDim.y + 2 * khalf;

    // Index in shared memory
    const int shared_x = tx + khalf;
    const int shared_y = ty + khalf;

    // Coordinates of pixel this thread is responsible for loading
    int global_idx = y * nx + x;

    // Load central pixel into shared memory
    if (x < nx && y < ny) {
        shared[shared_y * shared_w + shared_x] = in[global_idx];
    }

    // Load halo layers
    for (int layer = 1; layer <= khalf; layer++) {
        // Top halo
        if (ty < layer && (y - layer) >= 0 && x < nx) {
            shared[(shared_y - layer) * shared_w + shared_x] = in[(y - layer) * nx + x];
        }

        // Bottom halo
        if (ty >= blockDim.y - layer && (y + layer) < ny && x < nx) {
            shared[(shared_y + layer) * shared_w + shared_x] = in[(y + layer) * nx + x];
        }

        // Left halo
        if (tx < layer && (x - layer) >= 0 && y < ny) {
            shared[shared_y * shared_w + (shared_x - layer)] = in[y * nx + (x - layer)];
        }

        // Right halo
        if (tx >= blockDim.x - layer && (x + layer) < nx && y < ny) {
            shared[shared_y * shared_w + (shared_x + layer)] = in[y * nx + (x + layer)];
        }

        // Top-left corner
        if (tx < layer && ty < layer && x >= layer && y >= layer) {
            shared[(shared_y - layer) * shared_w + (shared_x - layer)] = in[(y - layer) * nx + (x - layer)];
        }

        // Top-right corner
        if (tx >= blockDim.x - layer && ty < layer && (x + layer) < nx && (y - layer) >= 0) {
            shared[(shared_y - layer) * shared_w + (shared_x + layer)] = in[(y - layer) * nx + (x + layer)];
        }

        // Bottom-left corner
        if (tx < layer && ty >= blockDim.y - layer && (x - layer) >= 0 && (y + layer) < ny) {
            shared[(shared_y + layer) * shared_w + (shared_x - layer)] = in[(y + layer) * nx + (x - layer)];
        }

        // Bottom-right corner
        if (tx >= blockDim.x - layer && ty >= blockDim.y - layer && (x + layer) < nx && (y + layer) < ny) {
            shared[(shared_y + layer) * shared_w + (shared_x + layer)] = in[(y + layer) * nx + (x + layer)];
        }
    }

    __syncthreads();

    // Convolution only where the kernel fits
    if (x >= khalf && x < nx - khalf && y >= khalf && y < ny - khalf) {
        float pixel = 0.0f;

        for (int j = -khalf; j <= khalf; j++) {
            for (int i = -khalf; i <= khalf; i++) {
                int sx = shared_x + i;
                int sy = shared_y + j;
                float val = shared[sy * shared_w + sx];
                float coeff = kernel[(j + khalf) * kn + (i + khalf)];
                pixel += val * coeff;
            }
        }

        out[y * nx + x] = static_cast<pixel_t>(pixel);
    }
}

void convolution_device_SM(const pixel_t *in, pixel_t *out, const float *kernel, const int nx, const int ny, const int kn) {
    int khalf = kn >> 1;

    dim3 blockSize(16, 16);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, 
                  (ny + blockSize.y - 1) / blockSize.y);

    // Total shared memory needed per block:
    int shared_w = blockSize.x + 2 * khalf;
    int shared_h = blockSize.y + 2 * khalf;
    int shared_bytes = shared_w * shared_h * sizeof(pixel_t);

    convolution_kernel<<<gridSize, blockSize, shared_bytes>>>(in, out, kernel, nx, ny, kn, khalf);
}

__global__ void non_maximum_supression_kernel_SM(const pixel_t *after_Gx, const pixel_t * after_Gy, const pixel_t *G, pixel_t *nms, const int nx, const int ny){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        const int c = i + nx * j;
        const int nn = c - nx;
        const int ss = c + nx;
        const int ww = c + 1;
        const int ee = c - 1;
        const int nw = nn + 1;
        const int ne = nn - 1;
        const int sw = ss + 1;
        const int se = ss - 1;

        float dir = (float)(fmod(atan2f(after_Gy[c], after_Gx[c]) + M_PI, M_PI) / M_PI) * 8;

        if (((dir <= 1 || dir > 7) && G[c] > G[ee] && G[c] > G[ww]) ||
            ((dir > 1 && dir <= 3) && G[c] > G[nw] && G[c] > G[se]) ||
            ((dir > 3 && dir <= 5) && G[c] > G[nn] && G[c] > G[ss]) ||
            ((dir > 5 && dir <= 7) && G[c] > G[ne] && G[c] > G[sw])) {
            nms[c] = G[c];
        } else {
            nms[c] = 0;
        }
    }
}

__global__ void first_edges_kernel_SM(const pixel_t *nms, pixel_t *out, const int nx, const int ny, const int tmax){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < nx && j < ny){
        size_t c = j * nx + i;
        if(nms[c] >= tmax){
            out[c] = MAX_BRIGHTNESS;
        } else{
            out[c] = 0;
        }
    }
}

__global__ void hysteresis_edges_kernel_SM(
    const pixel_t *nms, pixel_t *out,
    const int nx, const int ny, const int tmin, int *changed
) {
    extern __shared__ unsigned char s_mem[];
    pixel_t *s_nms = (pixel_t*)s_mem;
    pixel_t *s_out = (pixel_t*)&s_nms[(blockDim.x + 2) * (blockDim.y + 2)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int idx = i + j * nx;

    int lx = tx + 1;
    int ly = ty + 1;
    int lidx = lx + ly * (blockDim.x + 2);

    // Shared memory dimensions
    const int s_width = blockDim.x + 2;
    //const int s_height = blockDim.y + 2;

    // Load nms and out to shared memory (no halo needed for nms)
    if (i < nx && j < ny) {
        s_nms[lidx] = nms[idx];
        s_out[lidx] = out[idx];
    }

    // Load halo for out (balanced by edge threads)
    if (tx == 0 && i > 0) s_out[lidx - 1] = out[idx - 1];            // left
    if (tx == blockDim.x - 1 && i < nx - 1) s_out[lidx + 1] = out[idx + 1];  // right
    if (ty == 0 && j > 0) s_out[lidx - s_width] = out[idx - nx];     // top
    if (ty == blockDim.y - 1 && j < ny - 1) s_out[lidx + s_width] = out[idx + nx]; // bottom

    // Corners
    if (tx == 0 && ty == 0 && i > 0 && j > 0)
        s_out[lidx - 1 - s_width] = out[idx - 1 - nx];
    if (tx == blockDim.x - 1 && ty == 0 && i < nx - 1 && j > 0)
        s_out[lidx + 1 - s_width] = out[idx + 1 - nx];
    if (tx == 0 && ty == blockDim.y - 1 && i > 0 && j < ny - 1)
        s_out[lidx - 1 + s_width] = out[idx - 1 + nx];
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1 && i < nx - 1 && j < ny - 1)
        s_out[lidx + 1 + s_width] = out[idx + 1 + nx];

    __syncthreads();

    bool local_changed;
    do {
        local_changed = false;
        __syncthreads();

        if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
            if (s_nms[lidx] >= tmin && s_out[lidx] == 0) {
                int neighbors[8] = {
                    lidx - s_width,         // top
                    lidx + s_width,         // bottom
                    lidx - 1,               // left
                    lidx + 1,               // right
                    lidx - s_width - 1,     // top-left
                    lidx - s_width + 1,     // top-right
                    lidx + s_width - 1,     // bottom-left
                    lidx + s_width + 1      // bottom-right
                };

                for (int k = 0; k < 8; k++) {
                    if (s_out[neighbors[k]] != 0) {
                        s_out[lidx] = MAX_BRIGHTNESS;
                        local_changed = true;

                        // If on block boundary, notify global flag
                        if (tx == 0 || ty == 0 || tx == blockDim.x - 1 || ty == blockDim.y - 1)
                            atomicOr(changed, 1);
                        break;
                    }
                }
            }
        }
        __syncthreads();
    } while (__syncthreads_or(local_changed));

    // Write back to global memory
    if (i < nx && j < ny)
        out[idx] = s_out[lidx];
}

__global__ void min_max_kernel_SM(const pixel_t *in, const int nx, const int ny, pixel_t *min_out, pixel_t *max_out){
    extern __shared__ int shared_data[];
    int *shared_min = shared_data;
    int *shared_max = shared_data + blockDim.x * blockDim.y;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * nx + x;

    if (x >= nx || y >= ny) return;

    // thread per pixel
    int pixel = in[idx];
    shared_min[tid] = pixel;
    shared_max[tid] = pixel;
    __syncthreads();

    // reduce shared memory with binary tree 
    for (int stride = (blockDim.x * blockDim.y) >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(min_out, shared_min[0]);
        atomicMax(max_out, shared_max[0]);
    }
}

__global__ void normalize_kernel_SM(pixel_t *inout, const int nx, const int ny, const int kn, const int min_val, const int max_val){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int khalf = kn >> 1;
    
    if (x >= khalf && x < nx - khalf && y >= khalf && y < ny - khalf) {
        int idx = y * nx + x;
        pixel_t pixel = MAX_BRIGHTNESS * ((int)inout[idx] - (float)min_val) / ((float)max_val - (float)min_val);
        inout[idx] = pixel;
    }
}

__global__ void generate_gaussian_kernel_SM(float *kernel, const int n, const float sigma){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < n && y < n) {
        float mean = floorf(n / 2.0f);
        int idx = y * n + x;
        
        kernel[idx] = expf(-0.5f * (powf((x - mean) / sigma, 2.0f) + powf((y - mean) / sigma, 2.0f))) / (2.0f * M_PI * sigma * sigma);
    }
}

void gaussian_filter_device_SM(const pixel_t *in, pixel_t *out, const int nx, const int ny, const float sigma){
    const int n = 2 * (int)(2 * sigma) + 3;
    
    float *d_kernel;
    pixel_t *d_min, *d_max;
    pixel_t h_min = INT_MAX, h_max = -INT_MAX;
    
    cudaMalloc((void**)&d_kernel, n * n * sizeof(float));
    cudaMalloc((void**)&d_min, sizeof(pixel_t));
    cudaMalloc((void**)&d_max, sizeof(pixel_t));
    
    cudaMemcpy(d_min, &h_min, sizeof(pixel_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(pixel_t), cudaMemcpyHostToDevice);
    
    //! Tune (I read in this presentation that this had best results: https://www.slideshare.net/slideshow/gaussian-image-blurring-in-cuda-c/56869492)
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    
    generate_gaussian_kernel<<<gridDim, blockDim>>>(d_kernel, n, sigma);
    
    convolution_device(in, out, d_kernel, nx, ny, n);

    // Image processing sizes (1 thread per pixel)
    dim3 blockSize(16, 16); 
    dim3 gridSize(ceil(nx / 16.0), ceil(ny / 16.0));
    
    // Get min and max 
    int sharedMemSize = 2 * blockSize.x * blockSize.y * sizeof(int);
    min_max_kernel<<<gridSize, blockSize, sharedMemSize>>>(out, nx, ny, d_min, d_max);
    
    cudaMemcpy(&h_min, d_min, sizeof(pixel_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(pixel_t), cudaMemcpyDeviceToHost);
    
    normalize_kernel<<<gridSize, blockSize>>>(out, nx, ny, n, h_min, h_max);
    
    cudaFree(d_kernel);
    cudaFree(d_min);
    cudaFree(d_max);
}

__global__ void gradient_merge_kernel_SM(pixel_t *after_Gx, pixel_t *after_Gy, pixel_t *G, const int nx, const int ny){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1){
        const int c = i + nx * j;
        G[c] = (pixel_t)(hypot((double)(after_Gx[c]), (double)( after_Gy[c]) ));
    }
}

// canny edge detector code to run on the GPU
void cannyDeviceSM( const int *h_idata, const int w, const int h, const int tmin, const int tmax, const float sigma, int * h_odata){
    const int nx = w;
    const int ny = h;

    // Image size (each pixel 1 byte as it is grayscled)
    const size_t nBytes = nx * ny * sizeof(pixel_t);

    // device input and output
    int *d_idata;
    int *d_odata;

    pixel_t *G, *after_Gx, *after_Gy, *nms;
    float *d_Gx, *d_Gy;

    const float Gx[] = {
                        -1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1
                    };

    const float Gy[] = {
                        1, 2, 1,
                        0, 0, 0,
                        -1,-2,-1
                    };

    cudaMalloc((void**)&d_idata, nBytes);
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_odata, nBytes);

    cudaMalloc((void**)&G, nBytes);
    cudaMalloc((void**)&after_Gx, nBytes);
    cudaMalloc((void**)&after_Gy, nBytes);
    cudaMalloc((void**)&nms, nBytes);

    cudaMalloc((void**)&d_Gx, sizeof(Gx));
    cudaMemcpy(d_Gx, Gx, sizeof(Gx), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_Gy, sizeof(Gy));
    cudaMemcpy(d_Gy, Gy, sizeof(Gy), cudaMemcpyHostToDevice);

    gaussian_filter_device(d_idata, d_odata, nx, ny, sigma);

    // Gradient along x
    convolution_device(d_odata, after_Gx, d_Gx, nx, ny, 3);

    // Gradient along y
    convolution_device(d_odata, after_Gy, d_Gy, nx, ny, 3);

    dim3 blockDim(16, 16); 
    dim3 gridDim(ceil((nx-2) / 16.0), ceil((ny-2) / 16.0)); // exclude the 2 border pixels on gradient merge (x0 and nx-1) (y0 and ny-1)

    gradient_merge_kernel<<<gridDim, blockDim>>>(after_Gx, after_Gy, G, nx, ny);

    // 1 thread per pixel
    dim3 blockSize(16, 16); 
    dim3 gridSize(ceil(nx / 16.0), ceil(ny / 16.0));

    non_maximum_supression_kernel<<<gridSize, blockSize>>>(after_Gx, after_Gy, G, nms, nx, ny);

    cudaMemset(d_odata, 0, sizeof(pixel_t) * nx * ny);
    first_edges_kernel<<<gridSize, blockSize>>>(nms, d_odata, nx, ny, tmax);

    int h_changed;
    int *d_changed;
    cudaMalloc(&d_changed, sizeof(int));

    // Shared memory size: 2 arrays (nms and out) of (blockDim.x+2)*(blockDim.y+2)
    size_t sharedMemSize = 2 * (blockSize.x + 2) * (blockSize.y + 2) * sizeof(pixel_t);

    do {
	h_changed = 0;
	cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

	hysteresis_edges_kernel<<<gridSize, blockSize, sharedMemSize>>>(nms, d_odata, nx, ny, tmin, d_changed);

	cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_changed != 0);

    cudaMemcpy(h_odata, d_odata, nx * ny * sizeof(pixel_t), cudaMemcpyDeviceToHost);
    cudaFree(d_changed);

    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaFree(after_Gx);
    cudaFree(after_Gy);
    cudaFree(G);
    cudaFree(nms);
}
