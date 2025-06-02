
// CLE 24'25

#include <cuda_runtime.h>

namespace CANNY::TENSOR {
    __global__ void convolution_kernel(const pixel_t *in, pixel_t *out, const float *kernel, const int nx, const int ny, const int kn, const int khalf){
        int x = blockIdx.x * blockDim.x + threadIdx.x + khalf;
        int y = blockIdx.y * blockDim.y + threadIdx.y + khalf;
        int z = blockIdx.z;

        if(x >= khalf && x < nx - khalf && y >= khalf && y < ny - khalf){
            float pixel = 0.0f;

            for (int j = -khalf; j <= khalf; j++){
                for (int i = -khalf; i <= khalf; i++) {
                    int xi = x - i;
                    int yj = y - j;
                    int idx = z * (nx * ny) + yj * nx + xi;
                    pixel += in[idx] * kernel[(j + khalf) * kn + (i + khalf)];
                }
            }

            out[z * (nx * ny) + y * nx + x] = static_cast<pixel_t>(pixel);
        }
    }


    void convolution_device(const pixel_t *in, pixel_t *out, const float *kernel, const int nx, const int ny, const int kn){
        int khalf = kn >> 1;

        dim3 gridSize(
            (nx - 2 * khalf + 15) / 16,
            (ny - 2 * khalf + 15) / 16,
            8
        );              
        dim3 blockSize(16, 16, 1); 
        
        convolution_kernel<<<gridSize, blockSize>>>(in, out, kernel, nx, ny, kn, khalf);
    }

    __global__ void non_maximum_supression_kernel(const pixel_t *after_Gx, const pixel_t * after_Gy, const pixel_t *G, pixel_t *nms, const int nx, const int ny){
        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int z = blockIdx.z;

        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            int slice_offset = z * nx * ny;
            int c = slice_offset + i + nx * j;

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

    __global__ void first_edges_kernel(const pixel_t *nms, pixel_t *out, const int nx, const int ny, const int tmax){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z;

        if(i < nx && j < ny){
            size_t c = z * nx * ny + j * nx + i;
            if(nms[c] >= tmax){
                out[c] = MAX_BRIGHTNESS;
            } else{
                out[c] = 0;
            }
        }
    }

    __global__ void hysteresis_edges_kernel(const pixel_t *nms, pixel_t *out, const int nx, const int ny, const int tmin, int *changed){
        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int z = blockIdx.z;

        if(i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1){
            int slice_offset = z * nx * ny;
            int t = slice_offset + j * nx + i;

            int nbs[8]; // neighbours
            nbs[0] = t - nx;     // nn
            nbs[1] = t + nx;     // ss
            nbs[2] = t + 1;      // ww
            nbs[3] = t - 1;      // ee
            nbs[4] = nbs[0] + 1; // nw
            nbs[5] = nbs[0] - 1; // ne
            nbs[6] = nbs[1] + 1; // sw
            nbs[7] = nbs[1] - 1; // se

            if (nms[t] >= tmin && out[t] == 0) {
                for(int k = 0; k < 8; k++)
                    if (out[nbs[k]] != 0) {
                        out[t] = MAX_BRIGHTNESS;
                        atomicOr(changed, 1);
                        break;
                    }
            }
        }
    }

    __global__ void min_max_kernel(const pixel_t *in, const int nx, const int ny, pixel_t *min_out, pixel_t *max_out){
        extern __shared__ int shared_data[];
        int *shared_min = shared_data;
        int *shared_max = shared_data + blockDim.x * blockDim.y;

        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z;

        if (x >= nx || y >= ny) return;

        int idx = z * (nx * ny) + y * nx + x;

        // thread per pixel
        int pixel = in[idx];
        shared_min[tid] = pixel;
        shared_max[tid] = pixel;

        __syncthreads();

        // reduce shared memory with binary tree 
        for (int stride = (blockDim.x * blockDim.y * blockDim.z) >> 1; stride > 0; stride >>= 1) {
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

    __global__ void normalize_kernel(pixel_t *inout, const int nx, const int ny, const int kn, const int min_val, const int max_val){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z;
        
        const int khalf = kn >> 1;
        
        if (x >= khalf && x < nx - khalf && y >= khalf && y < ny - khalf) {
            int idx = z * ny * nx + y * nx + x;
            pixel_t pixel = MAX_BRIGHTNESS * ((int)inout[idx] - (float)min_val) / ((float)max_val - (float)min_val);
            inout[idx] = pixel;
        }
    }

    __global__ void generate_gaussian_kernel(float *kernel, const int n, const float sigma){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x < n && y < n) {
            float mean = floorf(n / 2.0f);
            int idx = y * n + x;
            
            kernel[idx] = expf(-0.5f * (powf((x - mean) / sigma, 2.0f) + powf((y - mean) / sigma, 2.0f))) / (2.0f * M_PI * sigma * sigma);
        }
    }

    void gaussian_filter_device(const pixel_t *in, pixel_t *out, const int nx, const int ny, const float sigma){
        const int n = 2 * (int)(2 * sigma) + 3;
        
        float *d_kernel;
        pixel_t *d_min, *d_max;
        
        cudaMalloc((void**)&d_kernel, n * n * sizeof(float));
        cudaMalloc((void**)&d_min, sizeof(pixel_t));
        cudaMalloc((void**)&d_max, sizeof(pixel_t));
        
        //! Tune (I read in this presentation that this had best results: https://www.slideshare.net/slideshow/gaussian-image-blurring-in-cuda-c/56869492)
        dim3 blockDim(16, 16);
        dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
        generate_gaussian_kernel<<<gridDim, blockDim>>>(d_kernel, n, sigma);

        convolution_device(in, out, d_kernel, nx, ny, n);
        
        // Get min and max 
        int sharedMemSize = 2 * blockDim.x * blockDim.y * sizeof(int);

        for (int z = 0; z < 8; ++z) {
            pixel_t *img = out + z * nx * ny;

            pixel_t h_min = INT_MAX, h_max = -INT_MAX;
            cudaMemcpy(d_min, &h_min, sizeof(pixel_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_max, &h_max, sizeof(pixel_t), cudaMemcpyHostToDevice);

            dim3 gridMinMax((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

            min_max_kernel<<<gridMinMax, blockDim, sharedMemSize>>>(img, nx, ny, d_min, d_max);

            cudaMemcpy(&h_min, d_min, sizeof(pixel_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_max, d_max, sizeof(pixel_t), cudaMemcpyDeviceToHost);

            normalize_kernel<<<gridMinMax, blockDim>>>(img, nx, ny, n, h_min, h_max);
        }
        
        cudaFree(d_kernel);
        cudaFree(d_min);
        cudaFree(d_max);
    }

    __global__ void gradient_merge_kernel(pixel_t *after_Gx, pixel_t *after_Gy, pixel_t *G, const int nx, const int ny){
        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int z = blockIdx.z;

        if(i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1){
            int c = z * ny * nx + j * nx + i;
            G[c] = (pixel_t)(hypot((double)(after_Gx[c]), (double)( after_Gy[c]) ));
        }
    }

    // canny edge detector code to run on the GPU
    void cannyDevice( const int *h_idata, const int w, const int h, const int tmin, const int tmax, const float sigma, int * h_odata){
        const int nx = w;
        const int ny = h;

        // Image size (each pixel 1 byte as it is grayscled)
        const size_t nBytes = nx * ny * 8 * sizeof(int);

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
        dim3 gridDim(ceil((nx-2) / 16.0), ceil((ny-2) / 16.0), 8); // exclude the 2 border pixels on gradient merge (x0 and nx-1) (y0 and ny-1)

        gradient_merge_kernel<<<gridDim, blockDim>>>(after_Gx, after_Gy, G, nx, ny);

        // 1 thread per pixel
        dim3 blockSize(16, 16); 
        dim3 gridSize(ceil(nx / 16.0), ceil(ny / 16.0), 8);

        non_maximum_supression_kernel<<<gridSize, blockSize>>>(after_Gx, after_Gy, G, nms, nx, ny);
        
        cudaMemset(d_odata, 0, sizeof(pixel_t) * nx * ny * 8);
        first_edges_kernel<<<gridSize, blockSize>>>(nms, d_odata, nx, ny, tmax);

        int h_changed;
        int *d_changed;
        cudaMalloc(&d_changed, sizeof(int));
        do {
            h_changed = 0;
            cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
            hysteresis_edges_kernel<<<gridSize, blockSize>>>(nms, d_odata, nx, ny, tmin, d_changed);
            cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        } while (h_changed != 0);

        // d_odata -> h_odata
        cudaMemcpy(h_odata, d_odata, nx * ny * 8 * sizeof(pixel_t), cudaMemcpyDeviceToHost);

        cudaFree(d_changed);
        cudaFree(d_idata);
        cudaFree(d_odata);
        cudaFree(after_Gx);
        cudaFree(after_Gy);
        cudaFree(G);
        cudaFree(nms);
    }
}
