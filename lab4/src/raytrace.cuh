#ifndef RAYTRACE_DEVICE_CUH
#define RAYTRACE_DEVICE_CUH

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

struct Point_Light {
  double position[3]; // vec3
  double color[3];    // vec3
  double attenuation_k;
};

struct Material {
  double diffuse[3];  // vec3
  double ambient[3];  // vec3
  double specular[3]; // vec3
  double shine;
  double snell;
  double opacity;
  double reflectivity;
};

struct Object {
  double e;
  double n;
  Material mat;
  double scale[9];       // 3x3-matrix
  double unScale[9];     // 3x3-matrix
  double rotate[9];      // 3x3-matrix
  double unRotate[9];    // 3x3-matrix
  double translate[3];   // 3-vector
  double unTranslate[3]; // 3-vector
};

#define gpuErrChk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

void callRaytraceKernel(double *grid, Object *objects, Point_Light *lightsPPM,
  double *data, double *bgColor, double *e1, double *e2,
  double *e3, double *lookFrom, int Nx, int Ny,
  bool antiAliased, int blockPower);

#endif
