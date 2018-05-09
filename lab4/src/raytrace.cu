#include "raytrace.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <float.h>
#include "helper_cuda.h"
                                           
//helper_cuda.h contains the error checking macros. note that they're called
//CUDA_CALL and CUBLAS_CALL instead of the previous names

#define RECURSIONDEPTH 3
/*  We are going to use 0 based indexing because we are coming from C
    and aren't using any code or loops from FORTRAN. Cublas will deal
    with it. But you'll need to consider the row-major column major 
    change with the scaling and rotation matrices. 
*/
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__constant__ double vec3zero[3];

/********** Helper Functions ************/
/* Stores the element wise product of a and b into c. */
__device__ void hadamard_product(double *a, double *b, double *c) {
  c[0] = a[0] * b[0];
  c[1] = a[1] * b[1];
  c[2] = a[2] * b[2];
}

/* sets the elements of vec3 a to scalar b */
__device__ void set_vec(double *a, double b) {
  a[0] = b;
  a[1] = b;
  a[2] = b;
}

/* copies the elements of vec3 a into vec3 b */
__device__ void copy_vec(double *a, double *b) {
  //TODO write a cublas version in comments 
  /*
  #####################################
  int len_of_vector = 3;
  int stride = 1;
  cublasDcopy(handle, len_of_vector, a, stride, b, stride);
  #####################################
  */
  b[0] = a[0];
  b[1] = a[1];
  b[2] = a[2];
}

/* Adds the scalar product of vec3 a and scalar b into c. */
__device__ void axpy(double *a, double b, double *c, double *d) {
  //TODO: write an equivalent cublas call in comments
  // assume a cublas handle named "handle" is given. 
  //its axpy from BLAS but allows us to differentiate
  //our output vector from our second input.
  //you'll need to solve that for CUBLAS conversion

  /*
  #####################################
  int len_of_vector = 3;
  int stride = 1;

  // copy c into d
  cublasDcopy(handle, len_of_vector, c, stride, d, stride);


  // execute d[] = b*a[] + d[]
  cublasDaxpy(handle, len_of_vector, &b, a, stride, d, stride)
  #####################################
  */
  d[0] = (a[0] * b) + c[0];
  d[1] = (a[1] * b) + c[1];
  d[2] = (a[2] * b) + c[2];
}

/* multiples the scalar b by vec3 a element wise. */
__device__ void mult_vec_scalar(double *a, double b, double *c) {
  //TODO write a cublas version in comments 
  /*
  #####################################
  int len_of_vector = 3;
  int stride = 1;

  //copy vector a to c
  cublasDcopy(handle, len_of_vector, a, stride, c, stride);
  
  //use scaling function
  cublasDscal(handle, len_of_vector, &b, c, stride)
  #####################################
  */
  c[0] = a[0] * b;
  c[1] = a[1] * b;
  c[2] = a[2] * b;
}

// Stores the element wise addition of vec3 a and vec3 b into c
__device__ void elem_add(double *a, double *b, double *c) {
  //TODO write a cublas version in comments 

  /*
  #####################################
  int len_of_vector = 3;
  int stride = 1;

  //copy vector a to c
  cublasDcopy(handle, len_of_vector, a, stride, c, stride);
  
  double one = 1;

  cublasDaxpy(handle, len_of_vector, &one, b, stride, c, stride);
  #####################################
  */

  c[0] = a[0] + b[0];
  c[1] = a[1] + b[1];
  c[2] = a[2] + b[2];
}

// Stores the element wise subtraction n of vec3 a and vec3 b into c
__device__ void elem_sub(double *a, double *b, double *c) {
  //TODO write a cublas version in comments 
  /*
  #####################################
  int len_of_vector = 3;
  int stride = 1;

  //copy vector a to c
  cublasDcopy(handle, len_of_vector, a, stride, c, stride);
  
  double minus_one = -1;

  cublasDaxpy(handle, len_of_vector, &minus_one, b, stride, c, stride);
  #####################################
  */
  c[0] = a[0] - b[0];
  c[1] = a[1] - b[1];
  c[2] = a[2] - b[2];
}

/* Stores the component-wise minimum of a and b into out. */
__device__ void elem_min(double *a, double *b, double *out) { 
  out[0] = min(a[0], b[0]);
  out[1] = min(a[1], b[1]);
  out[2] = min(a[2], b[2]);
}

// does gemv from cublas, a and c are vec3, b is a 3x3 matrix
__device__ void gemv(double *a, double *b, double *c){
  //TODO write a cublas version in comments 
  /*
  Assuming B is in column major format
  then a, c are row vectors
  then c is given by a*b, however cublasDgemv needs the multiplication
  to be in terms of b*a, so we need to transpose the b matrix.
  #####################################
  int rows_of_matrix = 3;
  int cols_of_matrix = 3;
  int lead_dim_of_matrix = 3;
  int stride = 1;


  double one = 1, zero = 0;

  cublasDgemv(handle, CUBLAS_OP_T, rows_of_matrix, cols_of_matrix, &one, b, lead_dim_of_matrix, a, stride, &zero, c, stride);
  #####################################
  */
  double a0 = a[0];
  double a1 = a[1];
  double a2 = a[2];

  c[0] = (b[0] * a0) + (b[1] * a1) + (b[2] * a2);
  c[1] = (b[3] * a0) + (b[4] * a1) + (b[5] * a2);
  c[2] = (b[6] * a0) + (b[7] * a1) + (b[8] * a2);
}

/* Returns -1 for negative numbers, 1 for positive numbers, and 0 for zero. */
__device__ int sign(double s) {
  if (s > 0)
    return 1;
  if (s < 0)
    return -1;
  return 0;
}

/* Returns the norm of the given vector. */
__device__ double vec_norm(double *vec) {
  //TODO write a cublas version in comments 

  /*
  #####################################
  int len_of_vector = 3;
  int stride = 1;

  double result = 0;

  cublasDnrm2(handle, len_of_vector, vec, stride, &result);
  return result;
  #####################################
  */

  return sqrt((vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2]));
}

/* Normalizes the given vector by scaling it to the norm. */
__device__ void vec_scale(double *vec) {
  //TODO write a cublas version in comments 
  /*
  #####################################
  int len_of_vector = 3;
  int stride = 1;

  double n = 0;

  //find normalization value
  cublasDnrm2(handle, len_of_vector, vec, stride, &n);
  n = 1.0/n;

  //use scaling function
  cublasDscal(handle, len_of_vector, &n, vec, stride);
  #####################################
  */
  double n = vec_norm(   vec);
  // vec * 1/n hint hint
  vec[0] = vec[0] / n;
  vec[1] = vec[1] / n;
  vec[2] = vec[2] / n;
}

/* Returns the dot product of the given vectors. */
// a and b are vec3
__device__ double vec_dot(double *a, double *b) {
  //TODO write a cublas version in comments   
  /*
  #####################################
  int len_of_vector = 3;
  int stride = 1;

  double result = 0;

  cublasDdot(handle, len_of_vector, a, stride, b, stride, &result);
  return result;
  #####################################
  */
  return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}

/* Implicit Superquadric function. */
// vec is a vec3
__device__ double isq(double *vec, double *e, double *n) {
  // Test for n = 0 now to prevent divide-by-zero errors.
  if (n == 0)
    return FLT_MAX;

  double zTerm = pow(pow(vec[2], 2.0), 1.0 / (double)*n);

  // Test for e = 0 now to prevent divide-by-zero errors.
  if (e == 0)
    return zTerm;

  double xTerm = pow(pow(vec[0], 2.0), 1.0 / (double)*e);
  double yTerm = pow(pow(vec[1], 2.0), 1.0 / (double)*e);
  double xyTerm = pow(xTerm + yTerm, *e / (double)*n);
  return xyTerm + zTerm - 1.0;
}

/* Apply the Inverse Transform to a to get a new, usable a. */
// unScale and unRotate are 3x3 matrices. a and newA are vec3
__device__ void newa(double *unScale, double *unRotate, double *a,
                     double *newA) {
  gemv(a, unRotate, newA);
  gemv(newA, unScale, newA);
}

/* Apply the Inverse Transform to b to get a new, usable b. */
// unScale and unRotate are 3x3 matrices. unTranslate, b, and newB are vec3
__device__ void newb(double *unScale, double *unRotate, double *unTranslate,
                     double *b, double *newB) {
  // b + unTranslate
  elem_add(b, unTranslate, newB);

  // unRotate * (b + unTranslate)
  gemv(newB, unRotate, newB);

  // unScale * (unRotate * (b + unTranslate))
  gemv(newB, unScale, newB);
}

/* Finds the scalar coefficients of the quadratic equation with the two given
 * vectors. If positiveb is true then the returned coeffs will all be multiplied
 * by -1 if b is negative, to ensure that b is positive. */
// a, b, and c are vec3
__device__ void findCoeffs(double *a, double *b, double *c, bool positiveb) {
  c[0] = vec_dot(a, a);
  c[1] = 2 * vec_dot(a, b);
  c[2] = vec_dot(b, b) - 3;

  if (positiveb && c[1] < 0) {
    mult_vec_scalar(c, -1, c);
  }
}

/* Finds the roots of the quadratic with the coefficients specified by the input
 * Vector3d. If one of the roots is complex then FLT_MAX is returned instead. */
// coeffs is a 3-vector, roots is a 2-vector
__device__ void findRoots(double *coeffs, double *roots) {
  double interior = pow(coeffs[1], 2) - (4 * coeffs[0] * coeffs[2]);
  if (interior < 0) {
    roots[0] = FLT_MAX;
    roots[1] = FLT_MAX;
  } else {
    roots[0] = (-coeffs[1] - sqrt(interior)) / (double)(2 * coeffs[0]);
    roots[1] = (2 * coeffs[2]) / (double)(-coeffs[1] - sqrt(interior));
  }
}

/* Gradient of the isq function. */
// vec and grad are vec3
__device__ void isqGradient(double *vec, double *grad, double e, double n) {
  double xval = 0.0, yval = 0.0, zval = 0.0;
  // Check for n = 0 to prevent divide-by-zero errors
  if (n == 0) {
    xval = yval = zval = FLT_MAX;
  }
  // Check for e = 0 to prevent divide-by-zero errors
  else if (e == 0) {
    xval = yval = FLT_MAX;
    zval = (2 * vec[2] * pow(pow(vec[2], 2), ((double)1 / n) - 1)) / (double)n;
  } else {
    double xterm = pow(pow(vec[0], 2.0), (double)1 / e);
    double yterm = pow(pow(vec[1], 2.0), (double)1 / e);
    double xyterm = pow(xterm + yterm, ((double)e / n) - 1);
    double x2term = (2 * vec[0] * pow(pow(vec[0], 2.0), ((double)1 / e) - 1));
    double y2term = (2 * vec[1] * pow(pow(vec[1], 2.0), ((double)1 / e) - 1));
    xval = x2term * xyterm / (double)n;
    yval = y2term * xyterm / (double)n;
    zval =
        (2 * vec[2] * pow(pow(vec[2], 2.0), ((double)1 / n) - 1)) / (double)n;
  }

  grad[0] = xval;
  grad[1] = yval;
  grad[2] = zval;
}

/* Derivative of the isq function. */
// vec and a are 3-vectors
__device__ double gPrime(double *vec, double *a, double e, double n) {
  double tmp[3];
  isqGradient(vec, tmp, e, n);
  double val = vec_dot(   a, tmp);
  return val;
}

/* Uses Newton's method to find the t value at which a ray hits the
 * superquadric.
 * If the ray actually misses the superquadric then FLT_MAX is returned
 * instead.
 * a and b are vec3 */
__device__ double updateRule(double *a, double *b, double *e, double *n,
                             double t, double epsilon) {
  double vec[3];

  axpy(   a, t, b, vec);
  double gP = gPrime(   vec, a, *e, *n);
  double gPPrevious = gP;
  double g = 0.0;
  double tnew = t, told = t;
  bool stopPoint = false;

  while (!stopPoint) {
    told = tnew;
    axpy(   a, told, b, vec);
    gP = gPrime(   vec, a, *e, *n);
    g = isq(vec, e, n);

    if ((g - epsilon) <= 0) {
      stopPoint = true;
    } else if (sign(gP) != sign(gPPrevious) || gP == 0) {
      stopPoint = true;
      tnew = FLT_MAX;
    } else {
      tnew = told - (g / gP);
      gPPrevious = gP;
    }
  }

  return tnew;
}

/* Unit normal vector at a point on the superquadric */
// r is a 3x3 matrix
// vec1, vec2, and un are vec3
__device__ void unitNormal(  double *r, double *vec1, double *vec2, double *un,
                           double tt, double e, double n) {
  axpy(vec1, tt, vec2, un);
  isqGradient(un, un, e, n);
  gemv(   un, r, un);
  vec_scale(   un);
}

// Returns the angle between two vectors.
// Both a and b are vec3.
__device__ double vectorAngle(  double *a, double *b) {
  double d = vec_dot(   a, b);
  double mag = vec_norm(   a) * vec_norm(   b);
  return acos(d / (double)mag);
}

/* debugging purposes */
__device__ void print_objects(Object *p_objects, int numObjects) {
  for (int i = 0; i < numObjects; i++) {
    Object *o = &p_objects[i];
    printf("\nObject %d\n", i);
    printf("e: %f\t n: %f\n", o->e, o->n);
    printf("scale: [%f, %f, %f] unScale: [%f, %f, %f]\n", o->scale[0],
           o->scale[1], o->scale[2], o->unScale[0], o->unScale[1],
           o->unScale[2]);
    printf("       [%f, %f, %f]          [%f, %f, %f]\n", o->scale[3],
           o->scale[4], o->scale[5], o->unScale[3], o->unScale[4],
           o->unScale[5]);
    printf("       [%f, %f, %f]          [%f, %f, %f]\n", o->scale[6],
           o->scale[7], o->scale[8], o->unScale[6], o->unScale[7],
           o->unScale[8]);
    printf("rotate: [%f, %f, %f] unRotate: [%f, %f, %f]\n", o->rotate[0],
           o->rotate[1], o->rotate[2], o->unRotate[0], o->unRotate[1],
           o->unRotate[2]);
    printf("        [%f, %f, %f]           [%f, %f, %f]\n", o->rotate[3],
           o->rotate[4], o->rotate[5], o->unRotate[3], o->unRotate[4],
           o->unRotate[5]);
    printf("        [%f, %f, %f]           [%f, %f, %f]\n", o->rotate[6],
           o->rotate[7], o->rotate[8], o->unRotate[6], o->unRotate[7],
           o->unRotate[8]);
    printf("translate: (%f, %f, %f) unTranslate: (%f, %f, %f)\n",
           o->translate[0], o->translate[1], o->translate[2], o->unTranslate[0],
           o->unTranslate[1], o->unTranslate[2]);
    printf("Material-\n");
    printf("Diffuse: (%f, %f, %f)\n", o->mat.diffuse[0], o->mat.diffuse[1],
           o->mat.diffuse[2]);
    printf("Ambient: (%f, %f, %f)\n", o->mat.ambient[0], o->mat.ambient[1],
           o->mat.ambient[2]);
    printf("Specular: (%f, %f, %f)\n", o->mat.specular[0], o->mat.specular[1],
           o->mat.specular[2]);
    printf("shine: %f\t snell: %f\t opacity: %f\n", o->mat.shine, o->mat.snell,
           o->mat.opacity);
  }
}
__device__ void print_lights(Point_Light *p_lights, int numLights) {
  for (int i = 0; i < numLights; i++) {
    Point_Light *l = &p_lights[i];
    printf("\nLight %d\n", i);
    printf("Position: (%f, %f, %f)\n", l->position[0], l->position[1],
           l->position[2]);
    printf("Color: (%f, %f, %f)\n", l->color[0], l->color[1], l->color[2]);
    printf("Attenuation Factor: %f\n", l->attenuation_k);
  }
}

/********** Actual Raytracing Functions ***************************************/
// n is the normal. e is the eye. ind is the index of the object we're
// lighting.
__device__
void
lighting( double *point, double *n, double *e, Material *mat, Point_Light *l,
          const int numLights, Object *objects, const int numObjects,
          double epsilon, int ind, int generation, double *res,
             double *lightDoubles) {
  double diffuseSum[3] = {0.0, 0.0, 0.0};
  double specularSum[3] = {0.0, 0.0, 0.0};
  double refractedLight[3] = {0.0, 0.0, 0.0};
  double reflectedLight[3] = {0.0, 0.0, 0.0};

  double *dif = &mat->diffuse[0];
  double *spec = &mat->specular[0];
  double shine = mat->shine;

  double *newA = &lightDoubles[0];
  double *newB = &lightDoubles[3];
  double *coeffs = &lightDoubles[6];
  double *roots = &lightDoubles[30];

  // Get the unit direction from the point to the camera
  double eDirection[3];

  elem_sub(e, point, eDirection);

  vec_scale(   eDirection);

  for (int i = 0; i < numLights && generation > 0; i++) {
    // Retrieve the light's postion, color, and attenuation factor
    double attenuation = l[i].attenuation_k;

    // Get the unit direction and the distance between the light and the
    // point
    double lDirection[3];
    elem_sub(l[i].position, point, lDirection);

    double lightDist = vec_norm(   lDirection);
    vec_scale(   lDirection);

    // Check to see that the light isn't blocked before considering it
    // further.
    // The i > 0 condition is present to prevent the program from blocking
    // anything from the eyelight, for the obvious reason that anything we
    // can see will be illuminated by the eyelight.
    bool useLight = true;
    for (int k = 0; k < numObjects && useLight && i > 0; k++) {
      if (k != ind) {
        // Find the ray equation transformations
        newa(   &objects[k].unScale[0], &objects[k].unRotate[0], &lDirection[0],
             &newA[0]);
        newb(   &objects[k].unScale[0], &objects[k].unRotate[0],
             &objects[k].unTranslate[0], point, &newB[0]);

        // Find the quadratic equation coefficients
        findCoeffs(   newA, newB, coeffs, true);
        // Using the coefficients, find the roots
        findRoots(coeffs, roots);

        // Check to see if the roots are FLT_MAX - if they are then the
        // ray missed the superquadric. If they haven't missed then we
        // can continue with the calculations.
        if (roots[0] != FLT_MAX) {
          // Use the update rule to find tfinal
          double tini = min(roots[0], roots[1]);
          double tfinal = updateRule(   newA, newB, &objects[k].e,
                                     &objects[k].n, tini, epsilon);

          double ray[3];
          axpy(   lDirection, tfinal, point, ray);
          double objDist = vec_norm(   ray);
          if (tfinal != FLT_MAX && tfinal >= 0 && objDist < lightDist)
            useLight = false;
        }
      }
    }

    if (useLight) {

      // Find the attenuation term
      double atten = 1 / (double)(1 + (attenuation * pow(lightDist, 2)));
      // Add the attenuation factor to the light's color

      // Add the diffuse factor to the diffuse sum
      double nDotl = vec_dot(   n, lDirection);
      //just a constant we will (re)use here
      double b; 
      
      if (0 < nDotl) {
        b = atten * nDotl;
        axpy(   l[i].color, b, diffuseSum, diffuseSum);
      }

      // Add the specular factor to the specular sum
      double dirDif[3];
      elem_add(eDirection, lDirection, dirDif);
      vec_scale(   dirDif);
      double nDotDir = vec_dot(   n, dirDif);

      if (0 < nDotDir && 0 < nDotl) {
        b = pow(nDotDir, shine) * atten;
        axpy(   l[1].color, b, specularSum, specularSum);
      }
    }
  }

  double *minVec = &lightDoubles[0];
  double *maxVec = &lightDoubles[3];

  set_vec(minVec, 1);

  hadamard_product(diffuseSum, dif, diffuseSum);
  hadamard_product(specularSum, spec, specularSum);

  elem_add(diffuseSum, specularSum, maxVec);
  elem_add(maxVec, reflectedLight, maxVec);
  elem_add(maxVec, refractedLight, maxVec);
  elem_min(minVec, maxVec, res);
}

__global__ void raytraceKernel( 
  double *grid, Object *objects,
  Point_Light *lightsPPM, double *data,
  double *bgColor, double *e1, double *e2,
  double *e3, double *lookFrom, double *rayDoubles,
  double *lightDoubles, int Nx, int Ny,
  bool antiAliased) {

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  // //serialize it for cublas....this sucks
  // int i = 0;
  // int j = 0;


  while (i < Nx) {
    j = threadIdx.y + blockDim.y * blockIdx.y;
    // j = 0;

    while (j < Ny) {
      /* Do all of this within the while loop to prevent threads with i's
      * and j's outside of the image boundaris from accessing rayDoubles.
      */

      double dx = data[2] / (double)Nx;
      double dy = data[3] / (double)Ny;

      double ttrueFinal = 0.0;
      int finalObj = 0;
      bool hitObject = false;

      int rayInd = (j * Nx + i) * 26;
      double *finalNewA = &rayDoubles[rayInd];
      double *finalNewB = &rayDoubles[rayInd + 3];
      double *pointA = &rayDoubles[rayInd + 6];
      double *newA = &rayDoubles[rayInd + 9];
      double *newB = &rayDoubles[rayInd + 12];
      double *coeffs = &rayDoubles[rayInd + 15];
      double *intersect = &rayDoubles[rayInd + 18];
      double *intersectNormal = &rayDoubles[rayInd + 21];
      double *roots = &rayDoubles[rayInd + 24];

      double *lDoubles = &lightDoubles[(j * Nx + i) * 32];

      // The positions are subtracted by a Nx/2 or Ny/2 term to center
      // the film plane
      double px = (i * dx) - (data[2] / (double)2);
      double py = (j * dy) - (data[3] / (double)2);
      double pxColor[3];
      copy_vec(bgColor, pxColor);

      if (!antiAliased) {
        // Transform point A into film coordinates
        axpy(e1, px, vec3zero, pointA);
        axpy(e2, py, pointA, pointA);
        axpy(e3, data[5], pointA, pointA);

        hitObject = false;
        finalObj = 0, ttrueFinal = 0;
        for (int k = 0; k < data[0]; k++) {
          // Find the ray equation transformations
          newa(objects[k].unScale, objects[k].unRotate, pointA, newA);
          newb(objects[k].unScale, objects[k].unRotate, objects[k].unTranslate,
               lookFrom, newB);

          // Find the quadratic equation coefficients
          findCoeffs(newA, newB, coeffs, true);
          // Using the coefficients, find the roots
          findRoots(coeffs, roots);

          // Check to see if the roots are FLT_MAX - if they are then the
          // ray missed the superquadric. If they haven't missed then we
          // can continue with the calculations.
          if (roots[0] != FLT_MAX) {
            // Use the update rule to find tfinal
            double tini = min(roots[0], roots[1]);
            double tfinal = updateRule(newA, newB, &objects[k].e, &objects[k].n,
                                       tini, data[4]);

            /* Check to see if tfinal is FLT_MAX - if it is then the ray
            * missed the superquadric. Additionally, if tfinal is negative
            * then either the ray has started inside the object or is
            * pointing away from the object; in both cases the ray has
            * "missed". */
            if (tfinal != FLT_MAX && tfinal >= 0) {
              if (hitObject && tfinal < ttrueFinal) {
                ttrueFinal = tfinal;
                finalObj = k;
                copy_vec(newA, finalNewA);
                copy_vec(newB, finalNewB);
              } else if (!hitObject) {
                hitObject = true;
                ttrueFinal = tfinal;
                finalObj = k;
                copy_vec(newA, finalNewA);
                copy_vec(newB, finalNewB);
              }
            }
          }
        }
        if (hitObject) {
          axpy(pointA, ttrueFinal, lookFrom, intersect);
          unitNormal(objects[finalObj].rotate, finalNewA, finalNewB,
                     intersectNormal, ttrueFinal, objects[finalObj].e,
                     objects[finalObj].n);

          lighting(intersect, intersectNormal, lookFrom, &objects[finalObj].mat,
                   lightsPPM, data[1], objects, data[0], data[4], finalObj,
                   RECURSIONDEPTH, pxColor, lDoubles);
        }
      } else {
        double denom = 3 + (2 / sqrt((double)2));
        double pxCoeffs[] = {(1 / (2 * sqrt((double)2))) / denom,
                             (1 / (double)2) / denom,
                             (1 / (2 * sqrt((double)2))) / denom,
                             (1 / (double)2) / denom,
                             1 / denom,
                             (1 / (double)2) / denom,
                             (1 / (2 * sqrt((double)2))) / denom,
                             (1 / (double)2) / denom,
                             (1 / (2 * sqrt((double)2))) / denom};
        int counter = 0;
        for (int g = -1; g <= 1; g++) {
          for (int h = -1; h <= 1; h++) {
            double thisPx = px + (g * (dx / (double)2));
            double thisPy = py + (h * (dy / (double)2));

            // Transform point A into film Coordinates
            axpy(  e1, thisPx, vec3zero, pointA);
            axpy(  e2, thisPy, pointA, pointA);
            axpy(  e3, data[5], pointA, pointA);

            hitObject = false;
            finalObj = 0, ttrueFinal = 0;
            for (int k = 0; k < data[0]; k++) {
              // Find the ray equation transformations
              newa(  objects[k].unScale, objects[k].unRotate, pointA, newA);
              newb(  objects[k].unScale, objects[k].unRotate,
                   objects[k].unTranslate, lookFrom, newB);

              // Find the quadratic equation coefficients
              findCoeffs(  newA, newB, coeffs, true);
              // Using the coefficients, find the roots
              findRoots(coeffs, roots);

              // Check to see if the roots are FLT_MAX - if they are then the
              // ray missed the superquadric. If they haven't missed then we
              // can continue with the calculations.
              if (roots[0] != FLT_MAX) {
                // Use the update rule to find tfinal
                double tini = min(roots[0], roots[1]);
                double tfinal = updateRule(  newA, newB, &objects[k].e,
                                           &objects[k].n, tini, data[4]);

                /* Check to see if tfinal is FLT_MAX - if it is then the ray
                * missed the superquadric. Additionally, if tfinal is negative
                * then either the ray has started inside the object or is
                * pointing away from the object; in both cases the ray has
                * "missed". */
                if (tfinal != FLT_MAX && tfinal >= 0) {
                  if (hitObject && tfinal < ttrueFinal) {
                    ttrueFinal = tfinal;
                    finalObj = k;
                    copy_vec(newA, finalNewA);
                    copy_vec(newB, finalNewB);
                  } else if (!hitObject) {
                    hitObject = true;
                    ttrueFinal = tfinal;
                    finalObj = k;
                    copy_vec(newA, finalNewA);
                    copy_vec(newB, finalNewB);
                  }
                }
              }
            }
            if (hitObject) {
              axpy(  pointA, ttrueFinal, lookFrom, intersect);
              unitNormal(  objects[finalObj].rotate, finalNewA, finalNewB,
                         intersectNormal, ttrueFinal, objects[finalObj].e,
                         objects[finalObj].n);

              double color[] = {0, 0, 0};

              lighting(  intersect, intersectNormal, lookFrom,
                       &objects[finalObj].mat, lightsPPM, data[1], objects,
                       data[0], data[4], finalObj, RECURSIONDEPTH, color,
                       lDoubles);
              axpy(  color, pxCoeffs[counter], pxColor, pxColor);
            }
            counter++;
          }
        }
      }
      int index = (j * Nx + i) * 3;
      grid[index] = pxColor[0];
      grid[index + 1] = pxColor[1];
      grid[index + 2] = pxColor[2];

      j += blockDim.y * gridDim.y;
      // j += 1;
    }
    i += blockDim.x * gridDim.x;
    // i += 1;
  }
}

void callRaytraceKernel(double *grid, Object *objects, Point_Light *lightsPPM,
                        double *data, double *bgColor, double *e1, double *e2,
                        double *e3, double *lookFrom, int Nx, int Ny,
                        bool antiAliased, int blockPower) {

  int blockSize = pow(2, blockPower);

  dim3 blocks;
  blocks.x = blockSize;
  blocks.y = blockSize;

  int gx = (Nx / blockSize);
  int gy = (Ny / blockSize);
  if (gx < 1)
    gx = 1;
  if (gy < 1)
    gy = 1;
  dim3 gridSize;
  gridSize.x = gx;
  gridSize.y = gy;

  double vec3zero_host[3] = {0,0,0};
  gpuErrChk(cudaMemcpyToSymbol(vec3zero, vec3zero_host, 3*sizeof(double)));

  // Mostly debug info, but possibly interesting
  int numThreads = (blockSize * gx) * (blockSize * gy);
  printf("Image size: %d x %d (%d Pixels)\n", Nx, Ny, Nx * Ny);
  printf("Total number of threads: %d\n", (blockSize * gx) * (blockSize * gy));

  float factor = numThreads / (float)(1024 * 1024);
  size_t deviceLimit;
  gpuErrChk(cudaDeviceGetLimit(&deviceLimit, cudaLimitStackSize));
  printf("Original Device stack size: %d\n", (int)deviceLimit);
  printf("Total Device stack memory: %0.2f MB\n", (int)deviceLimit * factor);

  // increase the stack size due to recursion
  // (Also relevant for images larger than 400 x 400 or so, I suppose)
  gpuErrChk(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
  gpuErrChk(cudaDeviceGetLimit(&deviceLimit, cudaLimitStackSize));
  printf("New Device stack size: %d\n", (int)deviceLimit);
  printf("Total Device stack memory: %0.2f MB\n", (int)deviceLimit * factor);

  double *rayDoubles;
  gpuErrChk(cudaMalloc(&rayDoubles, sizeof(double) * Nx * Ny * 26));
  gpuErrChk(cudaMemset(rayDoubles, 0, sizeof(double) * Nx * Ny * 26));

  double *lightDoubles;
  gpuErrChk(cudaMalloc(&lightDoubles, sizeof(double) * Nx * Ny * 32));
  gpuErrChk(cudaMemset(lightDoubles, 0, sizeof(double) * Nx * Ny * 32));

  raytraceKernel<<<gridSize, blocks>>>(grid, objects, lightsPPM, data, bgColor,
                                       e1, e2, e3, lookFrom, rayDoubles,
                                       lightDoubles, Nx, Ny, antiAliased);
  
  gpuErrChk(cudaPeekAtLastError());
  gpuErrChk(cudaDeviceSynchronize());
  
  gpuErrChk(cudaFree(rayDoubles));
  gpuErrChk(cudaFree(lightDoubles));
}
