#include <stddef.h>

enum CaffePhase {
  CAFFE_TRAIN,
  CAFFE_TEST
};

enum CaffeMode {
  CAFFE_CPU,
  CAFFE_GPU,
};

struct CaffeNet;
struct CaffeBlob;
struct CaffeSolver;

CaffeNet* CaffeNetInit(const char* param_file, CaffePhase phase);
void CaffeNetCopyTrainedLayersFrom(CaffeNet* net, const char* trained_file);
void CaffeNetForwardPrefilled(CaffeNet* net);
CaffeBlob* CaffeNetGetBlobByName(CaffeNet* net, const char* name);
void CaffeNetBackward(CaffeNet* net);
void CaffeNetFree(CaffeNet* net);

const float* CaffeBlobCPUData(const CaffeBlob* blob);
float* CaffeBlobMutableCPUData(CaffeBlob* blob);

unsigned long long CaffeBlobCount(const CaffeBlob* blob);  // NOLINT
void CaffeBlobSetCPUData(CaffeBlob* blob, float* data);
void CaffeBlobFree(CaffeBlob* blob);

CaffeSolver* CaffeSolverInit(const char* param_file);
CaffeNet* CaffeSolverNet(CaffeSolver* solver);
void CaffeSolverSolve(CaffeSolver* solver);
void CaffeSolverFree(CaffeSolver* solver);

// Mode
void CaffeSetMode(CaffeMode mode);
