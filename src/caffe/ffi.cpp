extern "C" {
#include "caffe/ffi.hpp"
}

#include <boost/make_shared.hpp>

#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/upgrade_proto.hpp"



struct CaffeNet {
  boost::shared_ptr<caffe::Net<float> > net_;
};

struct CaffeBlob {
  boost::shared_ptr<caffe::Blob<float> > blob_;
};

struct CaffeSolver {
  boost::shared_ptr<caffe::Solver<float> > solver_;
};

CaffeNet* CaffeNetInit(const char* param_file, CaffePhase phase) {
  CaffeNet* n = new CaffeNet();
  n->net_ = boost::make_shared<caffe::Net<float> >(
      std::string(param_file),
      phase == CAFFE_TEST ? caffe::TEST : caffe::TRAIN);
  return n;
}

void CaffeNetCopyTrainedLayersFrom(CaffeNet* net, const char* trained_file) {
  return net->net_->CopyTrainedLayersFrom(std::string(trained_file));
}

CaffeBlob* CaffeNetGetBlobByName(CaffeNet* net, const char* name) {
  CaffeBlob* b = new CaffeBlob();
  b->blob_ = net->net_->blob_by_name(std::string(name));
  CHECK(b->blob_);
  return b;
}


unsigned long long CaffeBlobCount(const CaffeBlob* blob) {  // NOLINT
  CHECK(blob->blob_);
  return blob->blob_->count();
}

void CaffeNetForwardPrefilled(CaffeNet* net) { net->net_->ForwardPrefilled(); }
void CaffeNetBackward(CaffeNet* net) { net->net_->Backward(); }
const float* CaffeBlobCPUData(const CaffeBlob* blob) {
  return blob->blob_->cpu_data();
}
float* CaffeBlobMutableCPUData(CaffeBlob* blob) {
  return blob->blob_->mutable_cpu_data();
}
void CaffeBlobSetCPUData(CaffeBlob* blob, float* data) {
  return blob->blob_->set_cpu_data(data);
}

void CaffeBlobFree(CaffeBlob* blob) { delete blob; }
void CaffeNetFree(CaffeNet* net) { delete net; }

CaffeSolver* CaffeSolverInit(const char* param_file) {
  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(param_file, &solver_param);

  CaffeSolver* s = new CaffeSolver();
  s->solver_ = boost::shared_ptr<caffe::Solver<float> >(
      caffe::SolverRegistry<float>::CreateSolver(solver_param));
  return s;
}

CaffeNet* CaffeSolverNet(CaffeSolver* solver) {
  CaffeNet* n = new CaffeNet();
  n->net_ = solver->solver_->net();
  return n;
}

void CaffeSolverSolve(CaffeSolver* solver) { solver->solver_->Solve(); }
void CaffeSolverFree(CaffeSolver* solver) { delete solver; }

void CaffeSetMode(CaffeMode mode) {
  caffe::Caffe::set_mode(mode == CAFFE_CPU ? caffe::Caffe::CPU
                                           : caffe::Caffe::GPU);
}
