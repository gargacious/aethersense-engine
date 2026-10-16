#include <aethersense/compute_backend.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
using namespace aether;
void check(bool v,const char*m){if(!v)throw std::runtime_error(m);}
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}check(caught,"Expected SIMD rejection");}
int main(){try{
 if(!native_simd_available()){rejects([]{make_compute_backend("native_simd");});rejects([]{NativeSimdBackend backend;});check(native_simd_instruction_set()=="unsupported","Unsupported ISA falsely advertised");std::cout<<"PASS native_simd unavailable branch rejects explicit selection\n";return 0;}
 auto simd=make_compute_backend("native_simd"),reference=make_compute_backend("reference_cpu");
 check(simd->name()=="native_simd"&&simd->supports(ComputePrecision::float64)&&!simd->supports(ComputePrecision::float32)&&!simd->supports(static_cast<ComputePrecision>(999)),"SIMD precision capability");
 auto caps=simd->capabilities();check(caps.cpu_only&&caps.fft_float64&&!caps.fft_float32&&!caps.cfar_float64&&caps.max_fft_samples==65536&&caps.max_cfar_cells==0,"SIMD capability bounds");
 std::mt19937 random(20261016);double max_reference_error=0,max_inverse_error=0,max_direct_error=0;
 for(size_t n=1;n<=65536;n*=2){std::vector<std::complex<double>> input(n);for(auto &v:input)v={static_cast<double>(random())/4294967296.-.5,static_cast<double>(random())/4294967296.-.5};
  auto a=input,b=input;simd->fft64(a);reference->fft64(b);
  for(size_t i=0;i<n;++i){double err=std::abs(a[i]-b[i]);max_reference_error=std::max(max_reference_error,err);check(err<1e-8*std::max(1.,std::abs(b[i])),"SIMD/reference FFT mismatch");}
  if(n<=64)for(size_t k=0;k<n;++k){std::complex<double> expected{};for(size_t j=0;j<n;++j)expected+=input[j]*std::polar(1.,-2*std::numbers::pi*k*j/n);auto err=std::abs(a[k]-expected);max_direct_error=std::max(max_direct_error,err);check(err<1e-10,"SIMD/direct DFT mismatch");}
  simd->fft64(a,true);for(size_t i=0;i<n;++i){auto err=std::abs(a[i]-input[i]);max_inverse_error=std::max(max_inverse_error,err);check(err<1e-9,"SIMD inverse roundtrip");}
 }
 std::vector<std::complex<double>> tone(128);for(size_t i=0;i<tone.size();++i)tone[i]=std::polar(1.,2*std::numbers::pi*13*i/tone.size());simd->fft64(tone);check(std::abs(tone[13]-std::complex<double>(128,0))<1e-10,"SIMD analytic bin");
 std::vector<std::complex<float>> f(32);auto f_before=f;rejects([&]{simd->fft32(f);});check(f==f_before,"Unsupported float32 modified input");rejects([&]{simd->ca_cfar_2d(std::vector<double>(32,1),4,8,{});});
 for(size_t length:{size_t{0},size_t{3},size_t{131072}}){std::vector<std::complex<double>> bad(length);rejects([&]{simd->fft64(bad);});}
 std::vector<std::complex<double>> bad{{INFINITY,0}};rejects([&]{simd->fft64(bad);});bad[0]={NAN,0};rejects([&]{simd->fft64(bad);});
 bad={{std::numeric_limits<double>::max(),0},{std::numeric_limits<double>::max(),0}};auto saved=bad;rejects([&]{simd->fft64(bad);});check(bad==saved,"SIMD overflow modified caller input");
 std::cout<<"PASS actual SIMD FFT ISA="<<native_simd_instruction_set()<<" allpower2lengths1..65536; reference_max_error="<<max_reference_error<<" inverse_max_error="<<max_inverse_error<<" direct_DFT_max_error="<<max_direct_error<<"; finite/bounds/atomicity/capabilities\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
