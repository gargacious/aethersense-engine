#include <aethersense/engine.hpp>
#include <aethersense/fft_plan.hpp>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <numbers>
#include <thread>
std::atomic<bool> count_allocations{false};
std::atomic<size_t> allocations{0};
void *operator new(std::size_t size) {
  if(count_allocations.load(std::memory_order_relaxed))allocations.fetch_add(1,std::memory_order_relaxed);
  if(auto pointer=std::malloc(size?size:1))return pointer;throw std::bad_alloc();
}
void *operator new[](std::size_t size){return ::operator new(size);}
void operator delete(void *pointer) noexcept{std::free(pointer);}
void operator delete[](void *pointer) noexcept{std::free(pointer);}
void operator delete(void *pointer,std::size_t) noexcept{std::free(pointer);}
void operator delete[](void *pointer,std::size_t) noexcept{std::free(pointer);}
void *operator new(std::size_t size,std::align_val_t alignment) {
  if(count_allocations.load(std::memory_order_relaxed))allocations.fetch_add(1,std::memory_order_relaxed);
  void *pointer=nullptr;
  if(posix_memalign(&pointer,static_cast<size_t>(alignment),size?size:1))throw std::bad_alloc();
  return pointer;
}
void *operator new[](std::size_t size,std::align_val_t alignment){return ::operator new(size,alignment);}
void operator delete(void *pointer,std::align_val_t) noexcept{std::free(pointer);}
void operator delete[](void *pointer,std::align_val_t) noexcept{std::free(pointer);}
void operator delete(void *pointer,std::size_t,std::align_val_t) noexcept{std::free(pointer);}
void operator delete[](void *pointer,std::size_t,std::align_val_t) noexcept{std::free(pointer);}
namespace {
using Complex=std::complex<double>;
void check(bool valid,const char *message){if(!valid)throw std::runtime_error(message);}
template<class Function>void rejects(Function function) {
  bool rejected=false;try{function();}catch(const std::exception &){rejected=true;}
  check(rejected,"Invalid FFT plan/call was accepted");
}
std::vector<Complex> input(size_t size) {
  std::vector<Complex> output(size);
  for(size_t index=0;index<size;++index)
    output[index]={std::sin(.173*index)+.125*std::cos(.931*index),std::cos(.217*index)-.2*std::sin(.337*index)};
  return output;
}
std::vector<Complex> direct_dft(const std::vector<Complex> &values,bool inverse) {
  std::vector<Complex> output(values.size());
  for(size_t frequency=0;frequency<values.size();++frequency) {
    std::complex<long double> sum{};
    for(size_t sample=0;sample<values.size();++sample) {
      const long double angle=(inverse?2.L:-2.L)*std::numbers::pi_v<long double>*frequency*sample/values.size();
      sum+=std::complex<long double>(values[sample].real(),values[sample].imag())*
           std::complex<long double>(std::cos(angle),std::sin(angle));
    }
    if(inverse)sum/=static_cast<long double>(values.size());
    output[frequency]={static_cast<double>(sum.real()),static_cast<double>(sum.imag())};
  }
  return output;
}
double error(std::span<const Complex> actual,std::span<const Complex> expected,double absolute,double relative) {
  check(actual.size()==expected.size(),"Output length");double maximum=0;
  for(size_t index=0;index<actual.size();++index) {
    const double difference=std::abs(actual[index]-expected[index]);maximum=std::max(maximum,difference);
    check(difference<=absolute+relative*std::abs(expected[index]),"FFT numerical tolerance exceeded");
  }
  return maximum;
}
}
int main() {
  try {
    double oracle_error=0,reference_error=0,roundtrip_error=0;
    for(size_t size:{size_t(1),size_t(2),size_t(4),size_t(8),size_t(16),size_t(64)})for(bool inverse:{false,true}) {
      auto values=input(size);auto expected=direct_dft(values,inverse);
      aether::Fft64Plan plan(size);plan.execute(values,inverse);
      oracle_error=std::max(oracle_error,error(values,expected,2e-11,2e-11));
    }
    for(size_t size:{size_t(1),size_t(32),size_t(128),size_t(1024),size_t(65536)}) {
      auto values=input(size),original=values,reference=values;
      aether::Fft64Plan plan(size);plan.execute(values);aether::fft(reference);
      reference_error=std::max(reference_error,error(values,reference,2e-8,1e-10));
      plan.execute(values,true);
      roundtrip_error=std::max(roundtrip_error,error(values,original,2e-11,2e-11));
    }
    // Exact analytical impulse and signed-bin complex tone independently fix
    // normalization/sign convention, beyond inverse/forward roundtrip agreement.
    aether::Fft64Plan analytical(64);std::vector<Complex> impulse(64);impulse[0]={1,0};
    analytical.execute(impulse);for(auto value:impulse)check(std::abs(value-Complex(1,0))<1e-14,"Impulse response");
    std::vector<Complex> tone(64);
    for(size_t i=0;i<tone.size();++i)tone[i]=std::polar(1.,2*std::numbers::pi*7*i/64.);
    analytical.execute(tone);for(size_t bin=0;bin<tone.size();++bin)
      check(std::abs(tone[bin]-(bin==7?Complex(64,0):Complex{}))<1e-11,"Signed tone bin/amplitude");
    aether::Fft64Plan batch(16,3);auto values=input(48),reference=values;
    for(size_t index=0;index<3;++index)aether::fft(std::span(reference).subspan(index*16,16));
    batch.execute_batch(values,3);error(values,reference,2e-11,2e-11);
    for(size_t invalid:{size_t(0),size_t(3),size_t(65537),std::numeric_limits<size_t>::max()})
      rejects([&]{aether::Fft64Plan plan(invalid);});
    rejects([]{aether::Fft64Plan plan(8,0);});rejects([]{aether::Fft64Plan plan(8,257);});
    rejects([]{aether::Fft64Plan plan(65536,17);});
    aether::Fft64Plan maximum(65536,16);check(maximum.max_batches()==16,"Maximum scratch capacity");
    aether::Fft64Plan identities(1,256);auto identity_values=input(256),identity_original=identity_values;
    identities.execute_batch(identity_values,256);identities.execute_batch(identity_values,256,true);
    check(identity_values==identity_original,"N1 maximum batch boundary");
    aether::Fft64Plan atomic(8,2);auto good=input(16),original=good;
    rejects([&]{atomic.execute_batch(good,0);});check(good==original,"Shape failure changed input");
    rejects([&]{atomic.execute_batch(std::span(good).first(15),2);});check(good==original,"Length failure changed input");
    rejects([&]{atomic.execute_batch(good,3);});check(good==original,"Capacity failure changed input");
    for(double invalid:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}) {
      auto bad=original;bad[15]={invalid,0};auto saved=bad;
      rejects([&]{atomic.execute_batch(bad,2);});
      check(std::memcmp(bad.data(),saved.data(),bad.size()*sizeof(Complex))==0,"Nonfinite failure changed any batch");
    }
    auto overflow=original;for(size_t i=8;i<16;++i)overflow[i]={std::numeric_limits<double>::max(),0};
    auto saved=overflow;rejects([&]{atomic.execute_batch(overflow,2);});check(overflow==saved,"Overflow changed earlier valid batch");
    good=original;atomic.execute_batch(good,2);atomic.execute_batch(good,2,true);error(good,original,2e-11,2e-11);
    auto moved=std::move(atomic);rejects([&]{atomic.execute_batch(good,2);});moved.execute_batch(good,2);
    aether::Fft64Plan hot(64,8);auto hot_values=input(512);hot.execute_batch(hot_values,8);hot.execute_batch(hot_values,8,true);
    allocations=0;count_allocations=true;
    for(int repeat=0;repeat<20;++repeat){hot.execute_batch(hot_values,8);hot.execute_batch(hot_values,8,true);}
    count_allocations=false;check(allocations==0,"Warmed FFT execute allocated heap memory");
    std::cout<<"PASS FFT64 independent DFT/sign/normalization, reference/max-size parity, batch atomicity, bounded plan, zero warmed allocations\n"
             <<"max_dft_error="<<oracle_error<<" max_reference_error="<<reference_error
             <<" max_roundtrip_error="<<roundtrip_error<<" warmed_allocations="<<allocations<<'\n';
    return 0;
  }catch(const std::exception &failure){count_allocations=false;std::cerr<<"FAIL "<<failure.what()<<'\n';return 1;}
}
