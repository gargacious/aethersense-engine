#include <aethersense/fft_plan.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace aether {
Fft64Plan::Fft64Plan(size_t size,size_t max_batches):size_(size),max_batches_(max_batches) {
  if(!size || size>maximum_size || (size&(size-1)) || !max_batches ||
     max_batches>maximum_batches || size>maximum_elements/max_batches)
    throw std::invalid_argument("FFT plan requires power-of-two1..65536, batches1..256, <=1048576 elements");
  reversal_.resize(size);twiddles_.resize(size/2);scratch_.resize(size*max_batches);
  size_t levels=0;for(size_t value=size;value>1;value>>=1)++levels;
  for(size_t index=0;index<size;++index) {
    size_t value=index,reversed=0;
    for(size_t bit=0;bit<levels;++bit){reversed=(reversed<<1)|(value&1);value>>=1;}
    reversal_[index]=reversed;
  }
  for(size_t index=0;index<size/2;++index) {
    const double angle=-2.*std::numbers::pi*static_cast<double>(index)/static_cast<double>(size);
    twiddles_[index]={std::cos(angle),std::sin(angle)};
  }
}
size_t Fft64Plan::storage_bytes() const noexcept {
  return reversal_.size()*sizeof(size_t)+(twiddles_.size()+scratch_.size())*sizeof(std::complex<double>);
}
void Fft64Plan::execute(std::span<std::complex<double>> values,bool inverse) {
  execute_batch(values,1,inverse);
}
void Fft64Plan::execute_batch(std::span<std::complex<double>> values,size_t batches,bool inverse) {
  // Also rejects moved-from plans before indexing moved vectors. Plan dimensions
  // themselves remain bounded, but a moved-from object's buffers are unspecified.
  if(!batches || batches>max_batches_ || values.size()!=size_*batches ||
     reversal_.size()!=size_ || scratch_.size()!=size_*max_batches_)
    throw std::invalid_argument("FFT plan input shape/batch capacity mismatch");
  for(const auto &value:values)
    if(!std::isfinite(value.real()) || !std::isfinite(value.imag()))
      throw std::invalid_argument("FFT plan input must be finite");
  for(size_t batch=0;batch<batches;++batch) {
    auto work=std::span(scratch_).subspan(batch*size_,size_);
    auto input=values.subspan(batch*size_,size_);
    for(size_t index=0;index<size_;++index)work[reversal_[index]]=input[index];
    for(size_t length=2;length<=size_;length<<=1) {
      const size_t stride=size_/length;
      for(size_t offset=0;offset<size_;offset+=length)
        for(size_t index=0;index<length/2;++index) {
          const auto twiddle=inverse ? std::conj(twiddles_[index*stride]) : twiddles_[index*stride];
          const auto even=work[offset+index],odd=work[offset+index+length/2]*twiddle;
          work[offset+index]=even+odd;work[offset+index+length/2]=even-odd;
        }
    }
    if(inverse)for(auto &value:work)value/=static_cast<double>(size_);
    for(const auto &value:work)
      if(!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::overflow_error("FFT plan result is not finite; input unchanged");
  }
  std::copy_n(scratch_.begin(),values.size(),values.begin());
}
} // namespace aether
