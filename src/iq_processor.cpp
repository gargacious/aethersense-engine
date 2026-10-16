#include <aethersense/iq_processor.hpp>
#include <aethersense/compute_backend.hpp>
#include <aethersense/fft_plan.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>
namespace aether {
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::invalid_argument(message); }
void keys(const json &j, std::initializer_list<const char *> fields) {
  require(j.is_object() && j.size() == fields.size(), "IQ processing requires complete fields");
  for (auto key : fields) require(j.contains(key), "Missing/unknown IQ processing field");
}
int integer(const json &j, const char *key, int low, int high) {
  const auto &v = j.at(key);
  require(v.is_number_integer() && v.get<double>() >= low && v.get<double>() <= high,
          "IQ processing integer outside bounds");
  return v.get<int>();
}
template<class T> void reference_fft(std::span<std::complex<T>> row) {
  if constexpr (std::is_same_v<T,double>) { fft(row); }
  else {
    const size_t n = row.size();
    for (size_t i=1,j=0;i<n;++i) {
      size_t bit=n>>1;
      for (;j&bit;bit>>=1) j^=bit;
      j^=bit;
      if (i<j) std::swap(row[i],row[j]);
    }
    for (size_t len=2;len<=n;len<<=1) {
      const auto root=std::polar(T{1}, -T{2}*std::numbers::pi_v<T>/static_cast<T>(len));
      for (size_t i=0;i<n;i+=len) {
        std::complex<T> w{1,0};
        for (size_t j=0;j<len/2;++j) {
          const auto a=row[i+j], b=row[i+j+len/2]*w;
          row[i+j]=a+b; row[i+j+len/2]=a-b; w*=root;
        }
      }
    }
  }
}
// Scale mantissas separately from exponents: long double can equal double on
// some platforms, so merely widening an overflowing intermediate is insufficient.
// Retain the legacy expression when valid to preserve exact default arithmetic.
double calibrated_scale(double a, double b, double denominator, double factor,
                        double legacy) {
  int ea{}, eb{}, ed{}, ef{}, em{};
  const auto ma=std::frexp(static_cast<long double>(a),&ea);
  const auto mb=std::frexp(static_cast<long double>(b),&eb);
  const auto md=std::frexp(static_cast<long double>(denominator),&ed);
  const auto mf=std::frexp(static_cast<long double>(factor),&ef);
  const auto mantissa=std::frexp((ma*mb)/(md*mf),&em);
  const int exponent=ea+eb-ed-ef+em;
  int minimum_exponent{}, maximum_exponent{};
  std::frexp(static_cast<long double>(std::numeric_limits<double>::denorm_min()),&minimum_exponent);
  const auto maximum_mantissa=std::frexp(
      static_cast<long double>(std::numeric_limits<double>::max()),&maximum_exponent);
  require(exponent>=minimum_exponent &&
          (exponent<maximum_exponent ||
           (exponent==maximum_exponent && mantissa<=maximum_mantissa)),
          "Derived IQ calibration scale outside positive finite float64 range");
  const auto stable=std::scalbn(mantissa,exponent);
  const auto converted=static_cast<double>(stable);
  require(std::isfinite(stable) && stable>0 && std::isfinite(converted) && converted>0,
          "Derived IQ calibration scale outside positive finite float64 range");
  return std::isfinite(legacy) && legacy>0 ? legacy : converted;
}
template<class T> bool finite(std::complex<T> x) {
  return std::isfinite(x.real()) && std::isfinite(x.imag());
}
} // namespace
void IQProcessingConfig::validate(int samples, int chirps) const {
  require(version==1, "Unsupported IQ processing version");
  require(precision=="float32" || precision=="float64", "Unknown IQ precision");
  require(fft_backend=="reference_cpu" || fft_backend=="native_simd" ||
          fft_backend=="preallocated_cpu", "Unknown IQ FFT backend");
  require(cfar_mode=="range_1d" || cfar_mode=="range_doppler_2d", "Unknown IQ CFAR mode");
  require(precision=="float64" || fft_backend=="reference_cpu",
          "Float32 supports reference_cpu only; no implicit fallback");
  require(fft_backend!="native_simd" || native_simd_available(), "Native SIMD unavailable");
  require(training_rows>=1 && training_rows<=31 && training_columns>=1 && training_columns<=511 &&
          guard_rows>=0 && guard_rows<=31 && guard_columns>=0 && guard_columns<=511 &&
          2*(training_rows+guard_rows)+1<=chirps &&
          2*(training_columns+guard_columns)+1<=samples &&
          std::isfinite(pfa) && pfa>=1e-6 && pfa<=.1, "Invalid IQ 2D CFAR box/Pfa");
}
json IQProcessingConfig::to_json() const {
  return {{"schema_version",version},{"precision",precision},{"fft_backend",fft_backend},
          {"cfar_mode",cfar_mode},{"cfar_2d",{{"training_rows",training_rows},
          {"training_columns",training_columns},{"guard_rows",guard_rows},
          {"guard_columns",guard_columns},{"pfa",pfa}}}};
}
IQProcessingConfig IQProcessingConfig::from_json(const json &j) {
  keys(j,{"schema_version","precision","fft_backend","cfar_mode","cfar_2d"});
  IQProcessingConfig c;
  c.version=integer(j,"schema_version",1,1);
  require(j.at("precision").is_string() && j.at("fft_backend").is_string() &&
          j.at("cfar_mode").is_string(), "IQ processing selections must be strings");
  c.precision=j.at("precision"); c.fft_backend=j.at("fft_backend"); c.cfar_mode=j.at("cfar_mode");
  const auto &box=j.at("cfar_2d");
  keys(box,{"training_rows","training_columns","guard_rows","guard_columns","pfa"});
  c.training_rows=integer(box,"training_rows",1,31);
  c.training_columns=integer(box,"training_columns",1,511);
  c.guard_rows=integer(box,"guard_rows",0,31);
  c.guard_columns=integer(box,"guard_columns",0,511);
  require(box.at("pfa").is_number(),"IQ Pfa must be numeric"); c.pfa=box.at("pfa");
  return c;
}
json SignalProcessing::to_json() const {
  return {{"schema_version",1},{"precision",precision},{"fft_backend",fft_backend},
          {"cfar_mode",cfar_mode},{"cfar_accumulation_precision",cfar_accumulation_precision},
          {"geometry_precision",geometry_precision},{"output_storage",output_storage},
          {"threshold_power_role",threshold_power_role},
          {"fft_temporary_allocation",fft_temporary_allocation},
          {"native_instruction_set",native_instruction_set}};
}
struct IQProcessor::Impl {
  const Config config;
  const int n,m;
  const bool fp32, two_d, planned;
  SignalResult staging;
  std::vector<std::complex<double>> values64, doppler64;
  std::vector<std::complex<float>> values32, doppler32;
  std::vector<long double> prefix;
  std::unique_ptr<Fft64Plan> range_plan,doppler_plan;
  std::unique_ptr<NativeSimdBackend> native;
  explicit Impl(const Config &c):config(c),n(c.fft_size),m(c.chirps),
      fp32(c.iq_processing && c.iq_processing->precision=="float32"),
      two_d(c.iq_processing && c.iq_processing->cfar_mode=="range_doppler_2d"),
      planned(c.iq_processing && c.iq_processing->fft_backend=="preallocated_cpu") {
    config.validate();
    if (fp32) {values32.resize(2*n*m);doppler32.resize(2*n*m);}
    else {values64.resize(2*n*m);doppler64.resize(2*n*m);}
    if (two_d) prefix.resize((n+1)*(m+1));
    if (planned) {
      range_plan=std::make_unique<Fft64Plan>(n,2*m);
      doppler_plan=std::make_unique<Fft64Plan>(m,std::min(n,256));
    }
    if (c.iq_processing && c.iq_processing->fft_backend=="native_simd")
      native=std::make_unique<NativeSimdBackend>();
    prepare(staging);
  }
  void prepare(SignalResult &r) const {
    r.bins=n; r.chirps=m;
    r.i.resize(n); r.q.resize(n); r.spectrum.resize(n); r.phase.resize(n);
    r.range_profile.resize(n); r.threshold.resize(n); r.range_doppler.resize(n*m);
    r.detections.reserve(128);
    if (two_d) r.range_doppler_threshold.resize(n*m);
    if (config.iq_processing) {
      const auto &c=*config.iq_processing;
      SignalProcessing p;
      p.precision=c.precision;p.fft_backend=c.fft_backend;p.cfar_mode=c.cfar_mode;
      p.cfar_accumulation_precision=two_d?"long_double_platform":"float64";
      p.threshold_power_role=two_d?"range_1d_diagnostic_only":"range_1d_detection";
      p.fft_temporary_allocation=planned?"precomputed_plan":native?"backend_per_call_scratch":"inplace_reused";
      p.native_instruction_set=native?std::string(native_simd_instruction_set()):"none";
      r.processing=std::move(p);
    }
  }
  template<class T> void transform(std::span<std::complex<T>> row) {
    if constexpr (std::is_same_v<T,double>) {
      if (native) { native->fft64(row); return; }
    }
    reference_fft(row);
  }
  void threshold_1d() {
    auto &r=staging;
    std::fill(r.threshold.begin(),r.threshold.end(),-1.);
    const int t=config.cfar_training,g=config.cfar_guard;
    const double alpha=2*t*(std::pow(config.cfar_pfa,-1./(2*t))-1);
    for (int i=t+g;i<n-t-g;++i) {
      double sum=0;
      for (int k=g+1;k<=g+t;++k) sum+=r.range_profile[i-k]+r.range_profile[i+k];
      r.threshold[i]=alpha*sum/(2*t);
    }
  }
  void threshold_2d() {
    const auto &c=*config.iq_processing;
    const int ry=c.training_rows+c.guard_rows,rx=c.training_columns+c.guard_columns;
    const int count=(2*ry+1)*(2*rx+1)-(2*c.guard_rows+1)*(2*c.guard_columns+1);
    const auto alpha=static_cast<long double>(count)*
      std::expm1(-std::log(static_cast<long double>(c.pfa))/count);
    std::fill(prefix.begin(),prefix.end(),0.L);
    const size_t stride=n+1;
    for (int y=0;y<m;++y) {
      long double sum=0;
      for (int x=0;x<n;++x) {
        sum+=staging.range_doppler[y*n+x];
        prefix[(y+1)*stride+x+1]=prefix[y*stride+x+1]+sum;
        require(std::isfinite(prefix[(y+1)*stride+x+1]),"IQ CFAR accumulation overflow");
      }
    }
    auto rect=[&](int y0,int x0,int y1,int x1) {
      return prefix[y1*stride+x1]-prefix[y0*stride+x1]-prefix[y1*stride+x0]+prefix[y0*stride+x0];
    };
    auto &out=staging.range_doppler_threshold;
    std::fill(out.begin(),out.end(),-1.);
    for (int y=ry;y<m-ry;++y) for (int x=rx;x<n-rx;++x) {
      const auto sum=rect(y-ry,x-rx,y+ry+1,x+rx+1)-
        rect(y-c.guard_rows,x-c.guard_columns,y+c.guard_rows+1,x+c.guard_columns+1);
      const auto v=alpha*std::max(0.L,sum)/count;
      require(std::isfinite(v) && v<=std::numeric_limits<double>::max(),"IQ CFAR threshold overflow");
      out[y*n+x]=static_cast<double>(v);
    }
  }
  void add_detection(const RawBlock &b,int k,int dbin,std::complex<double> cross) {
    auto &r=staging;
    require(finite(cross),"IQ angle accumulation overflow");
    require(r.detections.size()<128,"IQ detection count exceeds 128");
    double sx=std::clamp(std::arg(cross)/std::numbers::pi,-1.,1.),range=k*r.range_bin_m;
    double variance=std::max(1.,range*range*.0025);
    const uint64_t offset=two_d?static_cast<uint64_t>(dbin*n+k):static_cast<uint64_t>(k);
    const uint64_t scale=two_d?65536:2048;
    require(b.envelope.sequence_no<=(9007199254740991ULL-offset)/scale,"IQ detection ID exceeds safe integer");
    r.detections.push_back({b.envelope.sequence_no*scale+offset,range*sx,
      range*std::sqrt(1-sx*sx),variance,variance,range,(dbin-m/2)*r.velocity_bin_mps,.8});
  }
  template<class T> void run(const RawBlock &b,std::vector<std::complex<T>> &a,
                             std::vector<std::complex<T>> &d) {
    auto &r=staging;
    r.sample_rate_hz=b.sample_rate_hz;r.chirp_period_s=b.chirp_period_s;
    r.range_bin_m=calibrated_scale(299792458.,b.sample_rate_hz,b.slope_hz_s,2.*n,
                                  299792458.*b.sample_rate_hz/(2*b.slope_hz_s*n));
    r.velocity_bin_mps=calibrated_scale(b.wavelength_m,1.,b.chirp_period_s,2.*m,
                                      b.wavelength_m/(2*b.chirp_period_s*m));
    require(r.range_bin_m<=std::numeric_limits<double>::max()/(n-1) &&
            r.velocity_bin_mps<=std::numeric_limits<double>::max()/(m/2) &&
            std::isfinite((n-1)*r.range_bin_m) && std::isfinite((m/2)*r.velocity_bin_mps),
            "Derived IQ plotted axis extent exceeds finite float64 range");
    r.detections.clear(); std::fill(r.range_profile.begin(),r.range_profile.end(),0.);
    for (size_t i=0;i<a.size();++i) {
      a[i]={static_cast<T>(b.iq[i].real()),static_cast<T>(b.iq[i].imag())};
      require(finite(a[i]),"IQ precision conversion overflow");
    }
    for (int k=0;k<n;++k) {r.i[k]=a[k].real();r.q[k]=a[k].imag();}
    const T gain=config.window=="hann"?(static_cast<T>(n)-T{1})/(T{2}*static_cast<T>(n)):T{1};
    for (int ch=0;ch<2;++ch) for (int j=0;j<m;++j) {
      auto row=std::span(a).subspan((ch*m+j)*n,n);
      std::complex<T> dc=0;
      for (auto v:row) dc+=v;
      dc/=static_cast<T>(n);
      for (int k=0;k<n;++k) {
        const T window=config.window=="hann"?T{.5}-T{.5}*std::cos(T{2}*std::numbers::pi_v<T>*static_cast<T>(k)/static_cast<T>(n-1)):T{1};
        row[k]=(row[k]-dc)*window;
      }
      if (!planned) {
        transform(row);
        for (auto &v:row) v/=static_cast<T>(n)*gain;
      }
    }
    if constexpr (std::is_same_v<T,double>) if (planned) {
      range_plan->execute_batch(a,2*m);
      for (auto &v:a) v/=static_cast<T>(n)*gain;
    }
    for (auto v:a) require(finite(v),"IQ range FFT overflow");
    for (int k=0;k<n;++k) {
      r.spectrum[k]=std::abs(a[k]);r.phase[k]=std::arg(a[k]);
      if constexpr (std::is_same_v<T,double>) {
        for (int j=0;j<m;++j) r.range_profile[k]+=std::norm(a[j*n+k])/m;
      } else {
        T sum=0; for (int j=0;j<m;++j) sum+=std::norm(a[j*n+k])/static_cast<T>(m);
        r.range_profile[k]=sum;
      }
    }
    threshold_1d();
    const T dg=(static_cast<T>(m)-T{1})/(T{2}*static_cast<T>(m));
    const int channels=two_d?2:1;
    // Gather [channel][range][chirp], contiguous batches for planned Doppler FFTs.
    for (int ch=0;ch<channels;++ch) for (int k=0;k<n;++k) {
      auto col=std::span(d).subspan((ch*n+k)*m,m);
      for (int j=0;j<m;++j) col[j]=a[(ch*m+j)*n+k]*
        (T{.5}-T{.5}*std::cos(T{2}*std::numbers::pi_v<T>*static_cast<T>(j)/static_cast<T>(m-1)));
      if (!planned) transform(col);
    }
    if constexpr (std::is_same_v<T,double>) if (planned)
      for (int ch=0;ch<channels;++ch) for (int k=0;k<n;k+=256) {
        const size_t batches=std::min(256,n-k);
        doppler_plan->execute_batch(std::span(d).subspan((ch*n+k)*m,batches*m),batches);
      }
    for (int ch=0;ch<channels;++ch) for (int k=0;k<n;++k) for (int j=0;j<m;++j) {
      const auto value=d[(ch*n+k)*m+j]/(static_cast<T>(m)*dg);
      require(finite(value),"IQ Doppler FFT overflow");
      if (ch==0) r.range_doppler[((j+m/2)%m)*n+k]=std::norm(value);
    }
    for (auto v:r.range_profile) require(std::isfinite(v),"IQ range power overflow");
    for (auto v:r.range_doppler) require(std::isfinite(v),"IQ Doppler power overflow");
    for (auto v:r.threshold) require(std::isfinite(v),"IQ range threshold overflow");
    if (two_d) {
      threshold_2d();
      for (int y=1;y<m-1;++y) for (int k=1;k<n/2;++k) {
        const int index=y*n+k;
        const auto value=r.range_doppler[index],threshold=r.range_doppler_threshold[index];
        if (threshold<0 || value<=threshold) continue;
        bool peak=true;
        for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
          const int other=(y+dy)*n+k+dx;
          if (other!=index && (r.range_doppler[other]>value ||
              (r.range_doppler[other]==value && other<index))) peak=false;
        }
        if (!peak) continue;
        const int j=(y+m/2)%m;
        const auto cross=d[(n+k)*m+j]*std::conj(d[k*m+j]);
        add_detection(b,k,y,{static_cast<double>(cross.real()),static_cast<double>(cross.imag())});
      }
    } else for (int k=1;k<n/2-1;++k)
      if (r.threshold[k]>=0 && r.range_profile[k]>r.threshold[k] &&
          r.range_profile[k]>r.range_profile[k-1] && r.range_profile[k]>r.range_profile[k+1]) {
        int dbin=0;
        for (int j=1;j<m;++j) if (r.range_doppler[j*n+k]>r.range_doppler[dbin*n+k]) dbin=j;
        std::complex<T> cross=0;
        for (int j=0;j<m;++j) cross+=a[(m+j)*n+k]*std::conj(a[j*n+k]);
        require(finite(cross),"IQ angle accumulation overflow");
        add_detection(b,k,dbin,{static_cast<double>(cross.real()),static_cast<double>(cross.imag())});
      }
    for (const auto &v:r.detections)
      require(std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.var_x)&&
              std::isfinite(v.radial_velocity_mps),"IQ detection geometry overflow");
  }
};
IQProcessor::IQProcessor(const Config &c):impl_(std::make_unique<Impl>(c)) {}
IQProcessor::~IQProcessor()=default;
IQProcessor::IQProcessor(IQProcessor &&) noexcept=default;
IQProcessor &IQProcessor::operator=(IQProcessor &&) noexcept=default;
SignalResult IQProcessor::make_output() const { SignalResult r;impl_->prepare(r);return r; }
void IQProcessor::process_into(const RawBlock &b,SignalResult &out) {
  b.validate();
  require(b.samples_per_chirp==impl_->n && b.chirps==impl_->m,"Raw dimensions differ from immutable config");
  if (impl_->fp32) impl_->run(b,impl_->values32,impl_->doppler32);
  else impl_->run(b,impl_->values64,impl_->doppler64);
  out=impl_->staging;
}
SignalResult IQProcessor::process(const RawBlock &b) { auto r=make_output();process_into(b,r);return r; }
size_t IQProcessor::workspace_bytes() const noexcept {
  const auto &p=*impl_;
  return (p.values64.size()+p.doppler64.size())*sizeof(std::complex<double>)+
    (p.values32.size()+p.doppler32.size())*sizeof(std::complex<float>)+p.prefix.size()*sizeof(long double)+
    (p.range_plan?p.range_plan->storage_bytes():0)+(p.doppler_plan?p.doppler_plan->storage_bytes():0);
}
SignalResult process_iq(const RawBlock &b,const Config &c) { return IQProcessor(c).process(b); }
} // namespace aether
