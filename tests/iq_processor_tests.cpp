#include <aethersense/iq_processor.hpp>
#include <aethersense/compute_backend.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <numbers>
#include <stdexcept>
using namespace aether;
constexpr double pi=std::numbers::pi;
static std::atomic<size_t> allocations{};
static bool count_allocations=false;
void *operator new(std::size_t n) {
  if (count_allocations) ++allocations;
  if (void *p=std::malloc(n?n:1)) return p;
  throw std::bad_alloc();
}
void operator delete(void *p) noexcept {std::free(p);}
void *operator new[](std::size_t n) {return ::operator new(n);}
void operator delete[](void *p) noexcept {::operator delete(p);}
void check(bool ok,const char *m) {if (!ok) throw std::runtime_error(m);}
template<class F> void rejects(F f) {bool rejected=false;try {f();}catch(const std::exception&){rejected=true;} check(rejected,"Expected rejection");}
// Frozen pre-extension implementation: catches arithmetic/order and whole-signal regressions.
SignalResult legacy_iq(const RawBlock &b, const Config &c) {
  b.validate();
  if (c.fft_size != b.samples_per_chirp || c.chirps != b.chirps)
    throw std::invalid_argument("Raw dimensions differ from immutable config");
  int n = b.samples_per_chirp, m = b.chirps;
  SignalResult r;
  r.sample_rate_hz = b.sample_rate_hz;
  r.chirp_period_s = b.chirp_period_s;
  r.bins = n;
  r.chirps = m;
  r.range_bin_m = 299792458. * b.sample_rate_hz / (2 * b.slope_hz_s * n);
  r.velocity_bin_mps = b.wavelength_m / (2 * b.chirp_period_s * m);
  r.i.resize(n);
  r.q.resize(n);
  r.spectrum.resize(n);
  r.phase.resize(n);
  r.range_profile.resize(n);
  r.range_doppler.resize(n * m);
  for (int k = 0; k < n; ++k) {
    r.i[k] = b.iq[k].real();
    r.q[k] = b.iq[k].imag();
  }
  auto a = b.iq;
  double gain = c.window == "hann" ? (n - 1.) / (2 * n) : 1.;
  for (int ch = 0; ch < 2; ++ch)
    for (int j = 0; j < m; ++j) {
      auto row = std::span(a).subspan((ch * m + j) * n, n);
      std::complex<double> dc = 0;
      for (auto v : row)
        dc += v;
      dc /= n;
      for (int k = 0; k < n; ++k)
        row[k] =
            (row[k] - dc) * (c.window == "hann" ? .5 - .5 * std::cos(2 * pi * k / (n - 1)) : 1.);
      fft(row);
      for (auto &v : row)
        v /= n * gain;
    }
  for (int k = 0; k < n; ++k) {
    r.spectrum[k] = std::abs(a[k]);
    r.phase[k] = std::arg(a[k]);
    for (int j = 0; j < m; ++j)
      r.range_profile[k] += std::norm(a[j * n + k]) / m;
  }
  r.threshold = ca_cfar(r.range_profile, c.cfar_training, c.cfar_guard, c.cfar_pfa);
  std::vector<std::complex<double>> col(m);
  double dg = (m - 1.) / (2 * m);
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < m; ++j)
      col[j] = a[j * n + k] * (.5 - .5 * std::cos(2 * pi * j / (m - 1)));
    fft(col);
    for (int j = 0; j < m; ++j)
      r.range_doppler[((j + m / 2) % m) * n + k] = std::norm(col[j] / (m * dg));
  }
  for (int k = 1; k < n / 2 - 1; ++k)
    if (r.threshold[k] >= 0 && r.range_profile[k] > r.threshold[k] &&
        r.range_profile[k] > r.range_profile[k - 1] &&
        r.range_profile[k] > r.range_profile[k + 1]) {
      int dbin = 0;
      for (int j = 1; j < m; ++j)
        if (r.range_doppler[j * n + k] > r.range_doppler[dbin * n + k])
          dbin = j;
      std::complex<double> cross = 0;
      for (int j = 0; j < m; ++j)
        cross += a[(m + j) * n + k] * std::conj(a[j * n + k]);
      double sx = std::clamp(std::arg(cross) / pi, -1., 1.), range = k * r.range_bin_m;
      double variance = std::max(1., range * range * .0025);
      r.detections.push_back({b.envelope.sequence_no * 2048 + static_cast<uint64_t>(k), range * sx,
                              range * std::sqrt(1 - sx * sx), variance, variance, range,
                              (dbin - m / 2) * r.velocity_bin_mps, .8});
    }
  return r;
}
RawBlock tone(const Config &c) {
  RawBlock b;
  b.samples_per_chirp=c.fft_size;b.chirps=c.chirps;
  b.chirp_period_s=std::max(.001,c.fft_size/1e6);
  b.iq.resize(2*c.fft_size*c.chirps);b.envelope.sequence_no=3;
  for(int ch=0;ch<2;++ch) for(int j=0;j<c.chirps;++j) for(int k=0;k<c.fft_size;++k)
    b.iq[(ch*c.chirps+j)*c.fft_size+k]=
      std::polar(1.,2*pi*(10.*k/c.fft_size+1.*j/c.chirps)+ch*.3)+
      .2*std::polar(1.,2*pi*(4.25*k/c.fft_size-2.*j/c.chirps)-ch*.4)+std::complex<double>(.2,-.1);
  return b;
}
void close_vector(const std::vector<double>&a,const std::vector<double>&b,double tolerance) {
  check(a.size()==b.size(),"Vector shape mismatch");
  for(size_t i=0;i<a.size();++i)
    check(std::abs(a[i]-b[i])<=tolerance*(1+std::abs(b[i])),"Numerical parity");
}
void calibration_scale_tests() {
  Config c;c.input="synthetic_iq";c.fft_size=32;c.chirps=8;c.cfar_training=3;c.cfar_guard=1;
  for (const auto *mode:{"legacy","float32","planned64"}) {
    c.iq_processing.reset();
    if(std::string(mode)!="legacy") {
      c.iq_processing=IQProcessingConfig{};
      if(std::string(mode)=="float32")c.iq_processing->precision="float32";
      else c.iq_processing->fft_backend="preallocated_cpu";
    }
    IQProcessor processor(c);
    auto zero=tone(c);std::fill(zero.iq.begin(),zero.iq.end(),std::complex<double>{});
    auto output=processor.process(zero);
    check(output.detections.empty(),"Calibration fixture requires no detections");
    const auto original=output.to_json();
    for(int defect=0;defect<6;++defect) {
      auto bad=zero;
      if(defect==0) {bad.sample_rate_hz=1e308;bad.slope_hz_s=1e-308;}
      if(defect==1) {bad.sample_rate_hz=1e-300;bad.slope_hz_s=1e308;bad.chirp_period_s=1e308;}
      if(defect==2) {bad.sample_rate_hz=1e308;bad.chirp_period_s=1e-300;bad.wavelength_m=1e308;}
      if(defect==3) {bad.chirp_period_s=1e308;bad.wavelength_m=std::numeric_limits<double>::denorm_min();}
      if(defect==4) {bad.sample_rate_hz=1e301;bad.slope_hz_s=1.;}
      if(defect==5) {bad.chirp_period_s=.1;bad.wavelength_m=1e308;}
      bad.validate(); // The derived scales, rather than raw field validity, must reject.
      rejects([&]{processor.process_into(bad,output);});
      check(output.to_json()==original,"Calibration failure mutated output");
    }
    for(int extreme=0;extreme<3;++extreme) {
      auto valid=zero;valid.chirp_period_s=1e308;valid.wavelength_m=1e308;
      if(extreme==0) {valid.sample_rate_hz=1e308;valid.slope_hz_s=1e308;}
      if(extreme==1) {valid.sample_rate_hz=1e-300;valid.slope_hz_s=1e-300;}
      if(extreme==2) {valid.sample_rate_hz=1e-10;valid.slope_hz_s=1e308;valid.wavelength_m=1e-10;}
      valid.validate();processor.process_into(valid,output);
      check(std::isfinite(output.range_bin_m)&&output.range_bin_m>0 &&
            std::isfinite(output.velocity_bin_mps)&&output.velocity_bin_mps>0,
            "Representable extreme calibration rejected or serialized nonfinite scale");
      const auto signal=output.to_json();
      check(signal["range_bin_m"].is_number()&&signal["velocity_bin_mps"].is_number(),
            "Calibration became JSON null");
      if(extreme<2) {
        check(std::abs(output.range_bin_m-299792458./64)<1e-8,"Stable range scale");
        check(output.velocity_bin_mps==1./16,"Stable velocity scale");
      }
    }
    processor.process_into(zero,output);
    check(output.to_json()==original,"Calibration failure workspace recovery");
  }
}
void detection_capacity_tests() {
  Config c;c.input="synthetic_iq";c.fft_size=1024;c.chirps=8;c.cfar_training=1;
  c.cfar_guard=8;c.cfar_pfa=.1;c.window="rectangular";
  RawBlock comb;comb.samples_per_chirp=1024;comb.chirps=8;comb.chirp_period_s=.002;
  std::vector<std::complex<double>> bins(1024);
  for(int k=2;k<512;k+=2)bins[k]=1.;
  fft(bins,true);
  for(int ch=0;ch<2;++ch)for(int j=0;j<8;++j)for(auto v:bins)comb.iq.push_back(v*1024.);
  comb.validate();
  check(legacy_iq(comb,c).detections.size()>128,"Comb must exercise legacy public detection cap");
  for(const auto *backend:{"legacy","reference_cpu","preallocated_cpu","native_simd","float32"}) {
    if(std::string(backend)=="native_simd"&&!native_simd_available())continue;
    c.iq_processing.reset();
    if(std::string(backend)!="legacy") {
      c.iq_processing=IQProcessingConfig{};
      if(std::string(backend)=="float32")c.iq_processing->precision="float32";
      else c.iq_processing->fft_backend=backend;
    }
    IQProcessor processor(c);auto zero=comb;
    std::fill(zero.iq.begin(),zero.iq.end(),std::complex<double>{});
    auto output=processor.process(zero);const auto original=output.to_json();
    bool rejected=false;
    try {processor.process_into(comb,output);} catch(const std::invalid_argument &e) {
      rejected=std::string(e.what()).find("count exceeds 128")!=std::string::npos;
    }
    check(rejected,"Detection count must reject visibly for all paths");
    check(output.to_json()==original,"Detection overflow mutated output");
    processor.process_into(zero,output);
    check(output.to_json()==original,"Detection overflow workspace recovery");
  }
}
int main(int argc,char **argv) {
 try {
  Config c;c.input="synthetic_iq";c.fft_size=32;c.chirps=16;c.cfar_training=3;c.cfar_guard=1;
  auto b=tone(c);
  if(argc==2 && std::string(argv[1])=="--oracle") {
    json result=json::array(),raw=json::array();
    for(auto z:b.iq)raw.push_back({z.real(),z.imag()});
    for(const auto &precision:{"float64","float32"}) {
      c.iq_processing=IQProcessingConfig{};
      c.iq_processing->precision=precision;c.iq_processing->cfar_mode="range_doppler_2d";
      auto processed=process_iq(b,c);
      json obs=json::array();for(const auto &d:processed.detections)obs.push_back(serialize(d));
      result.push_back({{"config",c.to_json()},{"signal",processed.to_json()},{"observations",obs}});
    }
    std::cout<<json{{"raw",raw},{"cases",result}}.dump()<<'\n';return 0;
  }
  calibration_scale_tests();
  detection_capacity_tests();
  for(int n:{32,128,1024}) for(int m:{8,32,64}) for(auto window:{"hann","rectangular"}) {
    c.fft_size=n;c.chirps=m;c.window=window;c.iq_processing.reset();
    Random rng(100+n+m);Envelope env;env.sequence_no=3;
    auto input=synthesize_iq(c,truth_at(c,0),rng,env);
    const auto legacy=legacy_iq(input,c),actual=process_iq(input,c);
    check(legacy.to_json()==actual.to_json(),"Default full-signal legacy parity");
    check(legacy.detections.size()==actual.detections.size(),"Legacy detection count");
    for(size_t i=0;i<legacy.detections.size();++i)
      check(serialize(legacy.detections[i])==serialize(actual.detections[i]),"Legacy detection parity");
  }
  c.fft_size=32;c.chirps=16;c.window="hann";b=tone(c);
  c.iq_processing=IQProcessingConfig{};
  check(Config::from_json(c.to_json()).to_json()==c.to_json(),"Config round trip");
  for(auto backend:{"reference_cpu","preallocated_cpu","native_simd"}) {
    if(std::string(backend)=="native_simd" && !native_simd_available())continue;
    c.iq_processing->fft_backend=backend;
    for(auto mode:{"range_1d","range_doppler_2d"}) {
      c.iq_processing->cfar_mode=mode;
      IQProcessor processor(c);auto output=processor.make_output();
      processor.process_into(b,output);const auto first=output.to_json();
      for(int i=0;i<5;++i)processor.process_into(b,output);
      check(output.to_json()==first,"Repeated owned workspace differs");
      auto expected=process_iq(b,c);
      check(output.to_json()==expected.to_json(),"Convenience parity");
      check(output.processing->fft_backend==backend,"Backend metadata");
      auto reference_config=c;reference_config.iq_processing->fft_backend="reference_cpu";
      const auto scalar=process_iq(b,reference_config);
      close_vector(output.range_doppler,scalar.range_doppler,3e-13);
      close_vector(output.range_profile,scalar.range_profile,3e-13);
      close_vector(output.range_doppler_threshold,scalar.range_doppler_threshold,3e-13);
      check(output.detections.size()==scalar.detections.size(),"Backend detection parity");
      for(size_t i=0;i<scalar.detections.size();++i) {
        check(output.detections[i].id==scalar.detections[i].id,"Backend detection ID parity");
        check(std::abs(output.detections[i].x-scalar.detections[i].x)<1e-9,"Backend geometry parity");
      }
      if(std::string(mode)=="range_doppler_2d") {
        Cfar2DConfig box;
        close_vector(output.range_doppler_threshold,ca_cfar_2d(output.range_doppler,16,32,box),0);
        check(!output.detections.empty(),"2D tone detection");
        for(const auto &d:output.detections) {
          const auto flat=d.id-b.envelope.sequence_no*65536;
          check(flat%32>0 && flat%32<16,"2D range selection");
          check(output.range_doppler[flat]>output.range_doppler_threshold[flat],"2D CUT threshold");
        }
      }
      auto bad=b;bad.iq[0]={NAN,0};
      rejects([&]{processor.process_into(bad,output);});
      check(output.to_json()==first,"Invalid input mutated output");
      bad=b;for(auto &z:bad.iq)z={std::numeric_limits<double>::max()/4,0};
      rejects([&]{processor.process_into(bad,output);});
      check(output.to_json()==first,"Overflow mutated output");
      processor.process_into(b,output);check(output.to_json()==first,"Workspace failure recovery");
      if(std::string(backend)!="native_simd") {
        allocations=0;count_allocations=true;
        processor.process_into(b,output);
        count_allocations=false;
        check(allocations==0,"Warmed reference/plan process_into allocated");
      }
    }
  }
  c.iq_processing->fft_backend="reference_cpu";c.iq_processing->precision="float32";
  IQProcessor float_processor(c);auto output=float_processor.make_output();
  float_processor.process_into(b,output);
  allocations=0;count_allocations=true;float_processor.process_into(b,output);count_allocations=false;
  check(allocations==0,"Warmed float32 process_into allocated");
  check(output.processing->precision=="float32","Float32 metadata");
  auto precise=c;precise.iq_processing->precision="float64";
  auto reference=process_iq(b,precise);
  close_vector(output.range_doppler,reference.range_doppler,2e-6);
  close_vector(output.range_profile,reference.range_profile,2e-6);
  for(auto backend:{"preallocated_cpu","native_simd"}) {
    auto bad=c;bad.iq_processing->fft_backend=backend;rejects([&]{bad.validate();});
  }
  auto valid=c.to_json();
  for(const auto *field:{"schema_version","precision","fft_backend","cfar_mode","cfar_2d"}) {
    auto j=valid;j["iq_processing"].erase(field);rejects([&]{Config::from_json(j);});
  }
  for(auto invalid:{0.,1.,1e-7,.100001,std::numeric_limits<double>::infinity()}) {
    auto j=valid;j["iq_processing"]["cfar_2d"]["pfa"]=invalid;rejects([&]{Config::from_json(j);});
  }
  for(auto field:{"training_rows","training_columns","guard_rows","guard_columns"}) {
    auto j=valid;j["iq_processing"]["cfar_2d"][field]=4294967297ULL;rejects([&]{Config::from_json(j);});
    j=valid;j["iq_processing"]["cfar_2d"][field]=1.25;rejects([&]{Config::from_json(j);});
  }
  auto j=valid;j["iq_processing"]["extra"]=true;rejects([&]{Config::from_json(j);});
  j=valid;j["iq_processing"]["cfar_2d"]["extra"]=true;rejects([&]{Config::from_json(j);});
  j=valid;j["input"]="synthetic_detections";rejects([&]{Config::from_json(j);});
  j=valid;j["iq_processing"]["cfar_mode"]="range_1d";
  j["iq_processing"]["cfar_2d"]["training_rows"]=8;rejects([&]{Config::from_json(j);});
  // Maximum supported grid/chunking; both channels undergo planned 2D Doppler.
  c.fft_size=1024;c.chirps=64;c.iq_processing->precision="float64";
  c.iq_processing->fft_backend="preallocated_cpu";
  IQProcessor maximum(c);auto maximum_output=maximum.process(tone(c));
  check(maximum_output.range_doppler_threshold.size()==65536,"Maximum 2D shape");
  check(maximum.workspace_bytes()<16*1024*1024,"Workspace budget");
  auto maximum_input=tone(c);maximum.process_into(maximum_input,maximum_output);
  allocations=0;count_allocations=true;maximum.process_into(maximum_input,maximum_output);count_allocations=false;
  check(allocations==0,"Maximum planned warmed process_into allocated");
  std::cout<<"PASS IQ default exact parity18 grids/windows; explicit3 backends/2 modes; float32; 2D; zero warmed allocations; atomic failures; calibration scales/axis extents; legacy/configured detection cap; bounds; maximum grid\n";
 }catch(const std::exception &e){count_allocations=false;std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
