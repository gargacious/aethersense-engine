#include <aethersense/engine.hpp>
#include <aethersense/fft_plan.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <sys/utsname.h>
#include <thread>
namespace {
using Clock=std::chrono::steady_clock;
double microseconds(Clock::time_point begin,Clock::time_point end) {
  return std::chrono::duration<double,std::micro>(end-begin).count();
}
}
int main(int argc,char **argv) {
  try {
    std::string output;
    if(argc==3 && std::string_view(argv[1])=="--output")output=argv[2];
    else if(argc!=1)throw std::invalid_argument("Usage: fft-plan-benchmark [--output FILE.json]");
    aether::json measurements=aether::json::array();
    struct Profile {size_t samples,batches,iterations;};
    for(const auto profile:{Profile{64,8,200},Profile{1024,16,50},Profile{65536,1,8}}) {
      std::vector<std::complex<double>> original(profile.samples*profile.batches);
      for(size_t index=0;index<original.size();++index)
        original[index]={std::sin(.173*index)+.125*std::cos(.931*index),std::cos(.217*index)-.2*std::sin(.337*index)};
      for(int repeat=0;repeat<3;++repeat) {
        auto values=original;
        std::optional<aether::Fft64Plan> plan;
        const auto cold_begin=Clock::now();plan.emplace(profile.samples,profile.batches);
        const auto cold_end=Clock::now();
        for(int warmup=0;warmup<5;++warmup){plan->execute_batch(values,profile.batches);plan->execute_batch(values,profile.batches,true);}
        const auto warm_begin=Clock::now();
        for(size_t iteration=0;iteration<profile.iterations;++iteration) {
          plan->execute_batch(values,profile.batches);plan->execute_batch(values,profile.batches,true);
        }
        const auto warm_end=Clock::now();
        double error=0;
        for(size_t index=0;index<values.size();++index) {
          if(!std::isfinite(values[index].real()) || !std::isfinite(values[index].imag()))
            throw std::runtime_error("Nonfinite warmed benchmark output");
          error=std::max(error,std::abs(values[index]-original[index]));
        }
        const double total_us=microseconds(warm_begin,warm_end);
        measurements.push_back({{"repeat",repeat},{"samples_per_fft",profile.samples},
          {"batches",profile.batches},{"warmup_pairs",5},{"measured_pairs",profile.iterations},
          {"cold_plan_construction_us",microseconds(cold_begin,cold_end)},
          {"warm_total_us",total_us},{"warm_forward_inverse_batch_pair_mean_us",total_us/profile.iterations},
          {"warm_mean_us_per_transform",total_us/(2*profile.iterations*profile.batches)},
          {"warm_transform_count",2*profile.iterations*profile.batches},
          {"owned_element_storage_bytes",plan->storage_bytes()},{"roundtrip_max_abs_error",error}});
      }
    }
    utsname machine{};uname(&machine);
    const aether::json report={{"schema_version",1},{"status","measured"},
      {"operation","standalone_preallocated_fft_plan_benchmark"},{"backend","scalar_complex_float64_radix2"},
      {"scope","Standalone plan only; main I/Q chain is unchanged. Forward+inverse pairs restore bounded input. No queue or interprocess work is measured."},
      {"clock","steady_clock"},{"units","microseconds"},
      {"allocation_evidence","Zero warmed new/new[] calls checked separately by fft-plan-tests; this timing executable does not instrument allocations"},
      {"cold_scope","Constructor including bit reversal/twiddle/scratch vector allocation; destructor excluded"},
      {"warm_scope","Five forward+inverse batch-pair warmups excluded; measured paired execution includes input/output finite validation and atomic copy; no input reset/allocation in timed loop"},
      {"host",{{"os",machine.sysname},{"kernel",machine.release},{"architecture",machine.machine},
                {"hardware_concurrency",std::thread::hardware_concurrency()},{"compiler",__VERSION__},
                {"cpp",__cplusplus},{"affinity","not controlled"},{"cpu_governor","not controlled"}}},
#ifdef __OPTIMIZE__
      {"compiler_optimization","enabled; see recorded build command for exact flags"},
#else
      {"compiler_optimization","not detected; see recorded build command for exact flags"},
#endif
      {"repeats_per_profile",3},{"measurements",measurements}};
    if(output.empty())std::cout<<report.dump(2)<<'\n';
    else {std::ofstream file(output);file<<report.dump(2)<<'\n';file.flush();if(!file)throw std::runtime_error("Cannot write benchmark output");}
    return 0;
  }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
