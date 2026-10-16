#include <aethersense/engine.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
using C = std::complex<double>;
static int checks;
static void require(bool condition) {
  ++checks;
  if (!condition) throw std::runtime_error("Standalone kernel assertion " + std::to_string(checks));
}
template<class E, class F> static void rejects(F action) {
  bool caught = false;
  try { action(); } catch (const E &) { caught = true; }
  require(caught);
}
int main() {
  try {
    std::vector<C> empty, three(3), huge(aether::standalone_fft_max_elements + 1);
    rejects<std::invalid_argument>([&]{ aether::fft(empty); });
    rejects<std::invalid_argument>([&]{ aether::fft(three); });
    rejects<std::invalid_argument>([&]{ aether::fft(huge); });
    std::vector<C> oversized_radix2(aether::standalone_fft_max_elements * 2);
    rejects<std::invalid_argument>([&]{ aether::fft(oversized_radix2); });
    require(oversized_radix2.front()==C(0) && oversized_radix2.back()==C(0));
    std::vector<C> singleton{{3,-4}};
    aether::fft(singleton); aether::fft(singleton,true);
    require(singleton[0]==C(3,-4));
    std::vector<C> imaginary_bad{{0,std::numeric_limits<double>::infinity()}};
    rejects<std::invalid_argument>([&]{aether::fft(imaginary_bad);});
    rejects<std::invalid_argument>([&]{aether::matched_filter(imaginary_bad,three);});
    // Whole input is validated before bit reversal or butterflies.
    for (double bad : {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
      std::vector<C> values{{1,2},{3,4},{5,6},{bad,0}};
      rejects<std::invalid_argument>([&]{ aether::fft(values); });
      require(values[0] == C(1,2) && values[1] == C(3,4) && values[2] == C(5,6));
      rejects<std::invalid_argument>([&]{ aether::matched_filter(values, three); });
      rejects<std::invalid_argument>([&]{ aether::matched_filter(three, values); });
    }
    std::vector<C> x{{1,2},{-2,3},{4,-1},{0,2},{1,-3},{-4,2},{3,0},{2,1}};
    auto transformed=x;
    aether::fft(transformed);
    // Independent direct long-double DFT, no radix butterflies or shared helper.
    for (std::size_t k=0;k<x.size();++k) {
      std::complex<long double> expected{};
      for (std::size_t j=0;j<x.size();++j) {
        const auto angle=-2*std::numbers::pi_v<long double>*k*j/x.size();
        expected += std::complex<long double>(x[j].real(),x[j].imag()) *
                    std::complex<long double>(std::cos(angle),std::sin(angle));
      }
      require(std::abs(transformed[k]-C(expected.real(),expected.imag())) < 1e-12);
    }
    aether::fft(transformed,true);
    for(std::size_t k=0;k<x.size();++k) require(std::abs(transformed[k]-x[k])<1e-12);
    std::vector<C> impulse(aether::standalone_fft_max_elements);
    impulse[0]=1;
    aether::fft(impulse);
    require(impulse.front()==C(1) && impulse.back()==C(1));
    aether::fft(impulse,true);
    require(impulse.front()==C(1) && impulse.back()==C(0));
    const double maximum=std::numeric_limits<double>::max();
    std::vector<C> overflow{{maximum,0},{maximum,0}};
    rejects<std::overflow_error>([&]{aether::fft(overflow);});
    std::vector<C> one{{1,0}}, maxone{{maximum,0}}, two{{2,0}};
    rejects<std::invalid_argument>([&]{aether::matched_filter(empty,one);});
    rejects<std::invalid_argument>([&]{aether::matched_filter(one,empty);});
    rejects<std::invalid_argument>([&]{aether::matched_filter(huge,one);});
    rejects<std::invalid_argument>([&]{aether::matched_filter(one,huge);});
    // Exact maximum output succeeds; adding one reference element exceeds it.
    auto result=aether::matched_filter(impulse,one);
    require(result.size()==aether::matched_filter_max_output_elements && result[0]==C(1));
    rejects<std::invalid_argument>([&]{aether::matched_filter(impulse,std::vector<C>(2));});
    std::vector<C> budget(4096, C(1));
    result=aether::matched_filter(budget,budget);
    require(result.size()==8191 && result[4095]==C(4096));
    budget.push_back(1);
    rejects<std::invalid_argument>([&]{aether::matched_filter(budget,budget);});
    rejects<std::overflow_error>([&]{aether::matched_filter(maxone,two);});
    std::vector<C> sum_overflow{{maximum,0},{maximum,0}}, reference(2,C(1));
    rejects<std::overflow_error>([&]{aether::matched_filter(sum_overflow,reference);});
    std::vector<C> signal{{1,2},{3,-1}}, ref{{2,1},{-1,2}};
    result=aether::matched_filter(signal,ref);
    // Hand-calculated full convolution with conjugate-reversed reference.
    require(result==std::vector<C>({C(3,-4),C(-1,-2),C(5,-5)}));
    std::vector<double> powers{1,2,3,4,5,6,7};
    const auto thresholds=aether::ca_cfar(powers,2,1,0.0625);
    require(thresholds==std::vector<double>({-1,-1,-1,16,-1,-1,-1}));
    rejects<std::invalid_argument>([&]{aether::ca_cfar(powers,std::numeric_limits<int>::max(),0,0.1);});
    rejects<std::invalid_argument>([&]{aether::ca_cfar(powers,1,std::numeric_limits<int>::max(),0.1);});
    rejects<std::invalid_argument>([&]{aether::ca_cfar(powers,std::numeric_limits<int>::max(),std::numeric_limits<int>::max(),0.1);});
    std::vector<double> power_limit(aether::standalone_cfar_max_elements,1);
    auto limit_thresholds=aether::ca_cfar(power_limit,1,0,0.25);
    require(limit_thresholds.size()==power_limit.size() && limit_thresholds[1]==2 &&
            limit_thresholds.front()==-1 && limit_thresholds.back()==-1);
    power_limit.push_back(1);
    rejects<std::invalid_argument>([&]{aether::ca_cfar(power_limit,1,0,0.1);});
    std::vector<double> extreme_power(3,maximum);
    rejects<std::overflow_error>([&]{aether::ca_cfar(extreme_power,1,0,0.25);});
    extreme_power.assign(3,maximum/4);
    rejects<std::overflow_error>([&]{aether::ca_cfar(extreme_power,1,0,0.01);});
    const auto tiny_pfa=aether::ca_cfar(std::vector<double>(3,0),1,0,std::numeric_limits<double>::denorm_min());
    require(tiny_pfa[1]==0);
    for (double bad : {-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
      auto invalid_powers=powers; invalid_powers[0]=bad;
      rejects<std::invalid_argument>([&]{aether::ca_cfar(invalid_powers,1,0,0.25);});
    }
    std::cout << checks << " standalone kernel checks passed\n";
  } catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
}
