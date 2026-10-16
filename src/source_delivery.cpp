#include <aethersense/source_delivery.hpp>
#include <aethersense/engine.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace aether {
namespace {
using J = nlohmann::json;
constexpr uint64_t safe = 9007199254740991ULL;
void require(bool b, const char* message) { if (!b) throw std::invalid_argument(std::string("Source delivery: ") + message); }
void fields(const J& j, std::initializer_list<const char*> required, std::initializer_list<const char*> optional = {}) {
  require(j.is_object(), "expected object");
  for (auto k : required) require(j.contains(k), "missing field");
  for (const auto& [k,v] : j.items()) {
    (void)v;
    auto found = [&](const char* s) { return k == s; };
    require(std::any_of(required.begin(),required.end(),found) || std::any_of(optional.begin(),optional.end(),found), "unknown field");
  }
}
uint64_t integer(const J& j, uint64_t max = safe) {
  require(j.is_number_integer(), "expected integer");
  if (j.is_number_unsigned()) { auto n=j.get<uint64_t>(); require(n<=max,"integer bound"); return n; }
  auto n=j.get<int64_t>(); require(n>=0 && static_cast<uint64_t>(n)<=max,"integer bound"); return static_cast<uint64_t>(n);
}
std::string text(const J& j, size_t cap = 64) { require(j.is_string() && j.get_ref<const std::string&>().size()<=cap,"string bound"); return j.get<std::string>(); }
J optional_number(const std::optional<uint64_t>& n) { return n ? J(*n) : J(nullptr); }
std::optional<uint64_t> nullable(const J& j) { return j.is_null() ? std::nullopt : std::optional<uint64_t>(integer(j)); }
const std::array<uint64_t SourceLedger::*,11> counters{
 &SourceLedger::generated,&SourceLedger::offered,&SourceLedger::accepted,&SourceLedger::overflow_rejected,
 &SourceLedger::failed_admission,&SourceLedger::cancelled_before_offer,&SourceLedger::failed_after_generation,
 &SourceLedger::processed_frames,&SourceLedger::late_rejected,&SourceLedger::cancelled_after_accept,&SourceLedger::failed_processing};
bool hash_valid(std::string_view s) { return s.size()==64 && std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}); }
void require_close_reason(std::string_view s) { require(s=="explicit_stop"||s=="duration_complete"||s=="storage_limit"||s=="failure"||s=="shutdown","close reason"); }
void checkpoint_shape(const SourceFrameCheckpoint& cp) {
  require(cp.ledger.scope=="run_cumulative_commit_fence"&&cp.ledger.source_gap_before&&
    cp.ledger.late_rejected_since_previous_commit&&cp.ledger.last_processed_sequence==cp.sequence&&
    cp.ledger.last_processed_timestamp_ns==cp.timestamp_ns,"typed checkpoint shape/binding");
}
uint64_t gap_after(const SourceLedger& prior,uint64_t seq) {
  if (!prior.last_processed_sequence) return seq;
  require(seq>*prior.last_processed_sequence,"nonincreasing processed sequence"); return seq-*prior.last_processed_sequence-1;
}
}
void SourceOptions::validate() const {
  require(version==1 && mode=="independently_paced","unsupported source options");
  require(std::isfinite(rate_hz)&&rate_hz>=1&&rate_hz<=100&&queue_capacity>=1&&queue_capacity<=8,"source option bounds");
  require(overflow=="fail_run"||overflow=="drop_newest_with_metric","source overflow policy");
  if(delivery) require(delivery->every_n>=2&&delivery->every_n<=1024&&delivery->delay_frames>=1&&delivery->delay_frames<=8,"delivery bounds");
}
J SourceOptions::to_json() const {
  J j={{"schema_version",version},{"mode",mode},{"rate_hz",rate_hz},{"queue_capacity",queue_capacity},{"overflow",overflow}};
  if(delivery) j["delivery"]={{"every_n",delivery->every_n},{"delay_frames",delivery->delay_frames}}; return j;
}
SourceOptions SourceOptions::from_json(const J& j) {
  fields(j,{"schema_version","mode","rate_hz","queue_capacity","overflow"},{"delivery"});
  SourceOptions o; o.version=static_cast<int>(integer(j.at("schema_version"),1)); o.mode=text(j.at("mode"));
  require(j.at("rate_hz").is_number(),"rate number"); o.rate_hz=j.at("rate_hz").get<double>();
  o.queue_capacity=static_cast<uint32_t>(integer(j.at("queue_capacity"),8)); o.overflow=text(j.at("overflow"));
  if(j.contains("delivery")) { const auto& d=j.at("delivery"); fields(d,{"every_n","delay_frames"});
    o.delivery=SourceDeliveryProfile{static_cast<uint32_t>(integer(d.at("every_n"),1024)),static_cast<uint32_t>(integer(d.at("delay_frames"),8))}; }
  o.validate(); return o;
}
void SourceLedger::validate(const Config& c,std::string_view expected) const {
  require(c.source_options.has_value(),"source config missing"); c.source_options->validate();
  require(version==1 && (scope=="run_cumulative_commit_fence"||scope=="capture_open_fence"||scope=="capture_close_fence"||scope=="run_sample"),"ledger version/scope");
  require(hash_valid(config_sha256)&&hash_valid(expected)&&config_sha256==expected,"config hash binding");
  for(auto p:counters) require(this->*p<=safe,"counter bound");
  require(generated<=180000 && ready_depth<=c.source_options->queue_capacity&&delayed_depth<=8&&compute_active<=1,"ledger capacity bound");
  require(generated==delayed_depth+offered+cancelled_before_offer+failed_after_generation,"generated accounting");
  require(offered==accepted+overflow_rejected+failed_admission,"offered accounting");
  require(accepted==ready_depth+compute_active+processed_frames+late_rejected+cancelled_after_accept+failed_processing,"accepted accounting");
  require(last_generated_sequence.has_value()==(generated!=0),"last generated presence");
  if(last_generated_sequence) require(*last_generated_sequence==generated-1 && *last_generated_sequence*c.dt_s<c.duration_s,"last generated/config bound");
  require(last_processed_sequence.has_value()==(processed_frames!=0)&&last_processed_timestamp_ns.has_value()==(processed_frames!=0),"processed endpoint presence");
  if(last_processed_sequence) {
    require(*last_processed_sequence<generated && *last_processed_sequence+1>=processed_frames,"processed sequence bound");
    require(*last_processed_timestamp_ns>=0&&*last_processed_timestamp_ns<=3600000000000LL&&
      *last_processed_timestamp_ns==std::llround(*last_processed_sequence*c.dt_s*1e9),"processed timestamp/config binding");
  }
  if(scope!="run_sample") require(compute_active==0,"persisted fence compute_active");
  require(!late_rejected||last_processed_sequence.has_value(),"late rejection without prior processed timestamp");
  if(last_processed_sequence) require(processed_frames+late_rejected<=*last_processed_sequence+1,"late/processed unique sequence bound");
  if(scope=="run_cumulative_commit_fence") {
    require(processed_frames>0&&compute_active==0&&source_gap_before&&late_rejected_since_previous_commit,"commit fields");
    require(*source_gap_before<=*last_processed_sequence && *late_rejected_since_previous_commit<=late_rejected,"commit delta bounds");
    if(processed_frames==1) require(*source_gap_before==*last_processed_sequence,"initial commit gap");
  } else require(!source_gap_before&&!late_rejected_since_previous_commit,"frame-only field in generic ledger");
}
J SourceLedger::to_json() const {
  J j={{"schema_version",version},{"scope",scope},{"config_sha256",config_sha256},
#define S(x) {#x,x}
    S(generated),S(offered),S(accepted),S(overflow_rejected),S(failed_admission),S(cancelled_before_offer),S(failed_after_generation),
    S(processed_frames),S(late_rejected),S(cancelled_after_accept),S(failed_processing),S(ready_depth),S(delayed_depth),S(compute_active),
#undef S
    {"last_generated_sequence",optional_number(last_generated_sequence)}, {"last_processed_sequence",optional_number(last_processed_sequence)},
    {"last_processed_timestamp_ns",last_processed_timestamp_ns?J(*last_processed_timestamp_ns):J(nullptr)}};
  if(source_gap_before) j["source_gap_before"]=*source_gap_before;
  if(late_rejected_since_previous_commit) j["late_rejected_since_previous_commit"]=*late_rejected_since_previous_commit;
  return j;
}
SourceLedger SourceLedger::from_json(const J& j,const Config& c,std::string_view hash) {
  fields(j,{"schema_version","scope","config_sha256","generated","offered","accepted","overflow_rejected","failed_admission",
    "cancelled_before_offer","failed_after_generation","processed_frames","late_rejected","cancelled_after_accept","failed_processing",
    "ready_depth","delayed_depth","compute_active","last_generated_sequence","last_processed_sequence","last_processed_timestamp_ns"},
    {"source_gap_before","late_rejected_since_previous_commit"});
  SourceLedger s; s.version=static_cast<int>(integer(j.at("schema_version"),1)); s.scope=text(j.at("scope")); s.config_sha256=text(j.at("config_sha256"));
#define R(x) s.x=integer(j.at(#x))
  R(generated);R(offered);R(accepted);R(overflow_rejected);R(failed_admission);R(cancelled_before_offer);R(failed_after_generation);
  R(processed_frames);R(late_rejected);R(cancelled_after_accept);R(failed_processing);R(ready_depth);R(delayed_depth);R(compute_active);
#undef R
  s.last_generated_sequence=nullable(j.at("last_generated_sequence"));s.last_processed_sequence=nullable(j.at("last_processed_sequence"));
  if(!j.at("last_processed_timestamp_ns").is_null()) s.last_processed_timestamp_ns=static_cast<int64_t>(integer(j.at("last_processed_timestamp_ns"),3600000000000ULL));
  if(j.contains("source_gap_before")) s.source_gap_before=integer(j.at("source_gap_before"));
  if(j.contains("late_rejected_since_previous_commit")) s.late_rejected_since_previous_commit=integer(j.at("late_rejected_since_previous_commit"));
  s.validate(c,hash);return s;
}
std::optional<SourceFrameCheckpoint> source_frame_checkpoint(const J& frame,const Config& c,std::string_view hash) {
  if(!c.source_options) {
    require(!frame.contains("source_delivery"),"undeclared frame extension");
    if(frame.contains("quality_flags")&&frame.at("quality_flags").is_array())
      for(const auto& flag:frame.at("quality_flags"))
        require(flag!="source_gap"&&flag!="late_delivery_rejected","undeclared source quality flag");
    return std::nullopt;
  }
  require(frame.contains("source_delivery"),"missing frame source_delivery");
  SourceFrameCheckpoint cp{SourceLedger::from_json(frame.at("source_delivery"),c,hash),integer(frame.at("sequence_no")),
    static_cast<int64_t>(integer(frame.at("timestamp_ns"),3600000000000ULL))};
  require(cp.ledger.scope=="run_cumulative_commit_fence"&&cp.ledger.last_processed_sequence==cp.sequence&&
    cp.ledger.last_processed_timestamp_ns==cp.timestamp_ns,"frame endpoint binding");
  const auto& flags=frame.at("quality_flags");require(flags.is_array(),"quality flags type");
  auto flag=[&](const char* k){return std::count(flags.begin(),flags.end(),J(k));};
  require(flag("source_gap")==(*cp.ledger.source_gap_before>0?1:0),"source gap flag binding");
  require(flag("late_delivery_rejected")==(*cp.ledger.late_rejected_since_previous_commit>0?1:0),"late flag binding");return cp;
}
void validate_source_ledger_progress(const SourceLedger& a,const SourceLedger& b) {
  require(a.config_sha256==b.config_sha256,"ledger config changed");
  require(a.last_processed_sequence.has_value()==a.last_processed_timestamp_ns.has_value()&&
    b.last_processed_sequence.has_value()==b.last_processed_timestamp_ns.has_value(),"typed processed endpoint pair");
  for(auto p:counters) require(b.*p>=a.*p,"cumulative counter decreased");
  if(a.last_generated_sequence) require(b.last_generated_sequence&&*b.last_generated_sequence>=*a.last_generated_sequence,"generated endpoint decreased");
  if(a.last_processed_sequence) require(b.last_processed_sequence&&*b.last_processed_sequence>=*a.last_processed_sequence&&
    b.last_processed_timestamp_ns&&*b.last_processed_timestamp_ns>=*a.last_processed_timestamp_ns,"processed endpoint decreased");
}
void validate_source_frame_progress(const SourceFrameCheckpoint& a,const SourceFrameCheckpoint& b) {
  checkpoint_shape(a);checkpoint_shape(b);
  validate_source_ledger_progress(a.ledger,b.ledger);
  require(b.ledger.processed_frames==a.ledger.processed_frames+1,"processed frame progression");
  require(b.sequence>a.sequence&&b.timestamp_ns>a.timestamp_ns,"nonincreasing frame sequence/time");
  require(b.ledger.source_gap_before==b.sequence-a.sequence-1,"source gap progression");
  require(b.ledger.late_rejected_since_previous_commit==b.ledger.late_rejected-a.ledger.late_rejected,"late delta progression");
}
void validate_source_capture_first(const SourceLedger& start,const SourceFrameCheckpoint& first) {
  checkpoint_shape(first);
  require(start.scope=="capture_open_fence"&&start.compute_active==0,"capture start scope/active");
  validate_source_ledger_progress(start,first.ledger);
  require(first.ledger.processed_frames==start.processed_frames+1,"capture first processed count");
  require(first.ledger.source_gap_before==gap_after(start,first.sequence),"capture first gap binding");
  const auto late=*first.ledger.late_rejected_since_previous_commit;
  require(late>=first.ledger.late_rejected-start.late_rejected&&late<=first.ledger.late_rejected,"capture first late bounds");
  if(start.processed_frames==0) require(late==first.ledger.late_rejected,"initial first late exact");
}
J SourceCaptureSummary::to_json() const {
  if(summary_status=="unavailable") return {{"schema_version",version},{"summary_status",summary_status}};
  require(capture_start&&capture_end,"capture ledgers missing");
  return {{"schema_version",version},{"summary_status",summary_status},{"capture_start",capture_start->to_json()},
    {"capture_end",capture_end->to_json()},{"close_reason",close_reason},{"first_recorded_sequence",optional_number(first_recorded_sequence)},
    {"last_recorded_sequence",optional_number(last_recorded_sequence)},{"recorded_processed_frames",recorded_processed_frames},
    {"uncaptured_processed_at_close",uncaptured_processed_at_close}};
}
SourceCaptureSummary SourceCaptureSummary::from_json(const J& j,const Config& c,std::string_view hash) {
  SourceCaptureSummary s;
  require(j.is_object()&&j.contains("summary_status"),"summary status missing");s.summary_status=text(j.at("summary_status"));
  if(s.summary_status=="unavailable") { fields(j,{"schema_version","summary_status"});s.version=static_cast<int>(integer(j.at("schema_version"),1));require(s.version==1,"summary version");return s; }
  fields(j,{"schema_version","summary_status","capture_start","capture_end","close_reason","first_recorded_sequence","last_recorded_sequence","recorded_processed_frames","uncaptured_processed_at_close"});
  require(s.summary_status=="complete","summary status");s.version=static_cast<int>(integer(j.at("schema_version"),1));require(s.version==1,"summary version");
  s.capture_start=SourceLedger::from_json(j.at("capture_start"),c,hash);s.capture_end=SourceLedger::from_json(j.at("capture_end"),c,hash);
  require(s.capture_start->scope=="capture_open_fence"&&s.capture_end->scope=="capture_close_fence","capture scope");
  s.close_reason=text(j.at("close_reason"));require_close_reason(s.close_reason);
  s.first_recorded_sequence=nullable(j.at("first_recorded_sequence"));s.last_recorded_sequence=nullable(j.at("last_recorded_sequence"));
  s.recorded_processed_frames=integer(j.at("recorded_processed_frames"),180000);s.uncaptured_processed_at_close=integer(j.at("uncaptured_processed_at_close"),1);return s;
}
void SourceCaptureSummary::validate(const Config& c,std::string_view hash,uint64_t frames,bool closed,
 const std::optional<SourceFrameCheckpoint>& first,const std::optional<SourceFrameCheckpoint>& last) const {
  require(version==1,"summary version");
  if(summary_status=="unavailable") { require(!closed&&!capture_start&&!capture_end&&close_reason.empty()&&!first_recorded_sequence&&!last_recorded_sequence&&!recorded_processed_frames&&!uncaptured_processed_at_close,"unavailable summary shape/status");return; }
  require(summary_status=="complete"&&capture_start&&capture_end,"complete summary missing");
  capture_start->validate(c,hash);capture_end->validate(c,hash);require_close_reason(close_reason);
  require(capture_start->scope=="capture_open_fence"&&capture_end->scope=="capture_close_fence"&&capture_start->compute_active==0&&capture_end->compute_active==0,"capture fence scope/active");
  validate_source_ledger_progress(*capture_start,*capture_end);
  if(close_reason!="storage_limit") require(capture_end->ready_depth==0&&capture_end->delayed_depth==0,"terminal source backlog");
  if(close_reason=="duration_complete") require(capture_end->generated*c.dt_s>=c.duration_s,"natural duration boundary");
  require(recorded_processed_frames==frames&&frames<=180000&&uncaptured_processed_at_close<=1,"capture count");
  require(!uncaptured_processed_at_close||close_reason=="storage_limit","uncaptured close reason");
  if(closed) {
    require(close_reason!="failure"&&close_reason!="shutdown","successful capture failure reason");
    require(capture_end->failed_admission==0&&capture_end->failed_after_generation==0&&capture_end->failed_processing==0,
      "successful capture has fatal source failures");
    if(close_reason=="storage_limit") require(uncaptured_processed_at_close==1,"quota close requires one uncaptured trigger");
  }
  if(closed) require(capture_end->processed_frames-capture_start->processed_frames==frames+uncaptured_processed_at_close,"capture processed window");
  require(first.has_value()==(frames!=0)&&last.has_value()==(frames!=0)&&first_recorded_sequence.has_value()==(frames!=0)&&last_recorded_sequence.has_value()==(frames!=0),"capture row endpoint presence");
  const SourceLedger* endpoint=&*capture_start;
  if(frames) {
    checkpoint_shape(*first);checkpoint_shape(*last);
    first->ledger.validate(c,hash);last->ledger.validate(c,hash);
    require(first->sequence==*first_recorded_sequence&&last->sequence==*last_recorded_sequence,"recorded sequence endpoints");
    require(first->ledger.processed_frames==capture_start->processed_frames+1&&last->ledger.processed_frames==capture_start->processed_frames+frames,"capture frame counts binding");
    validate_source_ledger_progress(*capture_start,first->ledger);validate_source_ledger_progress(last->ledger,*capture_end);
    validate_source_capture_first(*capture_start,*first);
    endpoint=&last->ledger;
  }
  if(!closed) return; // failed prefix endpoints above remain bound; full window need not equal recorded prefix.
  if(!uncaptured_processed_at_close) require(capture_end->last_processed_sequence==endpoint->last_processed_sequence&&capture_end->last_processed_timestamp_ns==endpoint->last_processed_timestamp_ns,"capture exact closing endpoint");
  else {
    require(capture_end->last_processed_sequence&&capture_end->last_processed_timestamp_ns,"uncaptured endpoint missing");
    if(endpoint->last_processed_sequence) require(*capture_end->last_processed_sequence>*endpoint->last_processed_sequence&&*capture_end->last_processed_timestamp_ns>*endpoint->last_processed_timestamp_ns,"uncaptured endpoint progression");
  }
}
} // namespace aether
