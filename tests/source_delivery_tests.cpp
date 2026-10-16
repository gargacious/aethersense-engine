#include <aethersense/engine.hpp>
#include <aethersense/source_delivery.hpp>
#include <functional>
#include <iostream>
#include <stdexcept>
using namespace aether;
namespace {
size_t checks{};
const std::string hash(64,'a');
void check(bool b,const char* message){++checks;if(!b)throw std::runtime_error(message);}
template<class F>void reject(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,"expected rejection");}
Config cfg(){Config c;c.executor="staged_threads";c.source_options=SourceOptions{1,"independently_paced",20,8,"drop_newest_with_metric",std::nullopt};return c;}
SourceLedger open(){SourceLedger s;s.scope="capture_open_fence";s.config_sha256=hash;return s;}
SourceLedger frame(uint64_t seq,uint64_t processed,uint64_t generated,uint64_t accepted,uint64_t ready=0,uint64_t late=0){
 auto s=open();s.scope="run_cumulative_commit_fence";s.generated=generated;s.offered=generated;s.accepted=accepted;
 s.overflow_rejected=generated-accepted;s.processed_frames=processed;s.ready_depth=ready;s.late_rejected=late;
 s.last_generated_sequence=generated-1;s.last_processed_sequence=seq;s.last_processed_timestamp_ns=seq*100000000;
 s.source_gap_before=0;s.late_rejected_since_previous_commit=0;return s;
}
SourceFrameCheckpoint cp(SourceLedger s){return {s,*s.last_processed_sequence,*s.last_processed_timestamp_ns};}
SourceCaptureSummary summary(const SourceLedger& start,const SourceLedger& end,uint64_t count,
 std::optional<uint64_t> first,std::optional<uint64_t> last){SourceCaptureSummary s;s.summary_status="complete";
 s.capture_start=start;s.capture_end=end;s.close_reason="explicit_stop";s.recorded_processed_frames=count;
 s.first_recorded_sequence=first;s.last_recorded_sequence=last;return s;}
json wire(const SourceLedger&s){return {{"sequence_no",*s.last_processed_sequence},{"timestamp_ns",*s.last_processed_timestamp_ns},
 {"quality_flags",json::array()},{"source_delivery",s.to_json()}};}
void options_tests(){
 Config legacy;check(!legacy.to_json().contains("source_options"),"legacy omission");
 auto c=cfg();check(Config::from_json(c.to_json()).to_json()==c.to_json(),"Config roundtrip");
 c.source_options->delivery=SourceDeliveryProfile{1024,8};c.source_options->rate_hz=100;c.source_options->queue_capacity=8;c.validate();
 c.source_options->delivery=SourceDeliveryProfile{2,1};c.source_options->rate_hz=1;c.source_options->queue_capacity=1;c.validate();
 auto j=c.to_json();
 for(auto key:{"schema_version","mode","rate_hz","queue_capacity","overflow"}){auto bad=j;bad["source_options"].erase(key);reject([&]{Config::from_json(bad);});}
 for(auto badval:{json(0),json(-1),json(9),json(1.5),json(true),json(UINT64_MAX)}){auto bad=j;bad["source_options"]["queue_capacity"]=badval;reject([&]{Config::from_json(bad);});}
 for(auto badval:{json(0),json(101),json("20"),json(nullptr)}){auto bad=j;bad["source_options"]["rate_hz"]=badval;reject([&]{Config::from_json(bad);});}
 for(auto key:{"every_n","delay_frames"}){auto bad=j;bad["source_options"]["delivery"].erase(key);reject([&]{Config::from_json(bad);});}
 auto bad=j;bad["source_options"]["unknown"]=0;reject([&]{Config::from_json(bad);});bad=j;bad["executor"]="single_thread";reject([&]{Config::from_json(bad);});
 bad=j;bad.erase("executor");reject([&]{Config::from_json(bad);});bad=j;bad["source_options"]["overflow"]="drop_oldest_noncritical";reject([&]{Config::from_json(bad);});
}
void ledger_tests(){
 auto c=cfg();auto a=frame(0,1,100,3,2);a.validate(c,hash);auto b=frame(1,2,100,3,1);b.validate(c,hash);
 validate_source_frame_progress(cp(a),cp(b)); // depths decrease while cumulative counters do not.
 check(SourceLedger::from_json(a.to_json(),c,hash).to_json()==a.to_json(),"ledger roundtrip");
 auto original=a.to_json();
 for(const auto& [key,value]:original.items()){
  (void)value;auto j=a.to_json();j.erase(key);reject([&]{SourceLedger::from_json(j,c,hash);});
 }
 for(const auto& key:{"generated","offered","accepted","overflow_rejected","failed_admission","cancelled_before_offer",
  "failed_after_generation","processed_frames","late_rejected","cancelled_after_accept","failed_processing","ready_depth","delayed_depth","compute_active",
  "source_gap_before","late_rejected_since_previous_commit"}){
  for(auto value:{json(-1),json(1.25),json(true),json(UINT64_MAX)}){auto j=a.to_json();j[key]=value;reject([&]{SourceLedger::from_json(j,c,hash);});}
 }
 for(auto scope:{"capture_open_fence","capture_close_fence","run_sample"}){auto s=open();s.scope=scope;s.validate(c,hash);
  auto j=s.to_json();j["source_gap_before"]=0;reject([&]{SourceLedger::from_json(j,c,hash);});}
 auto s=open();s.generated=s.offered=s.accepted=s.compute_active=1;s.last_generated_sequence=0;reject([&]{s.validate(c,hash);});
 s.scope="run_sample";s.validate(c,hash); // only live scope permits active reservation.
 auto j=a.to_json();j["generated"]=99;reject([&]{SourceLedger::from_json(j,c,hash);});
 j=a.to_json();j["last_processed_timestamp_ns"]=1;reject([&]{SourceLedger::from_json(j,c,hash);});
 j=a.to_json();j["config_sha256"]=std::string(64,'b');reject([&]{SourceLedger::from_json(j,c,hash);});
 j=a.to_json();j["unknown"]=0;reject([&]{SourceLedger::from_json(j,c,hash);});
 auto f=wire(a);check(source_frame_checkpoint(f,c,hash).has_value(),"frame binding valid");
 f["sequence_no"]=1;reject([&]{source_frame_checkpoint(f,c,hash);});f=wire(a);f["quality_flags"].push_back("late_delivery_rejected");reject([&]{source_frame_checkpoint(f,c,hash);});
 Config legacy;reject([&]{source_frame_checkpoint(f,legacy,"");});f.erase("source_delivery");reject([&]{source_frame_checkpoint(f,legacy,"");});
 auto impossible=b;impossible.late_rejected=1;impossible.ready_depth=0;
 reject([&]{impossible.validate(c,hash);});
 auto forged=b;forged.source_gap_before=1;reject([&]{validate_source_frame_progress(cp(a),cp(forged));});
 forged=b;forged.overflow_rejected--;reject([&]{validate_source_ledger_progress(a,forged);});
}
void capture_tests(){
 auto c=cfg();auto start=open();auto f0=cp(frame(0,1,100,3,2));auto f1=cp(frame(1,2,100,3,1));auto f2=cp(frame(2,3,100,3));
 auto end=f2.ledger;end.scope="capture_close_fence";end.source_gap_before.reset();end.late_rejected_since_previous_commit.reset();
 auto s=summary(start,end,3,0,2);s.validate(c,hash,3,true,f0,f2);
 auto round=SourceCaptureSummary::from_json(s.to_json(),c,hash);round.validate(c,hash,3,true,f0,f2);check(round.to_json()==s.to_json(),"summary roundtrip");
 // Future source overflow does not alter the actual captured contiguous prefix.
 check(f0.ledger.overflow_rejected==97&&f2.sequence==2,"future overflow prefix fixture");
 auto reason=s;reason.close_reason="duration_complete";reason.capture_end->generated=3;reason.capture_end->offered=3;reason.capture_end->overflow_rejected=0;reason.capture_end->last_generated_sequence=2;
 reject([&]{reason.validate(c,hash,3,true,f0,f2);});
 reason=s;reason.capture_end->accepted++;reason.capture_end->overflow_rejected--;reason.capture_end->ready_depth=1;
 reject([&]{reason.validate(c,hash,3,true,f0,f2);});
 for(int fatal=0;fatal<3;++fatal){auto bad=s;auto& e=*bad.capture_end;e.generated++;e.last_generated_sequence=*e.last_generated_sequence+1;
  if(fatal==0){e.offered++;e.failed_admission++;}
  if(fatal==1)e.failed_after_generation++;
  if(fatal==2){e.offered++;e.accepted++;e.failed_processing++;}
  e.validate(c,hash);validate_source_ledger_progress(f2.ledger,e);check(true,"balanced monotonic fatal summary fixture");
  reject([&]{bad.validate(c,hash,3,true,f0,f2);});
 }
 auto quota_zero=s;quota_zero.close_reason="storage_limit";reject([&]{quota_zero.validate(c,hash,3,true,f0,f2);});
 auto forged=s;forged.capture_end->last_processed_sequence=3;forged.capture_end->last_processed_timestamp_ns=300000000;
 reject([&]{forged.validate(c,hash,3,true,f0,f2);});
 forged=s;forged.uncaptured_processed_at_close=1;reject([&]{forged.validate(c,hash,3,true,f0,f2);});
 forged=s;forged.first_recorded_sequence=1;reject([&]{forged.validate(c,hash,3,true,f0,f2);});
 auto empty=summary(start,start,0,std::nullopt,std::nullopt);empty.capture_end->scope="capture_close_fence";empty.validate(c,hash,0,true,{},{});
 forged=empty;forged.capture_end=frame(0,1,1,1);forged.capture_end->scope="capture_close_fence";forged.capture_end->source_gap_before.reset();forged.capture_end->late_rejected_since_previous_commit.reset();
 reject([&]{forged.validate(c,hash,0,true,{},{});});
 forged.close_reason="storage_limit";forged.uncaptured_processed_at_close=1;forged.validate(c,hash,0,true,{},{});check(true,"zero-row quota uncaptured valid");
 auto unavailable=SourceCaptureSummary::from_json({{"schema_version",1},{"summary_status","unavailable"}},c,hash);unavailable.validate(c,hash,0,false,{},{});
 reject([&]{unavailable.validate(c,hash,0,true,{},{});});auto j=unavailable.to_json();j["capture_start"]=start.to_json();reject([&]{SourceCaptureSummary::from_json(j,c,hash);});
 // Mid-capture previous global commit can have late events before capture-open.
 SourceLedger mid=open();mid.generated=mid.offered=mid.accepted=12;mid.processed_frames=2;mid.late_rejected=5;mid.ready_depth=5;
 mid.last_generated_sequence=11;mid.last_processed_sequence=10;mid.last_processed_timestamp_ns=1000000000;mid.validate(c,hash);
 auto first=frame(11,3,12,12,2,7);first.source_gap_before=0;first.late_rejected_since_previous_commit=3;first.validate(c,hash);
 validate_source_capture_first(mid,cp(first));check(true,"midcapture late pre-open allowed");
 auto malformed=cp(first);malformed.ledger.late_rejected_since_previous_commit.reset();reject([&]{validate_source_capture_first(mid,malformed);});
 first.source_gap_before=1;reject([&]{validate_source_capture_first(mid,cp(first));});first.source_gap_before=0;
 first.late_rejected_since_previous_commit=1;reject([&]{validate_source_capture_first(mid,cp(first));});
 first.late_rejected_since_previous_commit=8;reject([&]{validate_source_capture_first(mid,cp(first));});
 (void)f1;
}
}
int main(){try{options_tests();ledger_tests();capture_tests();std::cout<<"PASS "<<checks<<" source codec assertions\n";}catch(const std::exception&e){std::cerr<<"FAIL after "<<checks<<": "<<e.what()<<'\n';return 1;}}
