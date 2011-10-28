#include "lm/search_hashed.hh"

#include "lm/binary_format.hh"
#include "lm/blank.hh"
#include "lm/lm_exception.hh"
// TODO: remove hack
#include "lm/model.hh"
#include "lm/read_arpa.hh"
#include "lm/vocab.hh"

#include "util/bit_packing.hh"
#include "util/file_piece.hh"

#include <fstream>
#include <string>
#include <vector>

#include <math.h>

namespace lm {
namespace ngram {

namespace {

struct Message {
  const float *from;
  Rest *to;

  void SetFrom(const ProbBackoff &in_from) const {}

  void SetFrom(const Rest &in_from) { from = &in_from.right; }

  void SetTo(ProbBackoff &in_to) const {}
  void SetTo(Prob &in_to) const {}

  void SetTo(Rest &in_to) { to = &in_to; }

  void Apply() {
    to->right += pow(10.0, to->backoff) * *from;
  }
};

template <class Lower> class MessageCollector;

template <> class MessageCollector<Rest> {
  public:
    MessageCollector() {}

    void Send(const Message &message) {
      messages_.push_back(message);
    }

    void Apply() {
      for (std::vector<Message>::iterator i = messages_.begin(); i != messages_.end(); ++i) {
        i->Apply();
      }
    }
  private:
    std::vector<Message> messages_;
};

template <> class MessageCollector<ProbBackoff> {
  public:
    MessageCollector() {}
    void Send(const Message &) {}
    void Apply() {}
};

template <> class MessageCollector<Prob> {
  public:
    MessageCollector() {}
    void Send(const Message &) {}
    void Apply() {}
};

template <class Middle> void AdjustLower(const WordIndex *vocab_ids, const unsigned int n, const Rest *unigram, const Middle &same_mid, float higher, Rest &to) {
  // TODO: this lookup was already done.  Don't repeat it.  
  uint64_t hash = static_cast<WordIndex>(vocab_ids[0]);
  for (const WordIndex *i = vocab_ids + 1; i < vocab_ids + n - 1; ++i) {
    hash = detail::CombineWordHash(hash, *i);
  }
  typename Middle::ConstIterator i;
  UTIL_THROW_IF(!same_mid.Find(hash, i), FormatLoadException, "But this should have been just added as a blank");
  float shift = -fabsf(i->GetValue().prob);
  float unigram_left = unigram[*vocab_ids].left;

  to.right += powf(10.0, higher) * (higher - unigram_left) - powf(10.0, to.backoff + shift) * (to.backoff + shift - unigram_left);
}

template <class Middle> void AdjustLower(const WordIndex *, const unsigned int, const ProbBackoff *, const Middle &, float, ProbBackoff &) {}

void AdjustLower(const WordIndex *vocab_ids, const Rest *unigram, float higher, Rest &to) {
  const Rest &basis = unigram[*vocab_ids];
  float unigram_left = basis.left;
  float shift = -fabsf(basis.prob);
  float add = powf(10.0, higher) * (higher - unigram_left) - powf(10.0, to.backoff + shift) * (to.backoff + shift - unigram_left);
  if (&to == unigram + 1) {
    std::cerr << "Adding " << add << " due to higher=" << higher << " and shift=" << shift << std::endl;
  }
  to.right += add;
}
void AdjustLower(const WordIndex *, const ProbBackoff *, float, ProbBackoff &) {}

/* These are passed to ReadNGrams so that n-grams with zero backoff that appear as context will still be used in state. */
template <class Middle> class ActivateLowerMiddle {
  public:
    ActivateLowerMiddle(const typename Middle::Value *unigram, Middle &middle) : unigram_(unigram), modify_(middle) {}

    void operator()(const WordIndex *vocab_ids, const unsigned int n, float higher) {
      uint64_t hash = static_cast<WordIndex>(vocab_ids[1]);
      for (const WordIndex *i = vocab_ids + 2; i < vocab_ids + n; ++i) {
        hash = detail::CombineWordHash(hash, *i);
      }
      typename Middle::MutableIterator i;
      // TODO: somehow get text of n-gram for this error message.
      if (!modify_.UnsafeMutableFind(hash, i))
        UTIL_THROW(FormatLoadException, "The context of every " << n << "-gram should appear as a " << (n-1) << "-gram");
      SetExtension(i->MutableValue().backoff);
      AdjustLower(vocab_ids, n, unigram_, modify_, higher, i->MutableValue());
    }

  private:
    const typename Middle::Value *unigram_;
    Middle &modify_;
};

template <class LowerValue> class ActivateUnigram {
  public:
    explicit ActivateUnigram(LowerValue *unigram) : modify_(unigram) {}

    void operator()(const WordIndex *vocab_ids, const unsigned int /*n*/, float higher) {
      // assert(n == 2);
      SetExtension(modify_[vocab_ids[1]].backoff);
      AdjustLower(vocab_ids, modify_, higher, modify_[vocab_ids[1]]);
    }

  private:
    LowerValue *modify_;
};

class AwfulGlobal {
  public:
    AwfulGlobal() {
      util::FilePiece uni("1");
      std::vector<uint64_t> number;
      ReadARPACounts(uni, number);
      assert(number.size() == 1);
      unigram_.resize(number[0] + 1);
      std::vector<char> vocab_backing(ProbingVocabulary::Size(number[0] + 1, Config()));
      ProbingVocabulary vocab;
      vocab.SetupMemory(&vocab_backing.front(), ProbingVocabulary::Size(number[0] + 1, Config()), number[0] + 1, Config()); 
      PositiveProbWarn warn;
      Read1Grams(uni, (size_t)number[0], vocab, &*unigram_.begin(), warn);

      if (vocab.SawUnk()) {
        unigram_.resize(number[0]);
      }

      models_[0] = new ProbingModel("2");
      models_[1] = new ProbingModel("3");
      models_[2] = new ProbingModel("4");
    }

    ~AwfulGlobal() {
      delete models_[0];
      delete models_[1];
      delete models_[2];
    }

    float GetRest(const WordIndex *vocab_ids, unsigned int n) {
      if (n == 1) {
        return unigram_[vocab_ids[0]].prob;
      } else {
        State ignored;
        return models_[n - 2]->FullScoreForgotState(vocab_ids + 1, vocab_ids + n, vocab_ids[0], ignored).prob;
      }
    }

    void ApplyUnigram(Rest *weights) {
      for (size_t i = 0; i < unigram_.size(); ++i) {
        weights[i].left = unigram_[i].prob;
      }
      unigram_.clear();
    }
    void ApplyUnigram(ProbBackoff * /*weights */) {}

  private:
    std::vector<ProbBackoff> unigram_;
    const ProbingModel *models_[3];
};

AwfulGlobal awful;

void SetRest(const WordIndex *vocab_ids, unsigned int n, Rest &weights) {
  weights.left = awful.GetRest(vocab_ids, n);
  weights.right = powf(10.0, weights.backoff) * weights.backoff;
}

void SetRest(const WordIndex *, unsigned int, ProbBackoff &) {}

void SetRest(const WordIndex *, unsigned int, Prob &) {}

// Fix SRI's stupidity wrt omitting B C D even though A B C D appears.  
template <class Middle> const typename Middle::Value &FixSRI(int lower, float negative_lower_prob, unsigned int n, const uint64_t *keys, const WordIndex *vocab_ids, typename Middle::Value *unigrams, std::vector<Middle> &middle) {
  typename Middle::Value blank;
  // Note that negative_lower_prob is the negative of the probability (so it's currently >= 0).  We still want the sign bit off to indicate left extension, so I just do -= on the backoffs.  
  blank.prob = negative_lower_prob;
  blank.backoff = kNoExtensionBackoff;
  // An entry was found at lower (order lower + 2).  
  // We need to insert blanks starting at lower + 1 (order lower + 3).
  unsigned int fix = static_cast<unsigned int>(lower + 1);
  uint64_t backoff_hash = detail::CombineWordHash(static_cast<uint64_t>(vocab_ids[1]), vocab_ids[2]);
  const typename Middle::Value *ret;
  if (fix == 0) {
    // Insert a missing bigram.  
    blank.prob -= unigrams[vocab_ids[1]].backoff;
    SetExtension(unigrams[vocab_ids[1]].backoff);
    // Bigram including a unigram's backoff
    SetRest(vocab_ids, 2, blank);
    ret = &middle[0].Insert(Middle::Packing::Make(keys[0], blank))->GetValue();
    fix = 1;
  } else {
    for (unsigned int i = 3; i < fix + 2; ++i) backoff_hash = detail::CombineWordHash(backoff_hash, vocab_ids[i]);
    ret = NULL;
  }
  // fix >= 1.  Insert trigrams and above.  
  for (; fix <= n - 3; ++fix) {
    typename Middle::MutableIterator gotit;
    if (middle[fix - 1].UnsafeMutableFind(backoff_hash, gotit)) {
      float &backoff = gotit->MutableValue().backoff;
      SetExtension(backoff);
      blank.prob -= backoff;
    }
    SetRest(vocab_ids, fix + 2, blank);
    ret = &middle[fix].Insert(Middle::Packing::Make(keys[fix], blank))->GetValue();
    backoff_hash = detail::CombineWordHash(backoff_hash, vocab_ids[fix + 2]);
  }
  return *ret;
}

template <class Voc, class Store, class Middle, class Activate> void ReadNGrams(util::FilePiece &f, const unsigned int n, const size_t count, const Voc &vocab, typename Middle::Value *unigrams, std::vector<Middle> &middle, Activate activate, Store &store, PositiveProbWarn &warn) {
  ReadNGramHeader(f, n);

  // vocab ids of words in reverse order
  std::vector<WordIndex> vocab_ids(n);
  std::vector<uint64_t> keys(n-1);
  typename Store::Packing::Value value;
  typename Middle::MutableIterator found;
  MessageCollector<typename Middle::Value> messages;
  Message message;
  for (size_t i = 0; i < count; ++i) {
    ReadNGram(f, n, vocab, &*vocab_ids.begin(), value, warn);

    SetRest(&vocab_ids.front(), n, value);

    keys[0] = detail::CombineWordHash(static_cast<uint64_t>(vocab_ids.front()), vocab_ids[1]);
    for (unsigned int h = 1; h < n - 1; ++h) {
      keys[h] = detail::CombineWordHash(keys[h-1], vocab_ids[h+1]);
    }
    // Initially the sign bit is on, indicating it does not extend left.  Most already have this but there might +0.0.  
    util::SetSign(value.prob);
    message.SetTo(store.Insert(Store::Packing::Make(keys[n-2], value))->MutableValue());

    // Go back and find the longest right-aligned entry, informing it that it extends left.  Normally this will match immediately, but sometimes SRI is dumb.  
    int lower;
    util::FloatEnc fix_prob;
    for (lower = n - 3; ; --lower) {
      if (lower == -1) {
        typename Middle::Value &val = unigrams[vocab_ids.front()];
        fix_prob.f = val.prob;
        fix_prob.i &= ~util::kSignBit;
        val.prob = fix_prob.f;
        message.SetFrom(val);
        break;
      }
      if (middle[lower].UnsafeMutableFind(keys[lower], found)) {
        // Turn off sign bit to indicate that it extends left.  
        fix_prob.f = found->MutableValue().prob;
        fix_prob.i &= ~util::kSignBit;
        found->MutableValue().prob = fix_prob.f;
        message.SetFrom(found->GetValue());
        // We don't need to recurse further down because this entry already set the bits for lower entries.  
        break;
      }
    }
    if (lower != static_cast<int>(n) - 3) {
      message.SetFrom(FixSRI(lower, fix_prob.f, n, &*keys.begin(), &*vocab_ids.begin(), unigrams, middle));
    }
    activate(&*vocab_ids.begin(), n, value.prob);
    messages.Send(message);
  }

  messages.Apply();

  store.FinishedInserting();
}

} // namespace
namespace detail {

/*const float kRestWeights[4][6] = {
  {0.0158082, 0.142494, 0.00694743, 0.72096, 0.0844032, 0.0781648},
  {-0.00637761, 0.43229, -0.0558925, 0.575986, -0.00787255, -0.0102447},
  {0.0067138, 0.638819, -0.0523763, 0.403326, -0.0298767, -0.00771542},
  {0.00247556, 0.680603, -0.0418086, 0.349169, -0.0272333, 0.00609867},
};*/

//std::fstream RestLog("rest_log", std::ios::out);

/*void LogRest(unsigned char order, float prob, const Rest &weights) {
  RestLog << (unsigned)order << ' ' << prob << ' ' << weights.backoff << ' ' << weights.rest << ' ' << weights.lower << ' ' << weights.upper;
}*/
 
template <class MiddleT, class LongestT> uint8_t *TemplateHashedSearch<MiddleT, LongestT>::SetupMemory(uint8_t *start, const std::vector<uint64_t> &counts, const Config &config) {
  std::size_t allocated = Unigram::Size(counts[0]);
  unigram = Unigram(start, allocated);
  start += allocated;
  for (unsigned int n = 2; n < counts.size(); ++n) {
    allocated = Middle::Size(counts[n - 1], config.probing_multiplier);
    middle_.push_back(Middle(start, allocated));
    start += allocated;
  }
  allocated = Longest::Size(counts.back(), config.probing_multiplier);
  longest = Longest(start, allocated);
  start += allocated;
  return start;
}

void UnigramRight(const ProbBackoff *, std::size_t) {}

void UnigramRight(Rest *unigram, std::size_t size) {
  double sum_double = 0.0;
  for (const Rest *i = unigram; i != unigram + size; ++i) {
    sum_double += pow(10.0, static_cast<double>(i->prob)) * static_cast<double>(i->prob - i->left);
  }
  std::cerr << "Unigram sum is " << sum_double << std::endl;
  float sum = static_cast<float>(sum_double);
  for (Rest *i = unigram; i != unigram + size; ++i) {
    i->right = pow(10.0, i->backoff) * (i->backoff + sum);
  }
  std::cerr << "Initial , is " << unigram[1].right << std::endl;
}

void DebugPrint(const Rest &rest) {
  std::cerr << "Logging , as " << rest.right << std::endl;
}

void DebugPrint(const ProbBackoff &backoff) {
}

template <class MiddleT, class LongestT> template <class Voc> void TemplateHashedSearch<MiddleT, LongestT>::InitializeFromARPA(const char * /*file*/, util::FilePiece &f, const std::vector<uint64_t> &counts, const Config &config, Voc &vocab, Backing &backing) {
  // TODO: fix sorted.
  SetupMemory(GrowForSearch(config, 0, Size(counts, config), backing), counts, config);

  PositiveProbWarn warn(config.positive_log_probability);

  Read1Grams(f, counts[0], vocab, unigram.Raw(), warn);
  CheckSpecials(config, vocab);
  awful.ApplyUnigram(unigram.Raw());
  UnigramRight(unigram.Raw(), counts[0] + 1 - vocab.SawUnk());
  DebugPrint(unigram.Raw()[1]);

  try {
    if (counts.size() > 2) {
      ReadNGrams(f, 2, counts[1], vocab, unigram.Raw(), middle_, ActivateUnigram<LowerValue>(unigram.Raw()), middle_[0], warn);
    }
    DebugPrint(unigram.Raw()[1]);
    for (unsigned int n = 3; n < counts.size(); ++n) {
      ReadNGrams(f, n, counts[n-1], vocab, unigram.Raw(), middle_, ActivateLowerMiddle<Middle>(unigram.Raw(), middle_[n-3]), middle_[n-2], warn);
    }
    if (counts.size() > 2) {
      ReadNGrams(f, counts.size(), counts[counts.size() - 1], vocab, unigram.Raw(), middle_, ActivateLowerMiddle<Middle>(unigram.Raw(), middle_.back()), longest, warn);
    } else {
      ReadNGrams(f, counts.size(), counts[counts.size() - 1], vocab, unigram.Raw(), middle_, ActivateUnigram<LowerValue>(unigram.Raw()), longest, warn);
    }
  } catch (util::ProbingSizeException &e) {
    UTIL_THROW(util::ProbingSizeException, "Avoid pruning n-grams like \"bar baz quux\" when \"foo bar baz quux\" is still in the model.  KenLM will work when this pruning happens, but the probing model assumes these events are rare enough that using blank space in the probing hash table will cover all of them.  Increase probing_multiplier (-p to build_binary) to add more blank spaces.\n");
  }
  DebugPrint(unigram.Raw()[1]);
  ReadEnd(f);
}

template <class MiddleT, class LongestT> void TemplateHashedSearch<MiddleT, LongestT>::LoadedBinary() {
  unigram.LoadedBinary();
  for (typename std::vector<Middle>::iterator i = middle_.begin(); i != middle_.end(); ++i) {
    i->LoadedBinary();
  }
  longest.LoadedBinary();
}

template class TemplateHashedSearch<ProbingHashedSearch::Middle, ProbingHashedSearch::Longest>;

template void TemplateHashedSearch<ProbingHashedSearch::Middle, ProbingHashedSearch::Longest>::InitializeFromARPA(const char *, util::FilePiece &f, const std::vector<uint64_t> &counts, const Config &, ProbingVocabulary &vocab, Backing &backing);

template class TemplateHashedSearch<RestProbingHashedSearch::Middle, RestProbingHashedSearch::Longest>;

template void TemplateHashedSearch<RestProbingHashedSearch::Middle, RestProbingHashedSearch::Longest>::InitializeFromARPA(const char *, util::FilePiece &f, const std::vector<uint64_t> &counts, const Config &, ProbingVocabulary &vocab, Backing &backing);

} // namespace detail
} // namespace ngram
} // namespace lm
