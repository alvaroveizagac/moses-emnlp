#ifndef LM_SEARCH_HASHED__
#define LM_SEARCH_HASHED__

#include "lm/model_type.hh"
#include "lm/config.hh"
#include "lm/read_arpa.hh"
#include "lm/return.hh"
#include "lm/weights.hh"

#include "util/bit_packing.hh"
#include "util/key_value_packing.hh"
#include "util/probing_hash_table.hh"

#include <algorithm>
#include <iostream>
#include <vector>

namespace util { class FilePiece; }

namespace lm {
namespace ngram {
struct Backing;
namespace detail {

inline uint64_t CombineWordHash(uint64_t current, const WordIndex next) {
  uint64_t ret = (current * 8978948897894561157ULL) ^ (static_cast<uint64_t>(1 + next) * 17894857484156487943ULL);
  return ret;
}

inline void GetRest(const ProbBackoff &weights, float &prob, float &rest) {
  util::FloatEnc val;
  val.f = weights.prob;
  val.i |= util::kSignBit;
  prob = val.f;
  rest = prob;
}

inline void GetRest(const Rest &weights, float &prob, float &rest) {
  util::FloatEnc val;
  val.f = weights.prob;
  val.i |= util::kSignBit;
  prob = val.f;
  rest = 0.39374 * prob + 0.60523 * weights.rest;
  //rest = weights.rest;
}

inline void SetRest(const ProbBackoff &/*weights*/, FullScoreReturn &ret) {
  ret.rest = ret.prob;
}

inline void SetRest(const Rest &weights, FullScoreReturn &ret) {
//  ret.rest = ret.prob * 1.1 + weights.rest * -0.1;
  ret.rest = 0.39374 * ret.prob + 0.60523 * weights.rest;
//  ret.rest = weights.rest;
}

template <class MiddleT, class LongestT> class TemplateHashedSearch {
  public:
    typedef uint64_t Node;
    typedef typename MiddleT::Value LowerValue;

    class Unigram {
      public:
        Unigram() {}

        Unigram(void *start, std::size_t /*allocated*/) : unigram_(static_cast<LowerValue*>(start)) {}

        static std::size_t Size(uint64_t count) {
          return (count + 1) * sizeof(LowerValue); // +1 for hallucinate <unk>
        }

        const LowerValue &Lookup(WordIndex index) const { return unigram_[index]; }

        LowerValue &Unknown() { return unigram_[0]; }

        void LoadedBinary() {}

        // For building.
        LowerValue *Raw() { return unigram_; }

      private:
        LowerValue *unigram_;
    };

    Unigram unigram;

    void LookupUnigram(WordIndex word, float &backoff, Node &next, FullScoreReturn &ret) const {
      const LowerValue &entry = unigram.Lookup(word);
      util::FloatEnc val;
      val.f = entry.prob;
      ret.independent_left = (val.i & util::kSignBit);
      ret.extend_left = static_cast<uint64_t>(word);
      val.i |= util::kSignBit;
      ret.prob = val.f;
      SetRest(entry, ret);
      backoff = entry.backoff;
      next = static_cast<Node>(word);
    }

    typedef MiddleT Middle;

    typedef LongestT Longest;
    Longest longest;

    static const unsigned int kVersion = 0;

    // TODO: move probing_multiplier here with next binary file format update.  
    static void UpdateConfigFromBinary(int, const std::vector<uint64_t> &, Config &) {}

    static std::size_t Size(const std::vector<uint64_t> &counts, const Config &config) {
      std::size_t ret = Unigram::Size(counts[0]);
      for (unsigned char n = 1; n < counts.size() - 1; ++n) {
        ret += Middle::Size(counts[n], config.probing_multiplier);
      }
      return ret + Longest::Size(counts.back(), config.probing_multiplier);
    }

    uint8_t *SetupMemory(uint8_t *start, const std::vector<uint64_t> &counts, const Config &config);

    template <class Voc> void InitializeFromARPA(const char *file, util::FilePiece &f, const std::vector<uint64_t> &counts, const Config &config, Voc &vocab, Backing &backing);

    const Middle *MiddleBegin() const { return &*middle_.begin(); }
    const Middle *MiddleEnd() const { return &*middle_.end(); }

    Node Unpack(uint64_t extend_pointer, unsigned char extend_length, float &prob, float &rest) const {
      const LowerValue *lower;
      if (extend_length == 1) {
        lower = &unigram.Lookup(static_cast<uint64_t>(extend_pointer));
      } else {
        typename Middle::ConstIterator found;
        if (!middle_[extend_length - 2].Find(extend_pointer, found)) {
          std::cerr << "Extend pointer " << extend_pointer << " should have been found for length " << (unsigned) extend_length << std::endl;
          abort();
        }
        lower = &found->GetValue();
      }
      GetRest(*lower, prob, rest);
      return extend_pointer;
    }

    bool LookupMiddle(const Middle &middle, WordIndex word, float &backoff, Node &node, FullScoreReturn &ret) const {
      node = CombineWordHash(node, word);
      typename Middle::ConstIterator found;
      if (!middle.Find(node, found)) return false;
      util::FloatEnc enc;
      enc.f = found->GetValue().prob;
      ret.independent_left = (enc.i & util::kSignBit);
      ret.extend_left = node;
      enc.i |= util::kSignBit;
      ret.prob = enc.f;
      SetRest(found->GetValue(), ret);
      backoff = found->GetValue().backoff;
      return true;
    }

    void LoadedBinary();

    bool LookupMiddleNoProb(const Middle &middle, WordIndex word, float &backoff, Node &node) const {
      node = CombineWordHash(node, word);
      typename Middle::ConstIterator found;
      if (!middle.Find(node, found)) return false;
      backoff = found->GetValue().backoff;
      return true;
    }

    bool LookupLongest(WordIndex word, float &prob, Node &node) const {
      // Sign bit is always on because longest n-grams do not extend left.  
      node = CombineWordHash(node, word);
      typename Longest::ConstIterator found;
      if (!longest.Find(node, found)) return false;
      prob = found->GetValue().prob;
      return true;
    }

    // Geenrate a node without necessarily checking that it actually exists.  
    // Optionally return false if it's know to not exist.  
    bool FastMakeNode(const WordIndex *begin, const WordIndex *end, Node &node) const {
      assert(begin != end);
      node = static_cast<Node>(*begin);
      for (const WordIndex *i = begin + 1; i < end; ++i) {
        node = CombineWordHash(node, *i);
      }
      return true;
    }

  private:
    std::vector<Middle> middle_;
};

// std::identity is an SGI extension :-(
struct IdentityHash : public std::unary_function<uint64_t, size_t> {
  size_t operator()(uint64_t arg) const { return static_cast<size_t>(arg); }
};

struct ProbingHashedSearch : public TemplateHashedSearch<
  util::ProbingHashTable<util::ByteAlignedPacking<uint64_t, ProbBackoff>, IdentityHash>,
  util::ProbingHashTable<util::ByteAlignedPacking<uint64_t, Prob>, IdentityHash> > {

  static const ModelType kModelType = HASH_PROBING;
};

struct RestProbingHashedSearch : public TemplateHashedSearch<
  util::ProbingHashTable<util::ByteAlignedPacking<uint64_t, Rest>, IdentityHash>,
  util::ProbingHashTable<util::ByteAlignedPacking<uint64_t, Prob>, IdentityHash> > {

  static const ModelType kModelType = REST_HASH_PROBING;
};

} // namespace detail
} // namespace ngram
} // namespace lm

#endif // LM_SEARCH_HASHED__
