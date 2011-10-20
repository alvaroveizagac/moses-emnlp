#ifndef LM_WEIGHTS__
#define LM_WEIGHTS__

// Weights for n-grams.  Probability and possibly a backoff.  

namespace lm {
struct Prob {
  float prob;
};
// No inheritance so this will be a POD.  
struct ProbBackoff {
  float prob;
  float backoff;

  float GetRest() const {
    return prob;
  }
};

struct Rest : ProbBackoff {
  float rest;

  float GetRest() const {
    return rest;
  }
};

} // namespace lm
#endif // LM_WEIGHTS__
