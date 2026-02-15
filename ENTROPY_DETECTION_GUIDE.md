# Entropy-Based Hallucination Detection

## 🎯 Key Insights

**Insight #1**: "If the model has a very low probability for specific tokens (like specific numbers or proper nouns), it is likely 'guessing' based on language patterns rather than knowledge."

**Insight #2**: "A sudden drop in confidence mid-sentence usually marks the exact point where a hallucination begins." ← **Softmax Bottleneck**

**Both insights are now integrated!** This is called **entropy-based hallucination detection with bottleneck analysis**.

## 🔥 Softmax Bottlenecks: The Smoking Gun

### What is a Softmax Bottleneck?

When an LLM transitions from **grounded content** (from context) to **hallucinated content** (made up), there's a sudden drop in token probability. This marks the **exact transition point**.

**Example**:
```
Context: "TechCorp runs A/B tests for several weeks."

Generated Answer:
"TechCorp runs tests for 2.5 weeks with 10,247 users."
                     ↑
                 Bottleneck here!
```

**Token Probabilities**:
```
Token         Probability    Analysis
"TechCorp"    0.92          ✅ From context, high confidence
"runs"        0.88          ✅ From context
"tests"       0.91          ✅ From context
"for"         0.87          ✅ From context
"2"           0.15          🚨 BOTTLENECK! Sudden drop from 0.87 to 0.15
"."           0.12          ❌ Hallucination continues
"5"           0.08          ❌ Still hallucinating
"weeks"       0.78          ✅ Back to grounded (general word)
"with"        0.82          ✅
"10"          0.06          🚨 Another bottleneck!
","           0.05          ❌
"247"         0.04          ❌ Still guessing the number
"users"       0.79          ✅ Back to grounded
```

**Detection**:
- **Bottleneck #1**: Position 5 - confidence drops from 0.87 → 0.15 (82% drop)
- **Bottleneck #2**: Position 10 - confidence drops from 0.82 → 0.06 (93% drop)

**Conclusion**: Model started hallucinating at position 5 ("2") and again at position 10 ("10")!

### Why Bottlenecks Occur

1. **Grounded tokens** (from context): High probability (~0.8-0.95)
2. **Transition point**: Model realizes it doesn't have the specific information
3. **Hallucinated tokens**: Low probability (~0.05-0.15) because model is guessing
4. **Return to grounded**: Probability recovers for general words

### Visual Representation

```
Confidence over sequence:

1.0 |  ██████████
    |  ██████████         ████████
0.8 |  ██████████    ███  ████████████
    |  ██████████   ████  ████████████
0.6 |  ██████████  █████  ████████████
    |              ██████
0.4 |             ███████
    |            ████████
0.2 |           █████████      ███
    |          ██████████     ████
0.0 |_________________________________
    Context    |   Hallucination  |
    tokens     ↑   (guessing)     ↑
          Bottleneck #1      Bottleneck #2
```

## 🧠 Why This Works

### Token Probability Reveals Confidence

When an LLM generates text, each token has a probability:

```
Context: "Machine learning models need training data."

High Confidence (from context):
"Machine" → 0.95 probability ✅
"learning" → 0.92 probability ✅
"models" → 0.88 probability ✅

Low Confidence (hallucinated):
"10,247" → 0.03 probability ❌ (guessing!)
"samples" → 0.08 probability ❌ (uncertain)
```

**The model KNOWS it's guessing** - the low probability proves it!

### Entropy = Uncertainty

**Shannon Entropy**: H = -Σ(p * log₂(p))

- **Low entropy** (1-2 bits): Model is confident, likely has knowledge
- **High entropy** (3-4 bits): Model is uncertain, likely hallucinating
- **Very high entropy** (>4 bits): Model is essentially guessing randomly

### Specific Tokens Are Red Flags

**High-Risk Tokens** (often low probability when hallucinated):
1. **Specific numbers**: "10,247", "37.6%", "$4,295"
2. **Proper nouns**: "Dr. Smith", "TechCorp Labs"
3. **Dates**: "March 15, 2023"
4. **Technical terms**: "photosynthetic coefficient"
5. **Quotes**: Exact wording of statements

**Why?** These require exact factual knowledge. If not in context, model must guess, resulting in low probability.

## 🔍 Implementation

### 1. Request Token Probabilities

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    logprobs=True,        # Get log probabilities
    top_logprobs=5        # Get top 5 alternatives
)

# Extract probabilities
for token_info in response.choices[0].logprobs.content:
    token = token_info.token
    logprob = token_info.logprob
    probability = exp(logprob)  # Convert to probability
    
    if probability < 0.1:  # Low probability!
        print(f"⚠️ Suspicious: '{token}' has only {probability:.1%} confidence")
```

### 2. Calculate Entropy

```python
# For each token, calculate entropy
probabilities = [exp(logprob) for logprob in token_logprobs]
entropy = -sum(p * log2(p) for p in probabilities)

if entropy > 3.0:
    print("⚠️ High entropy detected - likely hallucination!")
```

### 3. Find Suspicious Spans

```python
# Look for consecutive low-probability tokens
suspicious_spans = []

for i, tokens in enumerate(sliding_window(tokens, size=3)):
    avg_prob = mean([t.probability for t in tokens])
    if avg_prob < 0.1:
        suspicious_spans.append({
            'tokens': [t.text for t in tokens],
            'avg_probability': avg_prob
        })

# Example output:
# Suspicious: ["10", ",", "247"] with avg probability 0.04
```

## 📊 Real-World Example

### Scenario: Model Hallucinates Specific Data

**Context**:
> "TechCorp uses A/B testing. Tests run for several weeks with thousands of users."

**Question**: "How long does TechCorp run tests?"

**Hallucinated Answer**:
> "TechCorp runs A/B tests for exactly 2.5 weeks with 12,847 users."

**Entropy Analysis**:
```
Token          Probability   Flag
"TechCorp"     0.92         ✅ (from context)
"runs"         0.88         ✅ (from context)
"tests"        0.91         ✅ (from context)
"for"          0.87         ✅
"exactly"      0.45         ⚠️  (hedging word, medium confidence)
"2"            0.08         🚨 (VERY LOW - hallucinated!)
"."            0.12         🚨 (specific decimal)
"5"            0.06         🚨 (specific digit)
"weeks"        0.78         ✅ (from context)
"with"         0.82         ✅
"12"           0.04         🚨 (VERY LOW - hallucinated!)
","            0.05         🚨
"847"          0.03         🚨 (VERY LOW - hallucinated!)
"users"        0.79         ✅ (from context)

Entropy: 4.2 bits (HIGH!)
Avg Probability: 0.52
Min Probability: 0.03 (VERY SUSPICIOUS!)

Suspicious Spans:
  1. ["2", ".", "5"] - avg prob: 0.09
  2. ["12", ",", "847"] - avg prob: 0.04
```

**Conclusion**: Model is **guessing** the specific numbers! High entropy confirms hallucination.

### Contrast: Grounded Answer

**Grounded Answer**:
> "TechCorp runs A/B tests for several weeks with thousands of users [source: ab_testing_guide]."

**Entropy Analysis**:
```
Token          Probability   Flag
"TechCorp"     0.92         ✅
"runs"         0.88         ✅
"tests"        0.91         ✅
"for"          0.87         ✅
"several"      0.89         ✅ (vague, appropriate)
"weeks"        0.91         ✅
"with"         0.88         ✅
"thousands"    0.87         ✅ (vague, appropriate)
"of"           0.92         ✅
"users"        0.90         ✅

Entropy: 1.8 bits (LOW!)
Avg Probability: 0.89
Min Probability: 0.87

No suspicious spans detected.
```

**Conclusion**: Model is **confident** because it's using context, not guessing!

## 🔍 Bottleneck Detection Algorithm

### Step 1: Calculate Probability Deltas

```python
# For each consecutive pair of tokens
deltas = []
for i in range(1, len(probabilities)):
    delta = probabilities[i] - probabilities[i-1]
    deltas.append(delta)

# Example:
# Token:  "for"  →  "2"
# Prob:   0.87   →  0.15
# Delta:  -0.72  (HUGE DROP!)
```

### Step 2: Identify Bottleneck Criteria

A bottleneck is detected when **ALL** of:

1. **Significant drop**: Delta < -0.3 OR relative drop > 50%
2. **Drops to low confidence**: Current probability < 0.3
3. **Was previously confident**: Previous probability > 0.6

```python
def is_bottleneck(prev_prob, curr_prob):
    delta = curr_prob - prev_prob
    
    # Check all three conditions
    significant_drop = (
        delta < -0.3 or 
        (prev_prob > 0 and abs(delta/prev_prob) > 0.5)
    )
    
    drops_to_low = curr_prob < 0.3
    was_confident = prev_prob > 0.6
    
    return significant_drop and drops_to_low and was_confident
```

### Step 3: Extract Context Around Bottleneck

```python
if is_bottleneck(prev_prob, curr_prob):
    bottleneck = {
        'position': i,
        'context_before': ' '.join(tokens[i-3:i]),
        'bottleneck_token': tokens[i],
        'context_after': ' '.join(tokens[i+1:i+4]),
        'confidence_drop': abs(delta),
        'relative_drop_pct': abs(delta/prev_prob) * 100
    }
```

### Step 4: Rank by Severity

```python
# Sort bottlenecks by confidence drop
bottlenecks.sort(key=lambda x: x['confidence_drop'], reverse=True)

# Most severe bottleneck = most likely hallucination start point
primary_bottleneck = bottlenecks[0]
```

## 🎯 Detection Strategy

### Enhanced Confidence Scoring (with Bottlenecks)

```python
def calculate_confidence(token_probs, bottlenecks, context_score):
    confidence = 0.0
    
    # Tier 1: Token Probability (30%)
    avg_prob = mean(token_probs)
    if avg_prob > 0.7:
        confidence += 0.3
    elif avg_prob > 0.5:
        confidence += 0.15
    
    # Tier 2: Bottleneck Analysis (30%) ← NEW!
    if len(bottlenecks) == 0:
        confidence += 0.3  # No bottlenecks = good
    elif len(bottlenecks) == 1 and bottlenecks[0]['confidence_drop'] < 0.5:
        confidence += 0.15  # Minor bottleneck
    # Multiple or severe bottlenecks = 0 points
    
    # Tier 3: Entropy (20%)
    entropy = calculate_entropy(token_probs)
    if entropy < 2.0:
        confidence += 0.2
    elif entropy < 3.0:
        confidence += 0.1
    
    # Tier 4: Context Quality (15%)
    confidence += 0.15 * min(context_score, 1.0)
    
    # Tier 5: Citations (5%)
    if has_citations:
        confidence += 0.05
    
    return confidence
```

### Three-Tier Confidence Scoring

```python
def calculate_confidence(token_probs, context_score, has_citations):
    confidence = 0.0
    
    # Tier 1: Token Probability (40%)
    avg_prob = mean(token_probs)
    if avg_prob > 0.7:
        confidence += 0.4
    elif avg_prob > 0.5:
        confidence += 0.2
    
    # Tier 2: Entropy (30%)
    entropy = calculate_entropy(token_probs)
    if entropy < 2.0:
        confidence += 0.3
    elif entropy < 3.0:
        confidence += 0.15
    
    # Tier 3: Context Quality (20%)
    confidence += 0.2 * min(context_score, 1.0)
    
    # Tier 4: Citations (10%)
    if has_citations:
        confidence += 0.1
    
    return confidence
```

### Severity Classification

```python
if min_token_probability < 0.01:
    severity = "SEVERE"     # Model is basically guessing
elif entropy > 4.0:
    severity = "SEVERE"     # Very high uncertainty
elif entropy > 3.0:
    severity = "MODERATE"   # Moderate uncertainty
else:
    severity = "MINOR"      # Low uncertainty
```

## 🚨 Red Flags to Watch

### Critical Indicators

1. **Softmax Bottlenecks Detected** ← **HIGHEST PRIORITY**
   - Sudden confidence drop (>50%) mid-sentence
   - Marks exact start of hallucination
   - Example: 0.87 → 0.15 when generating specific number

2. **Min Token Probability < 0.05**
   - Model assigned <5% probability to at least one token
   - Almost certainly hallucinated

2. **Entropy > 3.5 bits**
   - High overall uncertainty
   - Model is not confident

3. **Suspicious Spans (3+ consecutive low-prob tokens)**
   - Specific numbers, names, or facts
   - Example: "Dr. John Smith at Stanford University"

4. **Low-Probability Proper Nouns**
   - Names of people, companies, places
   - If not in context, likely fabricated

5. **Specific Numerical Claims**
   - Precise percentages, amounts, dates
   - Example: "37.6%" vs "approximately 38%"

## 🛠️ Practical Implementation

### Step 1: Enable Logprobs in API Calls

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    logprobs=True,       # ← Enable this!
    top_logprobs=5       # ← Get alternatives
)
```

### Step 2: Analyze Token Probabilities

```python
import numpy as np

def analyze_tokens(logprobs_data):
    high_risk_tokens = []
    
    for token_info in logprobs_data.content:
        token = token_info.token
        prob = np.exp(token_info.logprob)
        
        # Flag low-probability tokens
        if prob < 0.1:
            high_risk_tokens.append({
                'token': token,
                'probability': prob,
                'is_number': token.isdigit(),
                'is_proper_noun': token[0].isupper()
            })
    
    return high_risk_tokens
```

### Step 3: Calculate Confidence Score

```python
def calculate_hallucination_risk(tokens, entropy, context_score):
    # Low probabilities = high risk
    min_prob = min(token['probability'] for token in tokens)
    
    risk_score = 0
    
    if min_prob < 0.05:
        risk_score += 40  # Critical
    elif min_prob < 0.1:
        risk_score += 25  # High
    
    if entropy > 3.5:
        risk_score += 30  # Critical
    elif entropy > 2.5:
        risk_score += 15  # Moderate
    
    if context_score < 0.5:
        risk_score += 20  # Poor retrieval
    
    if risk_score > 50:
        return "HIGH_RISK"
    elif risk_score > 30:
        return "MODERATE_RISK"
    else:
        return "LOW_RISK"
```

## 📈 Benefits of Entropy-Based Detection

### Advantages

1. **✅ Catches Specific Hallucinations**
   - Numbers, dates, names that LLM-only verification might miss

2. **✅ Model Self-Awareness**
   - Uses the model's own uncertainty as signal

3. **✅ No Additional LLM Calls**
   - Logprobs come with same API call

4. **✅ Language-Agnostic**
   - Works for any language

5. **✅ Quantifiable**
   - Entropy is a numerical measure

### Limitations

1. **❌ API Support Required**
   - Not all LLM APIs provide logprobs

2. **❌ Calibration Needed**
   - Thresholds may vary by model

3. **❌ Doesn't Catch All Types**
   - Confident but wrong (model trained on misinformation)

## 🎯 Best Practices

### 1. Combine Multiple Signals

```python
def detect_hallucination(answer, context):
    # Signal 1: LLM verification
    llm_check = verify_with_llm(answer, context)
    
    # Signal 2: Entropy analysis
    entropy_check = analyze_entropy(answer)
    
    # Signal 3: Pattern detection
    pattern_check = check_patterns(answer)
    
    # Signal 4: Context similarity
    similarity_check = compute_similarity(answer, context)
    
    # Weighted combination
    confidence = (
        0.3 * llm_check +
        0.3 * entropy_check +
        0.2 * pattern_check +
        0.2 * similarity_check
    )
    
    return confidence > 0.6
```

### 2. Monitor Over Time

```python
# Track entropy trends
entropy_history = []

for query, answer in query_answer_pairs:
    entropy = calculate_entropy(answer)
    entropy_history.append(entropy)

avg_entropy = mean(entropy_history)
print(f"System average entropy: {avg_entropy:.2f} bits")

if avg_entropy > 3.0:
    print("⚠️ System generating high-entropy answers!")
    print("Consider: improving retrieval, more context, better prompting")
```

### 3. Use as Early Warning

```python
if min_token_probability < 0.05:
    # Don't return answer to user
    return "I don't have enough information to answer confidently."
```

## 🎉 Summary

**Entropy-based detection works because**:
1. LLMs assign probabilities to each generated token
2. Low probability = model is uncertain/guessing
3. High entropy = model is not confident
4. **Sudden confidence drops = exact hallucination start point** ← NEW!
5. Specific numbers/names with low probability = likely hallucinated

**Bottleneck Detection**:
- Monitors probability trajectory over the sequence
- Detects sudden drops (>50% or >0.3 absolute)
- Pinpoints exact token where hallucination begins
- Provides context before and after the bottleneck

**Implementation**: Request `logprobs=True` in API calls, analyze token probabilities, detect confidence drops, flag high-entropy spans

**Result**: Catch hallucinations AND identify exactly where they start! 🚀

## 📊 Real-World Detection Example

### Input
```
Context: "Our company uses machine learning for predictions."
Question: "What's our ML model accuracy?"
Generated: "Our ML model achieves 94.7% accuracy with 50,000 training samples."
```

### Analysis Output
```
🔍 Hallucination Analysis:

Has Hallucination: TRUE
Confidence Score: 0.32 (LOW)
Severity: SEVERE

📊 Entropy Metrics:
  Avg Token Probability: 0.58
  Min Token Probability: 0.04
  Entropy: 3.8 bits (HIGH)
  High Entropy Tokens: 7

🚨 SOFTMAX BOTTLENECKS DETECTED: 2

Bottleneck #1:
  Position: 6
  Before: "model achieves"
  Drop at: "94"
  After: ". 7 %"
  Confidence: 0.85 → 0.12 (Drop: -0.73, 86%)
  ⚠️ HALLUCINATION STARTS HERE

Bottleneck #2:
  Position: 12
  Before: "accuracy with"
  Drop at: "50"
  After: ", 000 training"
  Confidence: 0.82 → 0.08 (Drop: -0.74, 90%)
  ⚠️ ANOTHER HALLUCINATION

Root Causes:
  - SOFTMAX_BOTTLENECK_DETECTED
  - MULTIPLE_CONFIDENCE_DROPS_2
  - VERY_LOW_TOKEN_CONFIDENCE
  - HIGH_ENTROPY_GENERATION
  - MISSING_CITATIONS
```

### Interpretation
1. **First bottleneck** at "94" - Model doesn't know the actual accuracy, starts guessing
2. **Second bottleneck** at "50" - Model doesn't know training set size, starts guessing again
3. **High entropy** (3.8 bits) confirms overall uncertainty
4. **Low min probability** (0.04) on specific digits reveals guessing

### Action Taken
```python
if len(bottlenecks) > 0:
    # Don't return hallucinated answer
    return "I don't have information about our model's specific accuracy. Please check the model documentation."
```
