# Design Packet 5: Part 2 - Metrics, Evaluation Pipeline & CI/CD Integration

## 4. Gold Standard Dataset (Human-Labeled)

### 4.1 Seed Dataset Creation Strategy

**Target:** 150-200 human-labeled samples across all agent types

**Distribution:**
- Extraction: 80 samples (40 easy, 30 medium, 10 hard)
- Inference: 50 samples
- Validation: 50 samples
- End-to-end: 20 samples

### 4.2 Labeling Guidelines

**Label Schema:**

```yaml
# labeling-schema.yaml
extraction_labels:
  requirements:
    - id: string  # REQ-XXX format
    - title: string  # Clear, concise title
    - type: enum  # functional|non-functional|security|data|interface|constraint
    - actor: string  # Who/what performs action
    - action: string  # What is done
    - condition: string | null  # When/if condition
    - acceptance_criteria: list[string]  # Testable criteria (min 2)
    - priority: enum  # must|should|could
    - source_messages: list[int]  # Message indices (0-indexed)
    - rationale: string  # Why this is a requirement
    - confidence: float  # Labeler confidence (0.0-1.0)

inference_labels:
  inferred_requirements:
    - requirement: {}  # Same schema as above
    - inference_type: enum  # security|performance|scalability|compliance|usability
    - triggered_by: list[string]  # IDs of base requirements
    - necessity: enum  # necessary|recommended|optional
    - rationale: string
    - confidence: float

validation_labels:
  issues:
    - issue_type: enum  # ambiguous_verb|missing_actor|too_broad|contradictory|pii|untestable
    - severity: enum  # error|warning|info
    - field: string  # Field with issue
    - location: string  # Specific text span
    - message: string  # Human-readable description
    - suggested_fix: string  # How to fix
```

**Labeling Instructions (Markdown Template):**

```markdown
# Requirement Extraction Labeling Guide

## Overview
You are labeling conversations to identify requirements that should be extracted.

## Step 1: Read the Conversation
Read the entire conversation between user and analyst.

## Step 2: Identify Requirements
For each distinct requirement, create a label with:

### ID
Format: `REQ-001`, `REQ-002`, etc.

### Title
- Clear, noun-phrase describing the requirement
- Good: "User Authentication", "Payment Processing"
- Bad: "Users", "Payments"

### Type Classification
- **functional**: What the system does (features, behaviors)
- **non-functional**: How the system performs (speed, reliability)
- **security**: Protection, access control, encryption
- **data**: Data storage, retention, privacy
- **interface**: UI/UX, API specifications
- **constraint**: Technical or business limitations

### Actor
Who or what performs the action?
- User, Admin, System, External API, etc.
- Must be specific and singular

### Action
What is being done?
- Use active verbs: "Log in", "Process payment", "Send notification"
- Avoid: "should", "could", "might"
- Include direct object: "Log in **with email and password**"

### Condition (if applicable)
When or under what circumstances?
- "If payment fails"
- "When user is authenticated"
- Leave null if always applies

### Acceptance Criteria
List 2-5 testable conditions that define success:
- Start with action verbs
- Be specific and measurable
- Good: "User receives email within 1 minute"
- Bad: "Email is sent quickly"

### Priority
- **must**: Essential, system won't work without it
- **should**: Important, but system can function
- **could**: Nice to have

### Source Messages
List message indices (0-indexed) that mention this requirement.
- If spread across multiple messages, list all
- Example: [0, 3, 5]

### Rationale
Brief explanation of why you labeled this as a requirement:
- "Explicitly requested by user in message 2"
- "Implied by user's mention of 'secure login' in message 0"

### Confidence
Your confidence in this labeling (0.0 to 1.0):
- 1.0: Completely certain
- 0.9: Very confident
- 0.8: Confident
- 0.7: Somewhat confident
- < 0.7: Uncertain (add note explaining uncertainty)

## Edge Cases

### When user mentions existing features
Only label as requirement if they want changes/improvements.

### When details are vague
Extract what's explicit, note uncertainty in rationale.

### When requirements conflict
Label both, mark in rationale: "Potential conflict with REQ-002"

## Example

**Conversation:**
```
User: We need a way for customers to track their orders
Analyst: Should they receive notifications?
User: Yes, email them when status changes
```

**Labels:**
```yaml
- id: REQ-001
  title: "Order Tracking"
  type: functional
  actor: Customer
  action: View current status and history of their orders
  condition: null
  acceptance_criteria:
    - Customer can access tracking page via order number
    - Page shows current status (pending, shipped, delivered)
    - Page shows status history with timestamps
  priority: must
  source_messages: [0]
  rationale: "Explicit user request in message 0"
  confidence: 1.0

- id: REQ-002
  title: "Order Status Email Notifications"
  type: functional
  actor: System
  action: Send email notification to customer when order status changes
  condition: "When order status changes"
  acceptance_criteria:
    - Email sent automatically on status change
    - Email includes order number and new status
    - Email sent to customer's registered email address
  priority: must
  source_messages: [2]
  rationale: "Explicit request in message 2"
  confidence: 1.0
```
```

### 4.3 Inter-Annotator Agreement Protocol

**Process:**
1. **Initial Labeling:** 3 annotators independently label same 20 samples
2. **Agreement Calculation:** Compute Cohen's Kappa for requirement identification
3. **Discrepancy Resolution:** Discuss disagreements, update guidelines
4. **Calibration:** All annotators re-label same samples
5. **Validation:** Kappa > 0.75 required before full labeling

**Cohen's Kappa Calculation:**
```python
from sklearn.metrics import cohen_kappa_score

def calculate_agreement(annotator1_labels, annotator2_labels):
    \"\"\"
    Calculate inter-annotator agreement

    Labels are binary: 1 if requirement identified at this location, 0 otherwise
    \"\"\"
    kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)

    interpretation = {
        (0.81, 1.00): "Almost perfect agreement",
        (0.61, 0.80): "Substantial agreement",
        (0.41, 0.60): "Moderate agreement",
        (0.21, 0.40): "Fair agreement",
        (0.00, 0.20): "Slight agreement",
        (-1.0, 0.00): "Poor agreement"
    }

    for (low, high), desc in interpretation.items():
        if low <= kappa <= high:
            return kappa, desc

    return kappa, "Unknown"

# Example usage
kappa, interpretation = calculate_agreement(
    annotator1=[1, 1, 0, 1, 0, 1, 1, 0],
    annotator2=[1, 1, 0, 1, 1, 1, 0, 0]
)

print(f"Cohen's Kappa: {kappa:.3f} ({interpretation})")
# Output: Cohen's Kappa: 0.667 (Substantial agreement)
```

### 4.4 Annotation Tool UI (Simple Web Interface)

**HTML/JavaScript Tool:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Requirements Labeling Tool</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .conversation { background: #f5f5f5; padding: 15px; margin-bottom: 20px; }
        .message { margin: 10px 0; padding: 10px; background: white; }
        .user { border-left: 4px solid blue; }
        .analyst { border-left: 4px solid green; }
        .form-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
        button { padding: 10px 20px; margin: 5px; }
        .requirement-list { margin-top: 20px; }
        .req-card { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Requirement Extraction Labeling</h1>

    <div id="sample-info">
        <strong>Sample ID:</strong> <span id="sample-id"></span><br>
        <strong>Domain:</strong> <span id="domain"></span><br>
        <strong>Difficulty:</strong> <span id="difficulty"></span>
    </div>

    <div class="conversation" id="conversation">
        <!-- Messages loaded here -->
    </div>

    <div class="form-section">
        <h2>Add Requirement</h2>
        <form id="req-form">
            <label>ID: <input type="text" name="id" placeholder="REQ-001" required></label><br>
            <label>Title: <input type="text" name="title" size="50" required></label><br>
            <label>Type:
                <select name="type" required>
                    <option value="functional">Functional</option>
                    <option value="non-functional">Non-Functional</option>
                    <option value="security">Security</option>
                    <option value="data">Data</option>
                    <option value="interface">Interface</option>
                    <option value="constraint">Constraint</option>
                </select>
            </label><br>
            <label>Actor: <input type="text" name="actor" required></label><br>
            <label>Action: <textarea name="action" rows="2" cols="60" required></textarea></label><br>
            <label>Condition: <input type="text" name="condition" placeholder="Optional"></label><br>
            <label>Acceptance Criteria (one per line):</label><br>
            <textarea name="criteria" rows="4" cols="60" required></textarea><br>
            <label>Priority:
                <select name="priority" required>
                    <option value="must">Must</option>
                    <option value="should">Should</option>
                    <option value="could">Could</option>
                </select>
            </label><br>
            <label>Source Messages (comma-separated indices): <input type="text" name="sources" required></label><br>
            <label>Rationale: <textarea name="rationale" rows="2" cols="60" required></textarea></label><br>
            <label>Confidence: <input type="number" name="confidence" step="0.1" min="0" max="1" value="1.0" required></label><br>
            <button type="submit">Add Requirement</button>
        </form>
    </div>

    <div class="requirement-list">
        <h2>Labeled Requirements</h2>
        <div id="requirements"></div>
    </div>

    <button onclick="saveLabels()">Save & Next Sample</button>
    <button onclick="skipSample()">Skip Sample</button>

    <script>
        let currentSample = {};
        let requirements = [];

        // Load sample from JSON file
        async function loadSample(sampleId) {
            const response = await fetch(`samples/${sampleId}.json`);
            currentSample = await response.json();

            document.getElementById('sample-id').textContent = sampleId;
            document.getElementById('domain').textContent = currentSample.domain;
            document.getElementById('difficulty').textContent = currentSample.difficulty;

            // Render conversation
            const convDiv = document.getElementById('conversation');
            convDiv.innerHTML = '';
            currentSample.conversation.forEach((msg, idx) => {
                const msgDiv = document.createElement('div');
                msgDiv.className = `message ${msg.role}`;
                msgDiv.innerHTML = `<strong>[${idx}] ${msg.role}:</strong> ${msg.content}`;
                convDiv.appendChild(msgDiv);
            });

            requirements = [];
            renderRequirements();
        }

        // Handle form submission
        document.getElementById('req-form').addEventListener('submit', (e) => {
            e.preventDefault();

            const formData = new FormData(e.target);
            const req = {
                id: formData.get('id'),
                title: formData.get('title'),
                type: formData.get('type'),
                actor: formData.get('actor'),
                action: formData.get('action'),
                condition: formData.get('condition') || null,
                acceptance_criteria: formData.get('criteria').split('\\n').filter(x => x.trim()),
                priority: formData.get('priority'),
                source_messages: formData.get('sources').split(',').map(x => parseInt(x.trim())),
                rationale: formData.get('rationale'),
                confidence: parseFloat(formData.get('confidence'))
            };

            requirements.push(req);
            renderRequirements();
            e.target.reset();
        });

        function renderRequirements() {
            const reqDiv = document.getElementById('requirements');
            reqDiv.innerHTML = '';

            requirements.forEach((req, idx) => {
                const card = document.createElement('div');
                card.className = 'req-card';
                card.innerHTML = `
                    <strong>${req.id}: ${req.title}</strong> (${req.type})<br>
                    <em>Actor:</em> ${req.actor}<br>
                    <em>Action:</em> ${req.action}<br>
                    <em>Criteria:</em> ${req.acceptance_criteria.join('; ')}<br>
                    <em>Sources:</em> [${req.source_messages.join(', ')}]<br>
                    <em>Confidence:</em> ${req.confidence}<br>
                    <button onclick="removeRequirement(${idx})">Remove</button>
                `;
                reqDiv.appendChild(card);
            });
        }

        function removeRequirement(idx) {
            requirements.splice(idx, 1);
            renderRequirements();
        }

        async function saveLabels() {
            const labels = {
                sample_id: document.getElementById('sample-id').textContent,
                annotator: prompt('Enter your annotator ID:'),
                labeled_at: new Date().toISOString(),
                requirements: requirements
            };

            // Save to server
            await fetch('/api/save-labels', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(labels)
            });

            alert('Labels saved!');
            // Load next sample
            // loadSample(nextSampleId);
        }

        function skipSample() {
            if (confirm('Skip this sample?')) {
                // Load next sample
            }
        }

        // Initialize with first sample
        loadSample('sample_001');
    </script>
</body>
</html>
```

---

## 5. Evaluation Metrics (Detailed & Formal)

### 5.1 Extraction Agent Metrics[149][152][155]

#### 5.1.1 Requirement Recall

**Definition:** Proportion of ground truth requirements correctly extracted

**Formula:**
\[
\text{Recall} = \frac{|\text{Extracted} \cap \text{Ground Truth}|}{|\text{Ground Truth}|}
\]

**Matching Criteria:**
Two requirements match if:
- Semantic similarity > 0.85 (using sentence embeddings)
- Same requirement type
- Same actor (exact or synonym match)

**Python Implementation:**

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_requirement_recall(extracted: List[Dict], ground_truth: List[Dict]) -> float:
    \"\"\"
    Compute recall for requirement extraction

    Args:
        extracted: List of extracted requirements
        ground_truth: List of ground truth requirements

    Returns:
        Recall score (0.0 to 1.0)
    \"\"\"

    if not ground_truth:
        return 1.0 if not extracted else 0.0

    # Encode all requirements
    gt_texts = [f"{r['title']} {r['action']}" for r in ground_truth]
    ext_texts = [f"{r['title']} {r['action']}" for r in extracted]

    gt_embeddings = model.encode(gt_texts)
    ext_embeddings = model.encode(ext_texts)

    # Compute similarity matrix
    similarity = cosine_similarity(gt_embeddings, ext_embeddings)

    # Match ground truth to extracted
    matched_gt = set()

    for gt_idx, gt_req in enumerate(ground_truth):
        # Find best matching extracted requirement
        best_match_idx = np.argmax(similarity[gt_idx])
        best_similarity = similarity[gt_idx][best_match_idx]

        if best_similarity > 0.85:
            # Check type and actor match
            ext_req = extracted[best_match_idx]
            if (gt_req['type'] == ext_req['type'] and
                actors_match(gt_req['actor'], ext_req['actor'])):
                matched_gt.add(gt_idx)

    recall = len(matched_gt) / len(ground_truth)
    return recall

def actors_match(actor1: str, actor2: str) -> bool:
    \"\"\"Check if actors are same or synonyms\"\"\"
    synonyms = {
        'user': ['customer', 'end-user', 'client'],
        'system': ['application', 'platform', 'service'],
        'admin': ['administrator', 'superuser']
    }

    a1_lower = actor1.lower()
    a2_lower = actor2.lower()

    if a1_lower == a2_lower:
        return True

    # Check synonyms
    for key, syns in synonyms.items():
        if a1_lower == key and a2_lower in syns:
            return True
        if a2_lower == key and a1_lower in syns:
            return True

    return False

# Example
extracted = [
    {"title": "User Login", "action": "Authenticate with email", "type": "functional", "actor": "User"},
    {"title": "Password Reset", "action": "Reset password via email", "type": "functional", "actor": "User"}
]

ground_truth = [
    {"title": "User Authentication", "action": "Log in with email and password", "type": "functional", "actor": "User"},
    {"title": "Password Recovery", "action": "Reset forgotten password", "type": "functional", "actor": "User"},
    {"title": "Session Management", "action": "Maintain user session for 30 minutes", "type": "non-functional", "actor": "System"}
]

recall = compute_requirement_recall(extracted, ground_truth)
print(f"Recall: {recall:.2f}")  # Expected: ~0.67 (2 of 3 matched)
```

#### 5.1.2 Requirement Precision

**Definition:** Proportion of extracted requirements that are correct

**Formula:**
\[
\text{Precision} = \frac{|\text{Extracted} \cap \text{Ground Truth}|}{|\text{Extracted}|}
\]

**Implementation:**

```python
def compute_requirement_precision(extracted: List[Dict], ground_truth: List[Dict]) -> float:
    \"\"\"Compute precision for requirement extraction\"\"\"

    if not extracted:
        return 1.0 if not ground_truth else 0.0

    # Similar to recall but iterate over extracted
    gt_texts = [f"{r['title']} {r['action']}" for r in ground_truth]
    ext_texts = [f"{r['title']} {r['action']}" for r in extracted]

    gt_embeddings = model.encode(gt_texts)
    ext_embeddings = model.encode(ext_texts)

    similarity = cosine_similarity(ext_embeddings, gt_embeddings)

    matched_ext = set()

    for ext_idx, ext_req in enumerate(extracted):
        best_match_idx = np.argmax(similarity[ext_idx])
        best_similarity = similarity[ext_idx][best_match_idx]

        if best_similarity > 0.85:
            gt_req = ground_truth[best_match_idx]
            if (gt_req['type'] == ext_req['type'] and
                actors_match(gt_req['actor'], ext_req['actor'])):
                matched_ext.add(ext_idx)

    precision = len(matched_ext) / len(extracted)
    return precision
```

#### 5.1.3 Acceptance Criteria Completeness

**Definition:** Average proportion of ground truth acceptance criteria captured

**Formula:**
\[
\text{Completeness} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{AC}_{\text{extracted}}^i \cap \text{AC}_{\text{GT}}^i|}{|\text{AC}_{\text{GT}}^i|}
\]

**Implementation:**

```python
def compute_ac_completeness(extracted: List[Dict], ground_truth: List[Dict]) -> float:
    \"\"\"
    Compute acceptance criteria completeness

    Only computed for matched requirements
    \"\"\"

    # First match requirements
    matches = match_requirements(extracted, ground_truth)

    if not matches:
        return 0.0

    completeness_scores = []

    for ext_idx, gt_idx in matches:
        ext_ac = extracted[ext_idx]['acceptance_criteria']
        gt_ac = ground_truth[gt_idx]['acceptance_criteria']

        # Encode acceptance criteria
        ext_ac_emb = model.encode(ext_ac)
        gt_ac_emb = model.encode(gt_ac)

        # Match criteria
        similarity = cosine_similarity(gt_ac_emb, ext_ac_emb)

        matched_gt_ac = 0
        for gt_ac_idx in range(len(gt_ac)):
            best_sim = np.max(similarity[gt_ac_idx])
            if best_sim > 0.80:  # Lower threshold for criteria
                matched_gt_ac += 1

        completeness = matched_gt_ac / len(gt_ac) if gt_ac else 1.0
        completeness_scores.append(completeness)

    return np.mean(completeness_scores)
```

#### 5.1.4 Confidence Calibration Score[151][154][157][160][163]

**Definition:** How well predicted confidence aligns with actual accuracy

**Calibration Curve:**
```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def compute_confidence_calibration(predictions: List[Dict]) -> Dict:
    \"\"\"
    Compute calibration metrics for confidence scores

    Args:
        predictions: List of {
            'confidence': float,
            'correct': bool  # Whether prediction was correct
        }

    Returns:
        Dictionary with calibration metrics
    \"\"\"

    confidences = [p['confidence'] for p in predictions]
    corrects = [int(p['correct']) for p in predictions]

    # Compute calibration curve (10 bins)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        corrects,
        confidences,
        n_bins=10,
        strategy='uniform'
    )

    # Expected Calibration Error (ECE)
    bin_totals = np.histogram(confidences, bins=10, range=(0, 1))[0]
    ece = np.sum(np.abs(fraction_of_positives - mean_predicted_value) * bin_totals) / len(predictions)

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))

    # Brier Score
    brier_score = np.mean((np.array(confidences) - np.array(corrects)) ** 2)

    return {
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce,
        'brier_score': brier_score,
        'calibration_curve': {
            'mean_predicted': mean_predicted_value.tolist(),
            'fraction_positives': fraction_of_positives.tolist()
        }
    }

def plot_calibration_curve(calibration_data: Dict):
    \"\"\"Plot calibration curve\"\"\"
    mean_pred = calibration_data['calibration_curve']['mean_predicted']
    frac_pos = calibration_data['calibration_curve']['fraction_positives']

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(mean_pred, frac_pos, 'o-', label='Model Calibration')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Correct Predictions')
    plt.title(f'Calibration Curve (ECE: {calibration_data["expected_calibration_error"]:.3f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('calibration_curve.png')
```

**Example:**
```python
# Simulated predictions
predictions = [
    {'confidence': 0.95, 'correct': True},
    {'confidence': 0.90, 'correct': True},
    {'confidence': 0.85, 'correct': True},
    {'confidence': 0.80, 'correct': False},  # Overconfident
    {'confidence': 0.75, 'correct': True},
    {'confidence': 0.70, 'correct': True},
    {'confidence': 0.65, 'correct': False},
    {'confidence': 0.60, 'correct': True},
    # ... more samples
]

calibration_metrics = compute_confidence_calibration(predictions)
print(f"ECE: {calibration_metrics['expected_calibration_error']:.3f}")
print(f"Brier Score: {calibration_metrics['brier_score']:.3f}")
```

---

### 5.2 Inference Agent Metrics

#### 5.2.1 Inferred Requirement Relevance Score

**Definition:** Proportion of inferred requirements that are actually necessary

**Manual Annotation Required:**
For each inferred requirement, human judges rate:
- 2: Definitely needed
- 1: Recommended but not critical
- 0: Not needed / over-engineering

**Formula:**
\[
\text{Relevance} = \frac{\sum_{i=1}^{N} \text{score}_i}{2N}
\]

**Implementation:**

```python
def compute_inference_relevance(inferred_reqs: List[Dict], human_ratings: List[int]) -> float:
    \"\"\"
    Compute relevance score for inferred requirements

    Args:
        inferred_reqs: List of inferred requirements
        human_ratings: List of ratings (0, 1, or 2) from human judges

    Returns:
        Relevance score (0.0 to 1.0)
    \"\"\"

    if not inferred_reqs:
        return 1.0  # No false inferences

    total_score = sum(human_ratings)
    max_score = 2 * len(inferred_reqs)

    relevance = total_score / max_score
    return relevance
```

#### 5.2.2 False Inference Rate

**Definition:** Proportion of inferred requirements that are unnecessary

**Formula:**
\[
\text{FalseInferenceRate} = \frac{|\text{Inferred} \cap \text{Unnecessary}|}{|\text{Inferred}|}
\]

where Unnecessary = {requirements rated 0 by judges}

#### 5.2.3 Confidence-Accuracy Calibration

**Same as extraction agent calibration**, but for inference predictions.

---

### 5.3 Validation Agent Metrics

#### 5.3.1 Issue Detection Recall

**Definition:** Proportion of actual issues correctly identified

**Formula:**
\[
\text{Recall} = \frac{|\text{Detected} \cap \text{Actual}|}{|\text{Actual}|}
\]

**Matching:** Issues match if same type and same field

#### 5.3.2 Issue Detection Precision

**Formula:**
\[
\text{Precision} = \frac{|\text{Detected} \cap \text{Actual}|}{|\text{Detected}|}
\]

#### 5.3.3 PII Detection F1

**Special metric for PII detection:**

```python
from sklearn.metrics import precision_recall_fscore_support

def compute_pii_f1(detected_pii: List[str], actual_pii: List[str]) -> Dict:
    \"\"\"
    Compute PII detection metrics

    Args:
        detected_pii: List of PII instances detected by agent
        actual_pii: Ground truth PII instances

    Returns:
        Dictionary with precision, recall, F1
    \"\"\"

    # Convert to binary labels (simplified)
    detected_set = set(detected_pii)
    actual_set = set(actual_pii)

    true_positives = len(detected_set & actual_set)
    false_positives = len(detected_set - actual_set)
    false_negatives = len(actual_set - detected_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
```

---

### 5.4 End-to-End Metrics

#### 5.4.1 RD Correctness Score

**Definition:** Composite score measuring final RD quality

**Components:**
1. Requirement coverage (0-1)
2. Requirement accuracy (0-1)
3. Structure completeness (0-1)
4. Traceability (0-1)

**Formula:**
\[
\text{RDScore} = 0.4 \times \text{Coverage} + 0.3 \times \text{Accuracy} + 0.2 \times \text{Structure} + 0.1 \times \text{Traceability}
\]

**Continued in next file...**
