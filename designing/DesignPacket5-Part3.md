# Design Packet 5: Part 3 - Evaluation Pipeline, CI/CD & Tools

## 6. Evaluation Pipeline Implementation

### 6.1 Evaluation Runner (Complete Python CLI)

```python
#!/usr/bin/env python3
# eval/evaluation_runner.py

import click
import yaml
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import uuid
from tqdm import tqdm

from src.agents.extraction.agent import ExtractionAgent
from src.agents.inference.agent import InferenceAgent
from src.agents.validation.agent import ValidationAgent
from eval.scorers.extraction_scorer import ExtractionScorer
from eval.scorers.inference_scorer import InferenceScorer
from eval.scorers.validation_scorer import ValidationScorer
from eval.report_generator import ReportGenerator
from eval.drift_detector import DriftDetector


class EvaluationRunner:
    \"\"\"Main evaluation orchestrator\"\"\"

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.run_id = str(uuid.uuid4())
        self.results = []

    async def run_extraction_evaluation(self) -> Dict:
        \"\"\"Run evaluation for extraction agent\"\"\"

        print("\\n=== Extraction Agent Evaluation ===\")

        # Load dataset
        dataset_config = self.config['datasets']['extraction']
        dataset = self.load_dataset(dataset_config)

        # Initialize agent in eval mode
        agent = ExtractionAgent(
            temperature=self.config['agents']['extraction']['temperature'],
            cache_enabled=True
        )

        # Initialize scorer
        scorer = ExtractionScorer()

        results = []

        for sample in tqdm(dataset, desc="Evaluating extraction"):
            # Run agent
            try:
                prediction = await agent.extract_requirements(
                    conversation=sample['conversation'],
                    deterministic=True
                )

                # Score prediction
                scores = scorer.score(
                    prediction=prediction,
                    ground_truth=sample['ground_truth_requirements']
                )

                results.append({
                    'sample_id': sample['id'],
                    'prediction': prediction,
                    'ground_truth': sample['ground_truth_requirements'],
                    'scores': scores,
                    'passed': self.check_thresholds(scores, 'extraction')
                })

            except Exception as e:
                results.append({
                    'sample_id': sample['id'],
                    'error': str(e),
                    'passed': False
                })

        # Aggregate scores
        aggregate_scores = scorer.aggregate(results)

        return {
            'agent': 'extraction',
            'num_samples': len(dataset),
            'aggregate_scores': aggregate_scores,
            'results': results
        }

    async def run_inference_evaluation(self) -> Dict:
        \"\"\"Run evaluation for inference agent\"\"\"

        print("\\n=== Inference Agent Evaluation ===\")

        dataset_config = self.config['datasets']['inference']
        dataset = self.load_dataset(dataset_config)

        agent = InferenceAgent(
            temperature=self.config['agents']['inference']['temperature']
        )

        scorer = InferenceScorer()

        results = []

        for sample in tqdm(dataset, desc="Evaluating inference"):
            try:
                # Agent infers requirements from base requirements
                prediction = await agent.infer_requirements(
                    base_requirements=sample['input']['explicit_requirements'],
                    domain=sample['input']['domain']
                )

                scores = scorer.score(
                    prediction=prediction,
                    ground_truth=sample['ground_truth']['inferred_requirements']
                )

                results.append({
                    'sample_id': sample['id'],
                    'prediction': prediction,
                    'ground_truth': sample['ground_truth'],
                    'scores': scores,
                    'passed': self.check_thresholds(scores, 'inference')
                })

            except Exception as e:
                results.append({
                    'sample_id': sample['id'],
                    'error': str(e),
                    'passed': False
                })

        aggregate_scores = scorer.aggregate(results)

        return {
            'agent': 'inference',
            'num_samples': len(dataset),
            'aggregate_scores': aggregate_scores,
            'results': results
        }

    async def run_validation_evaluation(self) -> Dict:
        \"\"\"Run evaluation for validation agent\"\"\"

        print("\\n=== Validation Agent Evaluation ===\")

        dataset_config = self.config['datasets']['validation']
        dataset = self.load_dataset(dataset_config)

        agent = ValidationAgent(
            temperature=self.config['agents']['validation']['temperature']
        )

        scorer = ValidationScorer()

        results = []

        for sample in tqdm(dataset, desc="Evaluating validation"):
            try:
                prediction = await agent.validate_requirement(
                    requirement=sample['flawed_requirement']
                )

                scores = scorer.score(
                    prediction=prediction,
                    ground_truth=sample['validation_issues']
                )

                results.append({
                    'sample_id': sample['id'],
                    'prediction': prediction,
                    'ground_truth': sample['validation_issues'],
                    'scores': scores,
                    'passed': self.check_thresholds(scores, 'validation')
                })

            except Exception as e:
                results.append({
                    'sample_id': sample['id'],
                    'error': str(e),
                    'passed': False
                })

        aggregate_scores = scorer.aggregate(results)

        return {
            'agent': 'validation',
            'num_samples': len(dataset),
            'aggregate_scores': aggregate_scores,
            'results': results
        }

    def load_dataset(self, config: Dict) -> List[Dict]:
        \"\"\"Load dataset from JSONL file\"\"\"

        dataset_path = Path(config['path'])

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = []
        with open(dataset_path) as f:
            for line in f:
                data.append(json.loads(line))

        # Apply split and size filters
        split = config.get('split', 'test')
        size = config.get('size')

        data = [d for d in data if d.get('split') == split]

        if size:
            data = data[:size]

        return data

    def check_thresholds(self, scores: Dict, agent_type: str) -> bool:
        \"\"\"Check if scores meet configured thresholds\"\"\"

        thresholds = self.config.get('thresholds', {})

        threshold_map = {
            'extraction': [
                ('precision', 'extraction_precision'),
                ('recall', 'extraction_recall')
            ],
            'inference': [
                ('false_inference_rate', 'inference_false_rate')
            ],
            'validation': [
                ('recall', 'validation_recall')
            ]
        }

        for score_key, threshold_key in threshold_map.get(agent_type, []):
            if score_key in scores:
                threshold = thresholds.get(threshold_key)
                if threshold:
                    # For false rates, score should be BELOW threshold
                    if 'false' in score_key or 'error' in score_key:
                        if scores[score_key] > threshold:
                            return False
                    else:
                        if scores[score_key] < threshold:
                            return False

        return True

    async def run_full_evaluation(self):
        \"\"\"Run evaluation for all agents\"\"\"

        start_time = datetime.utcnow()

        print(f"Starting evaluation run: {self.run_id}")
        print(f"Config: {self.config['name']}")

        results = {
            'run_id': self.run_id,
            'config': self.config['name'],
            'started_at': start_time.isoformat(),
            'git_commit': self.get_git_commit(),
            'agents': {}
        }

        # Run each agent evaluation
        if 'extraction' in self.config['datasets']:
            results['agents']['extraction'] = await self.run_extraction_evaluation()

        if 'inference' in self.config['datasets']:
            results['agents']['inference'] = await self.run_inference_evaluation()

        if 'validation' in self.config['datasets']:
            results['agents']['validation'] = await self.run_validation_evaluation()

        # Generate report
        end_time = datetime.utcnow()
        results['completed_at'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()

        # Save results
        self.save_results(results)

        # Generate HTML report
        report_gen = ReportGenerator(self.config)
        report_path = report_gen.generate(results)

        print(f"\\nEvaluation complete!")
        print(f"Report: {report_path}")

        # Check if all agents passed thresholds
        all_passed = all(
            all(r.get('passed', False) for r in agent_results['results'])
            for agent_results in results['agents'].values()
        )

        if not all_passed:
            print("\\n⚠️  Some evaluations failed to meet thresholds")
            return 1
        else:
            print("\\n✅ All evaluations passed")
            return 0

    def save_results(self, results: Dict):
        \"\"\"Save results to JSON file\"\"\"

        output_dir = Path(self.config['reporting']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"eval_{self.run_id}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved: {output_file}")

    def get_git_commit(self) -> Optional[str]:
        \"\"\"Get current git commit hash\"\"\"
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except:
            return None


@click.group()
def cli():
    \"\"\"Requirements Engineering Agent Evaluation CLI\"\"\"
    pass


@cli.command()
@click.option('--config', '-c', default='eval/configs/full-pipeline-eval.yaml', help='Evaluation config file')
def run(config):
    \"\"\"Run full evaluation pipeline\"\"\"

    runner = EvaluationRunner(config)
    exit_code = asyncio.run(runner.run_full_evaluation())
    exit(exit_code)


@cli.command()
@click.option('--config', '-c', required=True, help='Evaluation config file')
@click.option('--agent', '-a', type=click.Choice(['extraction', 'inference', 'validation']), required=True)
def run_agent(config, agent):
    \"\"\"Run evaluation for specific agent\"\"\"

    runner = EvaluationRunner(config)

    if agent == 'extraction':
        result = asyncio.run(runner.run_extraction_evaluation())
    elif agent == 'inference':
        result = asyncio.run(runner.run_inference_evaluation())
    elif agent == 'validation':
        result = asyncio.run(runner.run_validation_evaluation())

    print(json.dumps(result['aggregate_scores'], indent=2))


@cli.command()
@click.argument('baseline_file')
@click.argument('current_file')
@click.option('--threshold', '-t', default=0.03, help='Drift detection threshold')
def detect_drift(baseline_file, current_file, threshold):
    \"\"\"Detect performance drift between runs\"\"\"

    detector = DriftDetector(threshold=threshold)

    with open(baseline_file) as f:
        baseline = json.load(f)

    with open(current_file) as f:
        current = json.load(f)

    drift_report = detector.detect(baseline, current)

    print(json.dumps(drift_report, indent=2))

    if drift_report['has_drift']:
        print("\\n⚠️  Performance drift detected!")
        exit(1)
    else:
        print("\\n✅ No significant drift")
        exit(0)


@cli.command()
@click.argument('results_file')
def report(results_file):
    \"\"\"Generate HTML report from results JSON\"\"\"

    with open(results_file) as f:
        results = json.load(f)

    # Dummy config for report generation
    config = {'reporting': {'output_dir': 'reports/', 'formats': ['html']}}

    report_gen = ReportGenerator(config)
    report_path = report_gen.generate(results)

    print(f"Report generated: {report_path}")


if __name__ == '__main__':
    cli()
```

### 6.2 Scorer Implementations

**Extraction Scorer:**

```python
# eval/scorers/extraction_scorer.py

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ExtractionScorer:
    \"\"\"Score extraction agent predictions\"\"\"

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def score(self, prediction: List[Dict], ground_truth: List[Dict]) -> Dict:
        \"\"\"
        Score a single extraction prediction

        Returns:
            Dictionary with all metrics
        \"\"\"

        scores = {}

        # Recall
        scores['recall'] = self.compute_recall(prediction, ground_truth)

        # Precision
        scores['precision'] = self.compute_precision(prediction, ground_truth)

        # F1
        if scores['precision'] + scores['recall'] > 0:
            scores['f1'] = 2 * (scores['precision'] * scores['recall']) / (scores['precision'] + scores['recall'])
        else:
            scores['f1'] = 0.0

        # Type accuracy (of matched requirements)
        scores['type_accuracy'] = self.compute_type_accuracy(prediction, ground_truth)

        # Acceptance criteria completeness
        scores['ac_completeness'] = self.compute_ac_completeness(prediction, ground_truth)

        # Confidence calibration (requires correctness labels)
        # Will be computed in aggregate

        return scores

    def compute_recall(self, prediction: List[Dict], ground_truth: List[Dict]) -> float:
        \"\"\"Compute requirement recall\"\"\"

        if not ground_truth:
            return 1.0 if not prediction else 0.0

        matches = self.match_requirements(prediction, ground_truth)
        return len(matches) / len(ground_truth)

    def compute_precision(self, prediction: List[Dict], ground_truth: List[Dict]) -> float:
        \"\"\"Compute requirement precision\"\"\"

        if not prediction:
            return 1.0 if not ground_truth else 0.0

        matches = self.match_requirements(prediction, ground_truth)
        return len(matches) / len(prediction)

    def match_requirements(self, prediction: List[Dict], ground_truth: List[Dict]) -> List[tuple]:
        \"\"\"
        Match predicted requirements to ground truth

        Returns:
            List of (pred_idx, gt_idx) tuples
        \"\"\"

        if not prediction or not ground_truth:
            return []

        # Encode requirements
        pred_texts = [f"{r['title']} {r['action']}" for r in prediction]
        gt_texts = [f"{r['title']} {r['action']}" for r in ground_truth]

        pred_emb = self.model.encode(pred_texts)
        gt_emb = self.model.encode(gt_texts)

        # Compute similarity
        similarity = cosine_similarity(pred_emb, gt_emb)

        matches = []
        used_gt = set()

        # Greedy matching
        for pred_idx in range(len(prediction)):
            best_gt_idx = None
            best_sim = 0.0

            for gt_idx in range(len(ground_truth)):
                if gt_idx in used_gt:
                    continue

                if similarity[pred_idx][gt_idx] > best_sim:
                    best_sim = similarity[pred_idx][gt_idx]
                    best_gt_idx = gt_idx

            # Check if match is good enough
            if best_sim > 0.85:
                pred_req = prediction[pred_idx]
                gt_req = ground_truth[best_gt_idx]

                # Also check type and actor
                if (pred_req['type'] == gt_req['type'] and
                    self.actors_match(pred_req.get('actor'), gt_req.get('actor'))):
                    matches.append((pred_idx, best_gt_idx))
                    used_gt.add(best_gt_idx)

        return matches

    def actors_match(self, actor1: str, actor2: str) -> bool:
        \"\"\"Check if actors match (including synonyms)\"\"\"

        if not actor1 or not actor2:
            return False

        actor1 = actor1.lower()
        actor2 = actor2.lower()

        if actor1 == actor2:
            return True

        # Synonym groups
        synonyms = [
            {'user', 'customer', 'end-user', 'client'},
            {'system', 'application', 'platform', 'service'},
            {'admin', 'administrator', 'superuser'}
        ]

        for group in synonyms:
            if actor1 in group and actor2 in group:
                return True

        return False

    def compute_type_accuracy(self, prediction: List[Dict], ground_truth: List[Dict]) -> float:
        \"\"\"Compute type classification accuracy on matched requirements\"\"\"

        matches = self.match_requirements(prediction, ground_truth)

        if not matches:
            return 0.0

        correct_types = sum(
            1 for pred_idx, gt_idx in matches
            if prediction[pred_idx]['type'] == ground_truth[gt_idx]['type']
        )

        return correct_types / len(matches)

    def compute_ac_completeness(self, prediction: List[Dict], ground_truth: List[Dict]) -> float:
        \"\"\"Compute acceptance criteria completeness\"\"\"

        matches = self.match_requirements(prediction, ground_truth)

        if not matches:
            return 0.0

        completeness_scores = []

        for pred_idx, gt_idx in matches:
            pred_ac = prediction[pred_idx].get('acceptance_criteria', [])
            gt_ac = ground_truth[gt_idx].get('acceptance_criteria', [])

            if not gt_ac:
                completeness_scores.append(1.0)
                continue

            # Match criteria using embeddings
            if not pred_ac:
                completeness_scores.append(0.0)
                continue

            pred_ac_emb = self.model.encode(pred_ac)
            gt_ac_emb = self.model.encode(gt_ac)

            similarity = cosine_similarity(gt_ac_emb, pred_ac_emb)

            matched_gt_ac = sum(1 for i in range(len(gt_ac)) if np.max(similarity[i]) > 0.80)

            completeness = matched_gt_ac / len(gt_ac)
            completeness_scores.append(completeness)

        return np.mean(completeness_scores) if completeness_scores else 0.0

    def aggregate(self, results: List[Dict]) -> Dict:
        \"\"\"Aggregate scores across all samples\"\"\"

        valid_results = [r for r in results if 'scores' in r]

        if not valid_results:
            return {}

        # Average all metrics
        aggregate = {}

        for metric in ['recall', 'precision', 'f1', 'type_accuracy', 'ac_completeness']:
            values = [r['scores'][metric] for r in valid_results if metric in r['scores']]
            aggregate[metric] = np.mean(values) if values else 0.0
            aggregate[f'{metric}_std'] = np.std(values) if values else 0.0

        # Confidence calibration
        predictions_with_confidence = []
        for r in valid_results:
            pred = r['prediction']
            gt = r['ground_truth']

            # Determine correctness for each predicted requirement
            matches = self.match_requirements(pred, gt)
            matched_pred_indices = {pred_idx for pred_idx, _ in matches}

            for idx, req in enumerate(pred):
                if 'confidence' in req:
                    predictions_with_confidence.append({
                        'confidence': req['confidence'],
                        'correct': idx in matched_pred_indices
                    })

        if predictions_with_confidence:
            from eval.metrics.calibration import compute_confidence_calibration
            calibration = compute_confidence_calibration(predictions_with_confidence)
            aggregate['confidence_calibration'] = calibration

        return aggregate
```

### 6.3 Report Generator

```python
# eval/report_generator.py

from jinja2 import Template
from pathlib import Path
from datetime import datetime
import json

class ReportGenerator:
    \"\"\"Generate HTML evaluation reports\"\"\"

    def __init__(self, config: dict):
        self.config = config

    def generate(self, results: Dict) -> str:
        \"\"\"Generate HTML report from results\"\"\"

        output_dir = Path(self.config['reporting']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / f"report_{results['run_id']}.html"

        # Load template
        template_path = Path(__file__).parent / 'templates' / 'report_template.html'
        with open(template_path) as f:
            template = Template(f.read())

        # Render
        html = template.render(
            results=results,
            generated_at=datetime.utcnow().isoformat(),
            config=self.config
        )

        with open(report_file, 'w') as f:
            f.write(html)

        return str(report_file)
```

---

## 7. Continuous Evaluation in CI/CD

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/eval.yml

name: Agent Evaluation

on:
  push:
    paths:
      - 'src/agents/**'
      - 'src/prompts/**'
      - 'src/schemas/**'
      - 'src/orchestrator/**'
  pull_request:
    paths:
      - 'src/agents/**'
      - 'src/prompts/**'
  schedule:
    - cron: '0 2 * * *'  # Nightly at 2 AM UTC
  workflow_dispatch:  # Manual trigger

env:
  PYTHON_VERSION: '3.11'

jobs:
  synthetic-eval:
    name: Synthetic Dataset Evaluation
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for drift detection

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache Poetry dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pypoetry
          key: ${{ runner.os }}-poetry-eval-${{ hashFiles('**/poetry.lock') }}

      - name: Install dependencies
        run: |
          cd backend
          poetry install --with eval

      - name: Download cached LLM responses
        uses: actions/cache@v3
        with:
          path: eval/.cache
          key: llm-cache-${{ hashFiles('eval/datasets/**') }}

      - name: Run synthetic evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          cd backend
          poetry run python -m eval.evaluation_runner run \\
            --config eval/configs/synthetic-eval.yaml

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: synthetic-eval-results
          path: backend/reports/eval_*.json

  gold-eval:
    name: Gold Dataset Evaluation
    runs-on: ubuntu-latest
    timeout-minutes: 20
    needs: synthetic-eval

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          cd backend
          poetry install --with eval

      - name: Run gold evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          cd backend
          poetry run python -m eval.evaluation_runner run \\
            --config eval/configs/gold-eval.yaml

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: gold-eval-results
          path: backend/reports/eval_*.json

  drift-detection:
    name: Detect Performance Drift
    runs-on: ubuntu-latest
    needs: [synthetic-eval, gold-eval]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          cd backend
          poetry install --with eval

      - name: Download current results
        uses: actions/download-artifact@v3
        with:
          name: synthetic-eval-results
          path: backend/reports/current/

      - name: Download baseline results
        run: |
          # Get baseline from last successful run on main branch
          cd backend
          mkdir -p reports/baseline
          gh run download --name synthetic-eval-results \\
            --repo ${{ github.repository }} \\
            --branch main \\
            --dir reports/baseline/ || true
        env:
          GH_TOKEN: ${{ github.token }}

      - name: Detect drift
        id: drift
        run: |
          cd backend
          CURRENT=$(ls reports/current/eval_*.json | head -1)
          BASELINE=$(ls reports/baseline/eval_*.json | head -1)

          if [ -f \"$BASELINE\" ]; then
            poetry run python -m eval.evaluation_runner detect-drift \\
              \"$BASELINE\" \"$CURRENT\" --threshold 0.03
          else
            echo \"No baseline found, skipping drift detection\"
            exit 0
          fi
        continue-on-error: true

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const resultsPath = 'backend/reports/current/eval_*.json';
            const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));

            let comment = '## 📊 Evaluation Results\\n\\n';

            for (const [agent, data] of Object.entries(results.agents)) {
              comment += `### ${agent.charAt(0).toUpperCase() + agent.slice(1)} Agent\\n`;
              comment += '| Metric | Value |\\n';
              comment += '|--------|-------|\\n';

              for (const [metric, value] of Object.entries(data.aggregate_scores)) {
                if (typeof value === 'number') {
                  comment += `| ${metric} | ${value.toFixed(3)} |\\n`;
                }
              }
              comment += '\\n';
            }

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });

      - name: Fail if thresholds not met
        run: |
          cd backend
          CURRENT=$(ls reports/current/eval_*.json | head -1)

          # Check if any evaluations failed
          FAILED=$(cat \"$CURRENT\" | jq '.agents | to_entries[] | .value.results[] | select(.passed == false) | 1' | wc -l)

          if [ $FAILED -gt 0 ]; then
            echo \"❌ $FAILED evaluations failed to meet thresholds\"
            exit 1
          fi

          echo \"✅ All evaluations passed\"

  prompt-regression:
    name: Prompt Regression Tests
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          cd backend
          poetry install --with eval

      - name: Run prompt regression tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          cd backend
          poetry run pytest tests/golden -v --golden-regen=false

      - name: Check for prompt drift
        run: |
          cd backend
          poetry run python scripts/check_prompt_drift.py
```

---

**Continuing with local tools, dataset versioning, and final artifacts in next file...**
