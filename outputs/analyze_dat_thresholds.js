const fs = require("fs");

const rows = fs
  .readFileSync("C:/Users/ASUS/Downloads/full_dat_scored.jsonl", "utf8")
  .trim()
  .split(/\r?\n/)
  .map(JSON.parse);
const manifest = JSON.parse(
  fs.readFileSync(
    "D:/trustmargin-experiments/outputs/split_manifest.json",
    "utf8",
  ),
);
const validationIndexes = new Set(
  manifest.validation_ids.map((id) => Number(id.slice(6)) - 1),
);
const validation = rows.filter((_, index) => validationIndexes.has(index));
const heldout = rows.filter((_, index) => !validationIndexes.has(index));

function normalize(text) {
  return text
    .normalize("NFKC")
    .toLocaleLowerCase("und")
    .replace(/[^\p{L}\p{N}_\s]/gu, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function tokenF1(prediction, golds) {
  const predicted = normalize(prediction);
  let best = 0;
  for (const gold of golds) {
    const truth = normalize(gold);
    if (!predicted.length || !truth.length) {
      best = Math.max(best, Number(predicted.length === truth.length));
      continue;
    }
    const predictedCounts = new Map();
    const truthCounts = new Map();
    for (const token of predicted) {
      predictedCounts.set(token, (predictedCounts.get(token) || 0) + 1);
    }
    for (const token of truth) {
      truthCounts.set(token, (truthCounts.get(token) || 0) + 1);
    }
    let overlap = 0;
    for (const [token, count] of predictedCounts) {
      overlap += Math.min(count, truthCounts.get(token) || 0);
    }
    best = Math.max(best, (2 * overlap) / (predicted.length + truth.length));
  }
  return best;
}

function exactMatch(prediction, golds) {
  const predicted = normalize(prediction).join(" ");
  return Number(golds.some((gold) => normalize(gold).join(" ") === predicted));
}

for (const row of rows) {
  row.m_prior = row.scores.q_only.rag - row.scores.q_only.d2l;
  row.m_bind =
    row.scores.q_context.rag -
    row.scores.context_only.rag -
    (row.scores.q_context.d2l - row.scores.context_only.d2l);
  row._d2lF1 = tokenF1(row.d2l_answer, row.gold_answers);
  row._ragF1 = tokenF1(row.rag_answer, row.gold_answers);
  row._d2lExact = exactMatch(row.d2l_answer, row.gold_answers);
  row._ragExact = exactMatch(row.rag_answer, row.gold_answers);
}

function evaluate(data, lambdaBind, tau) {
  let f1 = 0;
  let exact = 0;
  let d2l = 0;
  let rag = 0;
  for (const row of data) {
    const trustScore = row.m_prior + lambdaBind * row.m_bind;
    const source = trustScore > tau ? "rag" : "d2l";
    source === "rag" ? rag++ : d2l++;
    f1 += source === "rag" ? row._ragF1 : row._d2lF1;
    exact += source === "rag" ? row._ragExact : row._d2lExact;
  }
  return { f1: f1 / data.length, exact: exact / data.length, d2l, rag };
}

function tune(data) {
  let best = null;
  for (let lambdaStep = 0; lambdaStep <= 20; lambdaStep++) {
    const lambdaBind = lambdaStep / 20;
    const scores = data
      .map((row) => row.m_prior + lambdaBind * row.m_bind)
      .sort((a, b) => a - b);
    const taus = [scores[0] - 1];
    for (let index = 0; index < scores.length; index++) {
      taus.push(
        index + 1 < scores.length
          ? (scores[index] + scores[index + 1]) / 2
          : scores[index] + 1,
      );
    }
    for (const tau of taus) {
      const metrics = evaluate(data, lambdaBind, tau);
      const candidate = { lambdaBind, tau, ...metrics };
      if (
        !best ||
        candidate.f1 > best.f1 + 1e-12 ||
        (Math.abs(candidate.f1 - best.f1) < 1e-12 &&
          candidate.exact > best.exact + 1e-12) ||
        (Math.abs(candidate.f1 - best.f1) < 1e-12 &&
          Math.abs(candidate.exact - best.exact) < 1e-12 &&
          candidate.d2l < best.d2l)
      ) {
        best = candidate;
      }
    }
  }
  return best;
}

const oldConfig = { lambdaBind: 0.5, tau: -1.5 };
const currentConfig = { lambdaBind: 0, tau: -1.27 };
const validationBest = tune(validation);
const allBest = tune(rows);
const questionTypes = [...new Set(rows.map((row) => row.question_type))];
const perTypeValidationBest = Object.fromEntries(
  questionTypes.map((type) => [
    type,
    tune(validation.filter((row) => row.question_type === type)),
  ]),
);

function evaluateTypePolicy(data) {
  let f1 = 0;
  let exact = 0;
  let d2l = 0;
  let rag = 0;
  for (const row of data) {
    const config = perTypeValidationBest[row.question_type];
    const selected =
      row.m_prior + config.lambdaBind * row.m_bind > config.tau ? "rag" : "d2l";
    selected === "rag" ? rag++ : d2l++;
    f1 += selected === "rag" ? row._ragF1 : row._d2lF1;
    exact += selected === "rag" ? row._ragExact : row._d2lExact;
  }
  return { f1: f1 / data.length, exact: exact / data.length, d2l, rag };
}

function selectedF1(row, config) {
  const score = row.m_prior + config.lambdaBind * row.m_bind;
  return score > config.tau ? row._ragF1 : row._d2lF1;
}

const heldoutDeltas = heldout.map(
  (row) => selectedF1(row, validationBest) - selectedF1(row, currentConfig),
);
let rngState = 42;
function random() {
  rngState ^= rngState << 13;
  rngState ^= rngState >>> 17;
  rngState ^= rngState << 5;
  return (rngState >>> 0) / 4294967296;
}
const bootstrapMeans = [];
for (let draw = 0; draw < 5000; draw++) {
  let total = 0;
  for (let index = 0; index < heldoutDeltas.length; index++) {
    total += heldoutDeltas[Math.floor(random() * heldoutDeltas.length)];
  }
  bootstrapMeans.push(total / heldoutDeltas.length);
}
bootstrapMeans.sort((a, b) => a - b);
const changedSelections = heldout.filter((row) => {
  const currentSource =
    row.m_prior + currentConfig.lambdaBind * row.m_bind > currentConfig.tau
      ? "rag"
      : "d2l";
  const tunedSource =
    row.m_prior + validationBest.lambdaBind * row.m_bind > validationBest.tau
      ? "rag"
      : "d2l";
  return currentSource !== tunedSource;
}).length;

const result = {
  counts: {
    all: rows.length,
    validation: validation.length,
    heldout: heldout.length,
    validationTypes: Object.fromEntries(
      questionTypes.map((type) => [
        type,
        validation.filter((row) => row.question_type === type).length,
      ]),
    ),
  },
  old: {
    validation: evaluate(validation, oldConfig.lambdaBind, oldConfig.tau),
    heldout: evaluate(heldout, oldConfig.lambdaBind, oldConfig.tau),
    all: evaluate(rows, oldConfig.lambdaBind, oldConfig.tau),
  },
  current: {
    validation: evaluate(validation, currentConfig.lambdaBind, currentConfig.tau),
    heldout: evaluate(heldout, currentConfig.lambdaBind, currentConfig.tau),
    all: evaluate(rows, currentConfig.lambdaBind, currentConfig.tau),
  },
  validationBest,
  validationBestHeldout: evaluate(
    heldout,
    validationBest.lambdaBind,
    validationBest.tau,
  ),
  validationBestAll: evaluate(
    rows,
    validationBest.lambdaBind,
    validationBest.tau,
  ),
  hindsightBestAll: allBest,
  perTypeValidationBest,
  perTypePolicyHeldout: evaluateTypePolicy(heldout),
  perTypePolicyAll: evaluateTypePolicy(rows),
  heldoutBaselines: {
    ragOnly: evaluate(heldout, 0, Number.NEGATIVE_INFINITY),
    d2lOnly: evaluate(heldout, 0, Number.POSITIVE_INFINITY),
  },
  heldoutChangeCurrentToValidationBest: {
    meanF1: heldoutDeltas.reduce((a, b) => a + b, 0) / heldoutDeltas.length,
    bootstrap95: [bootstrapMeans[124], bootstrapMeans[4874]],
    positiveRows: heldoutDeltas.filter((value) => value > 0).length,
    negativeRows: heldoutDeltas.filter((value) => value < 0).length,
    unchangedRows: heldoutDeltas.filter((value) => value === 0).length,
    changedSelections,
  },
  heldoutByType: Object.fromEntries(
    questionTypes.map((type) => {
      const subset = heldout.filter((row) => row.question_type === type);
      return [
        type,
        {
          n: subset.length,
          current: evaluate(subset, currentConfig.lambdaBind, currentConfig.tau),
          validationTuned: evaluate(
            subset,
            validationBest.lambdaBind,
            validationBest.tau,
          ),
        },
      ];
    }),
  ),
};

console.log(JSON.stringify(result, null, 2));
