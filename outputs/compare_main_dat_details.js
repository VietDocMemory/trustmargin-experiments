const fs = require("fs");

function readJsonl(path) {
  return fs.readFileSync(path, "utf8").trim().split(/\r?\n/).map(JSON.parse);
}

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
    const left = new Map();
    const right = new Map();
    for (const token of predicted) left.set(token, (left.get(token) || 0) + 1);
    for (const token of truth) right.set(token, (right.get(token) || 0) + 1);
    let overlap = 0;
    for (const [token, count] of left) {
      overlap += Math.min(count, right.get(token) || 0);
    }
    best = Math.max(best, (2 * overlap) / (predicted.length + truth.length));
  }
  return best;
}

function addScores(row) {
  row._d2lF1 = tokenF1(row.d2l_answer, row.gold_answers);
  row._ragF1 = tokenF1(row.rag_answer, row.gold_answers);
  row._finalF1 = tokenF1(row.final_answer, row.gold_answers);
  row._mPrior = row.scores.q_only.rag - row.scores.q_only.d2l;
  row._mBind =
    row.scores.q_context.rag -
    row.scores.context_only.rag -
    (row.scores.q_context.d2l - row.scores.context_only.d2l);
  return row;
}

function source(row, lambdaBind, tau) {
  return row._mPrior + lambdaBind * row._mBind > tau ? "rag" : "d2l";
}

const main = readJsonl(
  "C:/Users/ASUS/Downloads/full_retrieved_results.jsonl",
).map(addScores);
const dat = readJsonl("C:/Users/ASUS/Downloads/full_dat_results.jsonl").map(
  addScores,
);
const mainById = new Map(main.map((row) => [row.id, row]));

const actual = {
  matched: 0,
  selectionChanged: 0,
  toD2l: 0,
  toRag: 0,
  datBetter: 0,
  mainBetter: 0,
  tied: 0,
  changedDatBetter: 0,
  changedMainBetter: 0,
  changedTied: 0,
  sameRagAnswer: 0,
  sameD2lAnswer: 0,
  sameBothCandidates: 0,
  meanDelta: 0,
  byType: {},
};

for (const after of dat) {
  const before = mainById.get(after.id);
  if (!before) continue;
  actual.matched++;
  const delta = after._finalF1 - before._finalF1;
  actual.meanDelta += delta;
  if (delta > 1e-12) actual.datBetter++;
  else if (delta < -1e-12) actual.mainBetter++;
  else actual.tied++;
  const changed = before.selected_source !== after.selected_source;
  if (changed) {
    actual.selectionChanged++;
    after.selected_source === "d2l" ? actual.toD2l++ : actual.toRag++;
    if (delta > 1e-12) actual.changedDatBetter++;
    else if (delta < -1e-12) actual.changedMainBetter++;
    else actual.changedTied++;
  }
  const sameRag = before.rag_answer === after.rag_answer;
  const sameD2l = before.d2l_answer === after.d2l_answer;
  if (sameRag) actual.sameRagAnswer++;
  if (sameD2l) actual.sameD2lAnswer++;
  if (sameRag && sameD2l) actual.sameBothCandidates++;
  const type = after.question_type;
  actual.byType[type] ||= { n: 0, delta: 0, datBetter: 0, mainBetter: 0 };
  actual.byType[type].n++;
  actual.byType[type].delta += delta;
  if (delta > 1e-12) actual.byType[type].datBetter++;
  if (delta < -1e-12) actual.byType[type].mainBetter++;
}
actual.meanDelta /= actual.matched;
for (const value of Object.values(actual.byType)) value.delta /= value.n;

const controlled = {
  oldF1: 0,
  currentF1: 0,
  currentBetter: 0,
  oldBetter: 0,
  tied: 0,
  selectionChanged: 0,
  changedToD2l: 0,
  changedToRag: 0,
};
for (const row of dat) {
  const oldSource = source(row, 0.5, -1.5);
  const currentSource = source(row, 0, -1.27);
  const oldF1 = oldSource === "rag" ? row._ragF1 : row._d2lF1;
  const currentF1 = currentSource === "rag" ? row._ragF1 : row._d2lF1;
  controlled.oldF1 += oldF1;
  controlled.currentF1 += currentF1;
  if (currentF1 > oldF1 + 1e-12) controlled.currentBetter++;
  else if (currentF1 < oldF1 - 1e-12) controlled.oldBetter++;
  else controlled.tied++;
  if (oldSource !== currentSource) {
    controlled.selectionChanged++;
    currentSource === "d2l"
      ? controlled.changedToD2l++
      : controlled.changedToRag++;
  }
}
controlled.oldF1 /= dat.length;
controlled.currentF1 /= dat.length;
controlled.delta = controlled.currentF1 - controlled.oldF1;

console.log(JSON.stringify({ actualRuns: actual, controlledThresholdReplay: controlled }, null, 2));
