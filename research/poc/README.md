# The SIT320 toy model

Budget Guard grew out of a small simulation I wrote for my SIT320 research proposal. The proposal, a comparison note and the notebook aren't in this repo. This page is the short version of what the toy showed, so the main README has something to point at.

## The setup

A task is five steps in a row. At each step an agent can do one of three things:

- act with a cheap model, which costs 1
- hand the step to an expensive model, which costs 5
- ask the user a clarifying question, which costs 3 and can be done once per task

Every task has a hard budget, from 5 up to 30. A task only succeeds if all five steps succeed. If the agent can't afford any action at some step, the task fails and counts as a budget violation. The agent never sees how hard a step really is. It sees a noisy uncertainty score, and a noisy flag for whether the task is underspecified. Skipping a needed clarification makes every later step less likely to work.

I compared a few policies on the same 3,000 tasks:

- a fixed threshold in the style of ReDAct, calibrated once to defer about twice per task and never looking at the budget
- random deferral at the same rate, as a control
- a dynamic program over (step, budget left, belief about ambiguity), which looks at the budget and is exact for the toy's own model

## What came out

- The dynamic program never broke the budget and beat the fixed threshold at every budget I tried. The fixed threshold broke the cap on 47 percent of tasks at a budget of 5. At large budgets it underspent and left success on the table.
- Tuning the threshold for each budget closed about 61 percent of the gap to the dynamic program, but it still broke the cap on some tasks.
- Adding a simple guard that drops any action that would leave too little for the remaining steps removed every violation and closed about 84 percent of the gap.
- A threshold that moves with the budget left per remaining step, fitted once to the dynamic program's choices, closed about 93 percent of the gap with no violations, including on budgets it wasn't fitted on.
- The dynamic program's value never goes down when the budget goes up. I proved this by induction in the proposal and checked it against the actual value table the code produces. The main repo has the same check as a test.
- A threshold with no cap at all sat one to six points of success above the capped dynamic program. That gap is the price of making the cap hard.

## What it doesn't show

It's a toy. The uncertainty signal is made up, the state space is tiny, and nothing here involves a real model. It says that the budget belongs in the decision rule, not that this works on real questions.

## How Budget Guard differs

The real project changes the problem. Each question is a single turn, so a failure no longer ends a task. The reward is the number of correct answers over a stream of questions, which adds up, where the toy multiplied success over steps. The remaining budget still sits in the state, and the dynamic program, the monotonicity lemma and the hard cap carry over. The limitations section of the main README says more.

The proposal is archived on Zenodo, DOI 10.5281/zenodo.23101669.
