from semantickv.model.loader import load_model
from semantickv.ablation.counterfactual import compute_importance_scores

model, tok = load_model('./models/llama3-8b-instruct')
r = compute_importance_scores(
    model, tok,
    'France capital is Paris. Germany capital?',
    max_new_tokens=5,
    sample_fraction=0.3,
)
print('PASS', r['importance_scores'])
