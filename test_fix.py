from semantickv.model.loader import load_model
from semantickv.ablation.counterfactual import compute_importance_scores

model, tok = load_model('./models/llama3-8b-instruct')
prompt = (
    'Paris is the capital of France and is known for the Eiffel Tower. '
    'Berlin is the capital of Germany. London is the capital of England. '
    'What is the capital of Italy?'
)
r = compute_importance_scores(
    model, tok,
    prompt,
    max_new_tokens=5,
    sample_fraction=0.3,
)
print('PASS', r['importance_scores'])
