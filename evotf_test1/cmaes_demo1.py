import jax

from evosax import CMA_ES
from evosax.problems import BBOBFitness

# Instantiate the problem evaluator
rosenbrock = BBOBFitness("RosenbrockOriginal", num_dims=2, seed_id=2)

# Instantiate the search strategy
rng = jax.random.PRNGKey(0)
strategy = CMA_ES(popsize=20, num_dims=2, elite_ratio=0.5)
es_params = strategy.default_params
state = strategy.initialize(rng, es_params)

# Run ask-eval-tell loop - NOTE: By default minimization!
for t in range(1000):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state, es_params)
    fitness = rosenbrock.rollout(rng_eval, x) # Your population evaluation fct 
    state = strategy.tell(x, fitness, state, es_params)

# Get best overall population member & its fitness
#state.best_member, state.best_fitness
print("Best fitness:",state.best_fitness)
print("Best member:", state.best_member)