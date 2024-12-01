import random
import numpy as np
import gym
import compiler_gym
from typing import List, Tuple
from absl import app, flags

flags.DEFINE_list(
    "flags",
    [
        "-add-discriminators",
        "-adce",
        "-loop-unroll",
        "-loop-unswitch",
        "-loop-vectorize",
        "-aggressive-instcombine",
        "-alignment-from-assumptions",
        "-always-inline",
        "-argpromotion",
        "-attributor",
        "-barrier",
        "-bdce",
        "-loop-instsimplify",
        "-break-crit-edges",
        "-simplifycfg",
        "-dce",
        "-called-value-propagation",
        "-die",
        "-canonicalize-aliases",
        "-consthoist",
        "-constmerge",
        "-constprop",
        "-coro-cleanup",
        "-coro-early",
        "-coro-elide",
        "-coro-split",
        "-correlated-propagation",
        "-cross-dso-cfi",
    ],
    "List of optimizations to explore.",
)
flags.DEFINE_integer("population_size", 10, "Number of individuals in the population.")
flags.DEFINE_integer("generation_count", 50, "Number of generations to evolve.")
flags.DEFINE_integer("episode_len", 5, "Length of each sequence of optimizations.")
flags.DEFINE_float("mutation_rate", 0.1, "Probability of mutation.")
flags.DEFINE_float("crossover_rate", 0.8, "Probability of crossover.")

FLAGS = flags.FLAGS


def generate_individual() -> List[str]:
    """Generate a random individual (sequence of passes)."""
    return random.sample(FLAGS.flags, FLAGS.episode_len)


def evaluate_fitness(env, individual: List[str]) -> float:
    """Evaluate the fitness of an individual using a combination of metrics."""
    env.reset()
    total_reward = 0
    initial_ic = env.observation["IrInstructionCount"]
    initial_rt = env.observation["Runtime"][0]
    initial_auto = env.observation["Autophase"][51]

    for action in individual:
        action_index = env.action_space.flags.index(action)
        observation, reward, done, info = env.step(action_index)
        combined = rewards(env, initial_rt, initial_ic, initial_auto)
        total_reward += combined if combined is not None else 0
        if done:
            break

    return total_reward


def rewards(env, initial_rt, initial_ic, initial_auto):
    """Calculate a combined reward from runtime, instruction cost, and autophase."""
    after_ic = env.observation["IrInstructionCount"]
    after_rt = env.observation["Runtime"][-1]

    # Runtime improvement
    runtime = max(0, initial_rt - after_rt) * 0.5

    # Instruction cost reduction
    ic = max(0, initial_ic - after_ic) * 0.003

    # Autophase improvement
    after_auto = env.observation["Autophase"][51]
    auto = max(0, initial_auto - after_auto) * 0.002

    combined = runtime + ic + auto

    return combined


def crossover(parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
    """Perform crossover between two parents."""
    # Random chance to skip the crossover function
    if random.random() > FLAGS.crossover_rate:
        return parent1[:], parent2[:]
    # Crossover Implementation
    point = random.randint(1, FLAGS.episode_len - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    # Return the Childrens
    return child1, child2


def mutate(individual: List[str]):
    """Mutate an individual with a given mutation rate."""
    for i in range(len(individual)):
        if random.random() < FLAGS.mutation_rate:
            individual[i] = random.choice(FLAGS.flags)


def genetic_algorithm(env):
    """Run the genetic algorithm."""
    # Initialize the population
    population = [generate_individual() for _ in range(FLAGS.population_size)]

    for generation in range(FLAGS.generation_count):
        # Evaluate fitness of the population
        fitness_scores = [(individual, evaluate_fitness(env, individual)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Select the top individuals
        top_individuals = [ind for ind, _ in fitness_scores[: FLAGS.population_size // 2]]

        # Generate the next population
        next_population = []
        while len(next_population) < FLAGS.population_size:
            parent1, parent2 = random.sample(top_individuals, 2)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            next_population.extend([child1, child2])

        # Ensure population size remains constant
        population = next_population[: FLAGS.population_size]

        # Log the best individual and its fitness
        best_individual, best_fitness = fitness_scores[0]
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}, Best Individual: {best_individual}")
    
    # Evaluate and log metrics for the best individual
    initial_ic = env.observation["IrInstructionCount"]
    initial_rt = env.observation["Runtime"][0]
    initial_auto = env.observation["Autophase"][51]

    for action in best_individual:
        action_index = env.action_space.flags.index(action)
        env.step(action_index)

    final_ic = env.observation["IrInstructionCount"]
    final_rt = env.observation["Runtime"][-1]
    runtime_reduction = max(0, initial_rt - final_rt)
    ic_reduction = max(0, initial_ic - final_ic)
    overall_reward = 0.5 * runtime_reduction + 0.003 * ic_reduction

    print("\n=== Best Result Metrics ===")
    print(f"Best Individual: {best_individual}")
    print(f"Runtime Reduction: {runtime_reduction:.2f}")
    print(f"Instruction Count Reduction: {ic_reduction:.2f}")
    print(f"Overall Reward: {overall_reward:.2f}")
    # Return the best individual after all generations
    return fitness_scores[0][0], fitness_scores[0][1]


def main(argv):
    del argv  # Unused
    print("=== Configuration ===")
    print(f"Population Size: {FLAGS.population_size}")
    print(f"Generation Count: {FLAGS.generation_count}")
    print(f"Episode Length: {FLAGS.episode_len}")
    print(f"Mutation Rate: {FLAGS.mutation_rate}")
    print(f"Crossover Rate: {FLAGS.crossover_rate}")
    print("=====================")

    env = compiler_gym.make("llvm-v0")
    env.reset()

    best_individual, best_fitness = genetic_algorithm(env)
    print(f"Best Individual: {best_individual}, Fitness: {best_fitness}")

    env.close()


if __name__ == "__main__":
    app.run(main)
