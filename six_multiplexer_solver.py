import numpy.random as npr
import math
import copy
import random
import matplotlib.pyplot as plt

# Create tree node class for generated program
class Node:
    def __init__(self, data):
        self.children = []
        self.fitness = 0
        self.data = data

        self.depth = 1
        self.size = 1

    # Method of tree print sourced from Martijn Pieters on Stack Overflow
    def __str__(self, level=0):
        ret = "\t"*level+repr(self.data)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'

# Set Parameters for the Genetic Program
N_GENERATIONS = 100
N_POPULATION = 75
P_M = 0.03
P_C = 0.6

# Create the set of all combinations for register values
case_set = []

# Fill case_set in binary(0 to 63)
for i in range(64):
    case_set.append(i)
    case_set[i] = [int(j) for j in bin(case_set[i])[2:].zfill(6)]

# Define operators and terminals
operators = ['AND', 'OR', 'NOT', 'IF']
terminals = ['a0', 'a1', 'd0', 'd1', 'd2', 'd3']

# Simple function to evaluate how many children to generate
def num_child(data):
    if data == 'AND' or data == 'OR':
        num_child = 2
    elif data == 'IF':
        num_child = 3
    elif data == 'NOT':
        num_child = 1
    else:
        num_child = 0
    
    return num_child

# Recursive method to fill program tree
def fill_prog(prog):

    # Figure out how many children need to be generated
    num_generated_children = num_child(prog.data)

    # Generate children
    for i in range(num_generated_children):
        # Generate random number to select terminal or leaf to be generated
        select = random.random()

        if select < 0.4:
            select_op = random.randint(0,3)
            child_node = Node(operators[select_op])
            
            # Add child
            prog.children.append(child_node)
            
            # Recurse on child
            fill_prog(child_node)

        # If it is a leaf node that is generated, terminate that instance after adding it to the tree
        else:
            select_term = random.randint(0,5)
            child_node = Node(terminals[select_term])

            # Add child
            prog.children.append(child_node)
        
    # Add amount of children to parent's size
    for i in range(num_generated_children):
        prog.size += prog.children[i].size

# Function to generate program
def gen_prog():

    # Create root node
    select_root = random.randint(0,3)
    prog = Node(operators[select_root])

    # Fill the rest of the program tree
    fill_prog(prog)

    return prog

# Turn program tree into a single line string
def traverse(prog):
    stack = []
    program_string = []

    stack.append(prog)

    # Perform in-order traversal to visit all nodes
    while len(stack) != 0:
        visit_node = stack.pop()

        program_string.append(visit_node.data)

        reverse = visit_node.children
        reverse.reverse()

        # Add children from right to left
        for i in range(len(reverse)):
            stack.append(reverse[i])
    
    return program_string

# Function to recursively evaluate a program tree
def recursive_eval(prog, reg):
    # Check if the evaluated node is a terminal node
    if terminals.count(prog.data) == 1:
        if prog.data == 'a0':
            return reg[0]
        elif prog.data == 'a1':
            return reg[1]
        elif prog.data == 'd0':
            return reg[2]
        elif prog.data == 'd1':
            return reg[3]
        elif prog.data == 'd2':
            return reg[4]
        elif prog.data == 'd3':
            return reg[5]
    
    # List to contain calculated children
    calc = []

    # Evaluate children if it is an operator
    for i in range(len(prog.children)):
        calc.append(recursive_eval(prog.children[i], reg))
    
    # Apply operations

    if prog.data == 'AND':
        return calc[0] and calc[1]
    elif prog.data == 'OR':
        return calc[0] or calc[1]
    elif prog.data == 'NOT':
        return int(not calc[0])
    elif prog.data == 'IF':
        if calc[0] == 1:
            return calc[1]
        else:
            return calc[2]

# Function to evaluate generated program
def evaluate(prog, reg):

    pass_test = 0

    a0 = str(reg[0])
    a1 = str(reg[1])
    address = int(a0 + a1, 2)

    eval = recursive_eval(prog, reg)

    # Check if the corresponding register has the correct value
    if eval == reg[address + 2]:
        pass_test = 1

    return pass_test

# Function to evaluate fitness
def fitness(prog):

    fit = 0

    for i in range(64):
        fit += evaluate(prog, case_set[i])

    fit = fit / 64

    return fit

# Selection function for elitism strategy
def select_one(pop):
    max = sum([c.fitness for c in pop])
    selection_probability = [c.fitness/max for c in pop]
    return pop[npr.choice(len(pop), p=selection_probability)]

# Crossover function
def crossover(parent1, parent2):
    # Initialize stacks for traversal
    stack1 = []
    stack2 = []

    stack1.append(parent1)
    stack2.append(parent2)

    # Randomly pick a node from parents
    index1 = random.randint(0, parent1.size)
    index2 = random.randint(0, parent2.size)

    # Initialize the swap nodes
    visit_node1 = parent1
    visit_node2 = parent2

    if len(parent1.children) == 0:
        visit_node1 = parent1
    else:
        # Perform in-order traversal to locate nodes
        for i in range(index1):
            if len(stack1) == 0:
                break

            visit_node1 = stack1.pop()

            # Add children from right to left
            for child in reversed(visit_node1.children):
                stack1.append(child)

    if len(parent2.children) == 0:
        visit_node2 = parent2
    else:
        for i in range(index2):
            if len(stack2) == 0:
                break

            visit_node2 = stack2.pop()

            # Add children from right to left
            for child in reversed(visit_node2.children):
                stack2.append(child)
    
    # Perform the swap
    temp_node = copy.deepcopy(visit_node1)
    visit_node1.data = visit_node2.data
    visit_node1.children.clear()
    for child in visit_node2.children:
        visit_node1.children.append(child)
    
    visit_node2.data = temp_node.data
    visit_node2.children.clear()
    for child in temp_node.children:
        visit_node2.children.append(child)
    
    # Update fitness values
    parent1.fitness = fitness(parent1)
    parent2.fitness = fitness(parent2)

# Mutation function
def mutate(prog):
    stack = []

    stack.append(prog)

    index = random.randint(0, prog.size)

    # Initialize the mutated node
    visit_node = prog

    if prog.size == 1:
        visit_node = prog
    else:
        for i in range(index):
            if len(stack) == 0:
                break

            visit_node = stack.pop()

            # Add children from right to left
            for child in reversed(visit_node.children):
                stack.append(child)
    
    added_tree = gen_prog()

    visit_node.data = added_tree.data
    visit_node.children.clear()
    for child in added_tree.children:
        visit_node.children.append(child)
    
    # Update fitness values
    prog.fitness = fitness(prog)

# Function to run the GP
def run_gp():

    # Initialize list for best fitness values from each generation
    generation_best = []

    # Initialize list of generation
    generation = []

    # Generate first generation of programs
    for i in range(N_POPULATION):
        prog = gen_prog()
        prog.fitness = fitness(prog)

        generation.append(prog)

    # Sort the generation based on highest fitness
    generation = sorted(generation, key=lambda prog: prog.fitness)
    generation.reverse()

    # Add best from generated generation
    generation_best.append(generation[0])

    print("Starting generation 1")

    # Enter the evolutionary process
    for i in range(N_GENERATIONS):
        print("Running generation " + str(i+1))

        child_produced = 0
        new_generation = []

        # Add two best from the previous generation to the new generation
        new_generation.append(copy.deepcopy(generation[0]))
        new_generation.append(copy.deepcopy(generation[1]))

        # Start crossover/mutation/carry over process. Guarantee that the 2 previous best survive
        while child_produced != N_POPULATION - 2:

            # Check if there is only 1 spot left. Perform a carry over if true
            if child_produced == N_POPULATION - 3:
                carry = copy.deepcopy(select_one(generation))
                
                child_produced += 1

                new_generation.append(carry)
            else:
                # Get variation selection value
                var_select = random.random()

                # Mutate
                if var_select < P_M:
                    mutated_prog = copy.deepcopy(select_one(generation))
                    mutate(mutated_prog)
                    
                    child_produced += 1

                    new_generation.append(mutated_prog)
                # Crossover
                elif P_M <= var_select <= P_C:
                    parent1 = copy.deepcopy(select_one(generation))
                    parent2 = copy.deepcopy(select_one(generation))

                    crossover(parent1, parent2)

                    child_produced += 2

                    new_generation.append(parent1)
                    new_generation.append(parent2)
                # Carryover
                else:
                    carry = copy.deepcopy(select_one(generation))
                    
                    child_produced += 1

                    new_generation.append(carry)

            
        # Sort the new generation based on highest fitness
        new_generation = sorted(new_generation, key=lambda prog: prog.fitness)
        new_generation.reverse()

        # Record the best value
        generation_best.append(new_generation[0])

        # Replace old generation
        generation = new_generation
    
    best_solution = generation_best.pop()

    generation_list = []

    for i in range(N_GENERATIONS):
        generation_list.append(i)

    generation_fitness = []
    for i in range(N_GENERATIONS):
        generation_fitness.append(generation_best[i].fitness)

    print(best_solution)
    print(best_solution.fitness)
    print(traverse(best_solution))
    
    plt.plot(generation_list, generation_fitness)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title('Best Fitness at Each Generation')
    plt.grid()
    plt.show()
    return

# Main function
def main():
    run_gp()
    print("Done!")

if __name__ == "__main__": main()