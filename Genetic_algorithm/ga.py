
import deap
from deap import tools
from deap import algorithms

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)
        
        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook





import random
from pandas import read_csv
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
class Zoo:
    
    NUM_FOLDS = 5
    
    def __init__(self, randomSeed):
        
        self.randomSeed = randomSeed
        
        self.data = read_csv('C:\\Users\\Дмитрий\\Desktop\\VS_Code\\4kurs\\TIABD\\Genetic_algorithm\\breast-cancer-wisconsin1.data', header = None, usecols=range(1, 11))
        
        self.X = self.data.iloc[:, 0:9]
        self.y = self.data.iloc[:, 9]
        
        self.kfold = model_selection.KFold(n_splits = self.NUM_FOLDS, random_state = self.randomSeed, shuffle=True)
        
        self.classifier = DecisionTreeClassifier(random_state = self.randomSeed)
        
    def __len__(self):
        return self.X.shape[1]
    
    def getMeanAccuracy(self, zeroOneList):
        
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = self.X.drop(self.X.columns[zeroIndices], axis = 1)
        
        cv_results = model_selection.cross_val_score(self.classifier, currentX, self.y, cv = self.kfold, scoring = 'accuracy')
        
        return cv_results.mean()
    
def main():
        zoo = Zoo(randomSeed=42)
        
        allOnes = [1] * len(zoo)
        print("-- Выделены все признаки: ", allOnes, ", верность = ", zoo.getMeanAccuracy(allOnes))
    
main()
     



from deap import base
from deap import creator
from deap import tools
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

POPULATION_SIZE = 50
P_CROSSOVER = 0.9 # provability for crossover
P_MUTATION = 0.3 # provability for mutating an individual
MAX_GENERATIONS = 20
HALL_OF_FAME_SIZE = 5
FEATURE_PENALTY_FACTOR = 0.001
# set the random seed: счетчик псевдослучайных чисел
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the Zoo test class:
zoo = Zoo(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(-1.0,)) # максимизиуерт значения но сама функция будет стримится к минимуму потому что сама функция без -1 будет всегда максимизировать значение 
 # FintessMax восприменяется как изменяемый объект 
creator.create("Individual", list, fitness = creator.FitnessMax) # fintess экземпляр ранее созданного класса Individual (еще оно лакальное свойство экземпляра класса ) 

toolbox.register("zeroOrOne", random.randint, 0, 1) # класс для суммы onemaxFitness

toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(zoo)) # создаем случайный список из 0 и 1 
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator) # генерация всей популяции 
# создает списки tools.initRepeat (хранение генов ,     генерация значения гена , число генов в хромосоме) 

# fitness calculation:
def zooClassificationAccuracy(individual): # расчитывает пирспасобленность каждого отдельного индивидуума 
    numFeaturesUsed = sum(individual)
    if numFeaturesUsed == 0:
        return 0.0,
    else:
        accuracy = zoo.getMeanAccuracy(individual)
        return accuracy - FEATURE_PENALTY_FACTOR * numFeaturesUsed, # return a tuple
    
toolbox.register("evaluate", zooClassificationAccuracy) # возвращает картеж значений приспасобленности каждого отдельного индивидуума 
# genetic operators:mutFlipBit
toolbox.register("select", tools.selTournament, tournsize = 2) # турнирный отбор , три особи как пример 

toolbox.register("mate", tools.cxTwoPoint)# выполняет скрещивания для родителей 

toolbox.register("mutate", tools.mutFlipBit, indpb = 1.0 / len(zoo)) # мутация  инвертирует обычные биты генов , вероятность мутации в хромосоме особи 

def main1():
    
    # create initial population (generation 0):
    population = toolbox.populationCreator(n = POPULATION_SIZE)
    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max) # максимальная приспасобленность 
    stats.register("avg", numpy.mean) # среднее приспасобленность
    stats.register('values', numpy.array)
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    # perform the Genetic Algorithm flow with hof feature added: вызов 
    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb = P_CROSSOVER, mutpb = P_MUTATION, ngen = MAX_GENERATIONS, stats = stats, halloffame = hof, verbose = True)
    
    # print best solutin found: хуита не работает 
    print("- Лучшие решения:")
    # for i in range(HALL_OF_FAME_SIZE):
    #     print(i, ": ", hof.items[i], ", приспособленность = ", hof.items[i].fitness.values[0], ", верность = ", zoo.getMeanAccuracy(hof.items[i]), ", признаков = ", sum(hof.items[i]))
    
    print(hof)
    # extract statistics:
    maxFitnessValues, meanFitnessValues  = logbook.select("max", "avg")
    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color = "red")
    plt.plot(meanFitnessValues, color = "green")
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()
    
main1()
     