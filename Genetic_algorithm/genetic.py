


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
    
if __name__ == "__main__":
    main()



from deap import base , algorithms
from deap import creator
from deap import tools
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

POPULATION_SIZE = 50
P_CROSSOVER = 0.9 # provability for crossover
P_MUTATION = 0.3 # provability for mutating an individual
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 5
FEATURE_PENALTY_FACTOR = 0.001
# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the Zoo test class:
zoo = Zoo(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness = creator.FitnessMax)

toolbox.register("zeroOrOne", random.randint, 0, 1)

toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(zoo))

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation:
def zooClassificationAccuracy(individual):
    numFeaturesUsed = sum(individual)
    if numFeaturesUsed == 0:
        return 0.0,
    else:
        accuracy = zoo.getMeanAccuracy(individual)
        return accuracy - FEATURE_PENALTY_FACTOR * numFeaturesUsed, # return a tuple
    
toolbox.register("evaluate", zooClassificationAccuracy)
# genetic operators:mutFlipBit
toolbox.register("select", tools.selTournament, tournsize = 2)

toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutFlipBit, indpb = 1.0 / len(zoo))

def main1():
    
    # create initial population (generation 0):
    population = toolbox.populationCreator(n = POPULATION_SIZE)
    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb = P_CROSSOVER, mutpb = P_MUTATION, ngen = MAX_GENERATIONS, stats = stats, halloffame = hof, verbose = True)
    
    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # print best solutin found:
    print("- Лучшие решения:")
    for i in range(HALL_OF_FAME_SIZE):
        print(hof.items[i],'hofitems',type(hof.items[i]))
        asd=hof.items[i]
        print(type(numpy.array(hof.items[i],dtype=object)),'asd=',type(asd))
        print(i, ": ", hof.items[i], ", приспособленность = ", hof.items[i].fitness.values[0], ", верность = ", ", признаков = ", sum(hof.items[i]))
        
    print(hof.items)
    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color = "red")
    plt.plot(meanFitnessValues, color = "green")
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()
    
if __name__ == "__main__":
    main1()