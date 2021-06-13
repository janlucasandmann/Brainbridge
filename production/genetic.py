""" Genetic algorithm for feature selection """

import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

def generateDnaString(numberOfGenes, X_input):
    # X_input must be transposed!

    possibleGenes = len(X_input)
    geneNumbers = []
    dnaString = []
    

    i = 0
    while i < numberOfGenes:
        rnd = -1
        
        while rnd in geneNumbers or rnd == -1:
            rnd = random.randint(0, (possibleGenes - 1))
        """
            
        c = 0
        while c < 1:
            rnd = random.randint(0, (possibleGenes - 1))
            print(rnd)
            if not rnd in geneNumbers:
                print("dumdidadeldei")
                c += 1
        """

        geneNumbers.append(rnd)
        dnaString.append(X_input[rnd])

        i += 1

    return [geneNumbers, np.transpose(dnaString)]

def evaluateDnaString(dnaString, events, trainingNumber, estimators):
    # Create and train random forest model

    clf=RandomForestClassifier(n_estimators=estimators)
    clf.fit(dnaString[:trainingNumber - 1], events[:trainingNumber - 1])

    prediction=clf.predict(dnaString[trainingNumber:])

    return metrics.accuracy_score(events[trainingNumber:], prediction)

def killWeakSubjects(dnaStrings, killRate, trainingNumber, estimators, events):

    kill = len(dnaStrings) * killRate
    res = [] 

    for i in dnaStrings:
        res.append(evaluateDnaString(i[1], events, trainingNumber, estimators))

    sortedDnaStrings = [x for _,x in sorted(zip(res,dnaStrings))]

    return sortedDnaStrings[len(sortedDnaStrings) - int(kill) + 1:]

def sex(subjectOne, subjectTwo, numberOfKids, X_input, numberOfMutations):
    # X_input must be transposed!

    possibleGenes = len(X_input)
    kids = []
    i = 0

    while i < numberOfKids:
        kidDnaString = []
        kidGeneNumbers = []

        c = 0
        while c < len(subjectOne[0]):
            rnd = random.randint(0,1)

            if rnd == 0:
                kidDnaString.append(np.transpose(subjectOne[1])[c])
                kidGeneNumbers.append(subjectOne[0][c])
            else:
                kidDnaString.append(np.transpose(subjectTwo[1])[c])
                kidGeneNumbers.append(subjectTwo[0][c])

            c += 1

        kids.append(mutateGenes(kidGeneNumbers, kidDnaString, numberOfMutations, X_input))

        i += 1

    return kids

def mutateGenes(geneNumbers, dnaString, numberOfMutations, X_input):
    # X_input must be transposed!


    possibleGenes = len(X_input)

    i = 0
    while i < numberOfMutations:
        geneNumber = random.randint(0, len(geneNumbers) - 1)
        rnd = random.randint(0, possibleGenes - 1)

        geneNumbers[geneNumber] = rnd
        dnaString[geneNumber] = X_input[rnd]

        i += 1

    return [geneNumbers, np.transpose(dnaString)]

def simulateEvolution(X_input, numberOfGenes, numberOfSubjects, numberOfGenerations, numberOfMutations, killRate, trainingNumber, estimators, events):
    # X_input must be transposed!

    subjects = []

    i = 0
    while i < numberOfSubjects:
        subjects.append(generateDnaString(numberOfGenes, X_input))
        i += 1

    k = 0
    while k < numberOfGenerations:
        print("Starting generation ", (k+1))
        decimatedSubjects = killWeakSubjects(subjects, killRate, trainingNumber, estimators, events)
        kids = []
        subjects = decimatedSubjects

        c = 0
        if k < numberOfGenerations - 1:
            while c < len(decimatedSubjects[0]):
                new_kids = sex(decimatedSubjects[c], decimatedSubjects[len(decimatedSubjects) - c - 1], 2, X_input, numberOfMutations)

                for i in new_kids:
                    subjects.append(i)
                c += 1
        k += 1

    return subjects[len(subjects) - 1][0]