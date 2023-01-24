# read in a file 
import os
import time 
import matplotlib.pyplot as plt
import pandas as pd

# read file line for line 
def readFileLineByLine(fileName):
    file = open(fileName, "r")
    lines = file.readlines()
    file.close()
    return lines

def printToFile():
    files = []
    for file in os.listdir("C:/Users/Johan/Desktop/UniWork/Postgrad/Github/Hyper-Heuristics-for-Auto-Design/Storage/ClassificationResults/Results"):
        files.append(file)

    averageTrainingAccuracy = []
    averageTestingAccuracy = []
    averageTrainingTime = []
    averageTestingTime = []
    averageTrainingComplexity = []
    averageTestingComplexity = []
    fileNumber = 17
    for fileIndex in range(fileNumber-1, fileNumber):
        dataFileLocation = "Storage/ClassificationResults/Results/" + files[fileIndex]

        print(dataFileLocation)

        # run through each line 
        testOrTrain = False
        lines = readFileLineByLine(dataFileLocation)
        for lineIndex in range(len(lines)):
            line = lines[lineIndex]
            
            type = line.split(" ")[0]
            print(type, testOrTrain)
            if(type == "Training" or testOrTrain == True):
                
                testOrTrain = True
                if(type == "Accuracy:"):
                    averageTrainingAccuracy.append(float(line.split(" ")[1]))
                elif(type == "Time:"):
                    averageTrainingTime.append(float(line.split(" ")[1]))
                elif(type == "Complexity:"):
                    averageTrainingComplexity.append(float(line.split(" ")[1]))
                    testOrTrain = False
            elif(type == "Testing" or testOrTrain == False):
                testOrTrain = False
                if(type == "Accuracy:"):
                    averageTestingAccuracy.append(float(line.split(" ")[1]))
                elif(type == "Time:"):
                    averageTestingTime.append(float(line.split(" ")[1]))
                elif(type == "Complexity:"):
                    averageTestingComplexity.append(float(line.split(" ")[1]))


    # save the average results of the run
    with open(dataFileLocation, 'a') as f:
        f.write("Final Results: \n")
        f.write('%s: %s\n' % ("Training Accuracies", averageTrainingAccuracy))
        f.write('%s: %s\n' % ("Testing Accuracies", averageTestingAccuracy))
        f.write('%s: %s\n' % ("Average Training Accuracy", sum(averageTrainingAccuracy)/len(averageTrainingAccuracy)))
        f.write('%s: %s\n' % ("Average Testing Accuracy", sum(averageTestingAccuracy)/len(averageTestingAccuracy)))
        f.write("\n")

        f.write('%s: %s\n' % ("Training Times", averageTrainingTime))
        f.write('%s: %s\n' % ("Testing Times", averageTestingTime))
        f.write('%s: %s\n' % ("Average Training Time", sum(averageTrainingTime)/len(averageTrainingTime)))
        f.write('%s: %s\n' % ("Average Testing Time", sum(averageTestingTime)/len(averageTestingTime)))
        f.write("\n")

        f.write('%s: %s\n' % ("Training Complexities", averageTrainingComplexity))
        f.write('%s: %s\n' % ("Testing Complexities", averageTestingComplexity))
        f.write('%s: %s\n' % ("Average Training Complexity", sum(averageTrainingComplexity)/len(averageTrainingComplexity)))
        f.write('%s: %s\n' % ("Average Testing Complexity", sum(averageTestingComplexity)/len(averageTestingComplexity)))
        f.write("\n")

def createeLineGraph(list):
    # create a line graph include legend
    plt.plot(list)
    plt.plot([sum(list)/len(list)]*len(list))
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.show()

def calculateAverageAndStdDev(list):
    average = sum(list)/len(list)
    variance = sum([((x - average) ** 2) for x in list]) / len(list)
    stdDev = variance ** 0.5
    return average, stdDev

def boxplot(list):
    # create a boxplot
    df = pd.DataFrame(list, index=["Decorate", "MultiClassClassifier", "Decision Table", "AutoDes", "ZeroR"])

    df.T.boxplot(vert=False)
    plt.subplots_adjust(left=0.25)
    plt.xlabel("Accuracy")
    plt.show()

# print(calculateAverageAndStdDev([69.71, 80.32 , 928.62, 428.30, 121.43, 19.27 , 137.39, 129.08, 147.95, 508.97, 73.97, 755.18 , 76.35, 52.61, 466.84 , 25.89 , 92.2 , 41.00, 58.23, 399.90, 395.26, 623.95, 22.3, 389.29]))
# print(calculateAverageAndStdDev([2.07, 0.84, 0.82, 1.67, 0.49, 0.83, 0.75, 1.96, 0.47, 0.56, 1.65, 0.32, 3.06, 1.07, 3.65, 0.19, 1.25, 0.82, 4.72, 0.45, 1.46, 0.61, 1.80, 0.29]))
# print(calculateAverageAndStdDev([76.02, 23.33, 45.66, 61.47, 73.11, 91.07, 71.80, 82.25, 70.27, 42.22, 80.97, 76.57, 87.74, 92.28, 78.05, 75.98, 77.91, 66.44, 23.98, 39.31, 58.43, 43.24, 95.65, 33.58]))
# print(calculateAverageAndStdDev([36.57, 10.43, 25.64, 45.36, 68.62, 77.85, 47.83, 64.92, 57.75, 29.30, 57.11, 54.99, 64.19, 71.46, 73.08, 79.48, 97.77, 43.00, 7.5, 22.90, 80.0, 32.57, 66.47, 23.53]))
# createeLineGraph([32.55813953488372, 32.55813953488372, 27.906976744186046, 18.6046511627907, 34.883720930232556, 30.23255813953488, 25.581395348837212, 18.6046511627907, 44.18604651162791, 27.906976744186046])
# x = [
#     [76.16, 25.22, 32.68, 45.76, 70.27, 65.52, 63.04, 55.50, 70.00, 35.51, 63.94, 55.55, 79.35, 92.28, 64.10, 52.22, 64.91, 54.72, 24.77, 14.28, 53.36, 25.65, 61.37, 9.09],
#     [36.57, 10.43, 25.64, 45.36, 68.62, 77.85, 47.83, 64.92, 57.75, 29.30, 57.11, 54.99, 64.19, 71.46, 73.08, 79.48, 97.77, 43.00, 7.50, 22.90, 80.00, 32.57, 66.47, 23.53],
#     [94.32, 68.58, 65.36, 73.12, 73.42, 95.27, 82.88, 83.47, 71.00, 68.22, 80.61, 84.81, 76.12, 99.33, 89.45, 97.21, 75.43, 77.02, 39.82, 87.66, 69.23, 65.72, 94.94, 58.48],
#     [99.66, 75.6637, 71.7073, 85.28, 68.8811, 96.5665, 80.9783, 85.2174, 75.2, 65.4206, 84.6939, 83.7037, 82.5806, 95.3606, 88.8889, 97.5594, 92.9825, 81.0811, 43.3628, 92.2944, 72.5962, 79.4326, 96.092, 65.1515],
#     [98.5523, 82.7434, 85.3659, 81.44, 70.979, 96.4235, 84.7826, 85.5072, 74.7, 71.4953, 80.6122, 80, 83.2258, 92.0228, 99.4994, 91.2281, 81.7568, 45.4277, 97.9221, 85.5769, 74.4681, 95.4023, 95.8586]
# ]

# lists = [[75.69444444444444, 75.69444444444444, 75.69444444444444, 76.11111111111111, 76.66666666666667, 75.69444444444444, 76.66666666666667, 75.69444444444444, 75.69444444444444, 76.66666666666667], 
#          [23.888888888888886, 23.888888888888886, 22.77777777777777, 19.444444444444443, 22.77777777777777, 24.444444444444443, 23.33333333333333, 22.77777777777777, 24.444444444444443, 25.555555555555557], 
#          [48.19277108433735, 45.78313253012048, 51.80722891566265, 53.6144578313253, 42.168674698795186, 53.6144578313253, 31.92771084337349, 42.77108433734939, 38.55421686746988, 48.19277108433735], 
#          [59.76095617529881, 58.964143426294825, 63.54581673306773, 57.37051792828686, 66.73306772908366, 60.55776892430279, 54.581673306772906, 65.5378486055777, 60.95617529880478, 66.73306772908366],
#          [72.36842105263158, 72.80701754385966, 72.80701754385966, 71.49122807017544, 75.0, 73.24561403508771, 72.36842105263158, 74.56140350877193, 71.9298245614035, 74.56140350877193],
#          [89.08765652951699, 91.23434704830053, 90.69767441860465, 90.51878354203936, 93.73881932021467, 93.38103756708408, 93.38103756708408, 88.72987477638641, 91.23434704830053, 88.72987477638641],
#          [63.94557823129252, 81.97278911564626, 84.35374149659864, 64.28571428571429, 81.97278911564626, 60.204081632653065, 63.94557823129252, 65.98639455782312, 82.6530612244898, 68.70748299319727],
#          [83.75451263537906, 83.75451263537906, 83.75451263537906, 83.75451263537906, 84.11552346570397, 83.75451263537906, 83.75451263537906, 83.93501805054152, 84.11552346570397, 67.87003610108303],
#          [71.375, 70.0, 70.75, 70.875, 71.625, 70.625, 70.125, 70.125, 67.125, 70.125],
#          [40.35087719298245, 45.02923976608187, 39.76608187134503, 43.859649122807014, 36.25730994152047, 45.02923976608187, 45.02923976608187, 37.42690058479532, 43.859649122807014, 45.614035087719294],
#          [82.12765957446808, 80.0, 82.12765957446808, 80.0, 83.40425531914893, 80.0, 80.0, 82.12765957446808, 80.0, 80.0],
#          [76.85185185185185, 76.85185185185185, 77.77777777777779, 77.31481481481481, 76.85185185185185, 76.85185185185185, 76.85185185185185, 75.0, 75.92592592592592, 75.46296296296296],
#          [87.90322580645162, 87.09677419354838, 87.90322580645162, 87.09677419354838, 87.90322580645162, 87.09677419354838, 88.70967741935483, 87.90322580645162, 87.90322580645162, 87.90322580645162],
#          [92.1523178807947, 92.1523178807947, 92.01986754966887, 92.1523178807947, 92.1523178807947, 92.1523178807947, 93.64238410596026, 92.1523178807947, 92.1523178807947, 92.1523178807947],
#          [78.4452296819788, 77.03180212014135, 77.03180212014135, 77.03180212014135, 78.4452296819788, 77.03180212014135, 79.50530035335689, 69.25795053003534, 83.03886925795054, 83.74558303886926],
#          [70.18366549433372, 71.0042985541227, 78.11645173896054, 78.27276279796796, 78.66354044548652, 78.11645173896054, 78.11645173896054, 71.16060961313013, 78.11645173896054, 78.11645173896054],
#          [77.08333333333334, 79.16666666666666, 79.16666666666666, 75.0, 77.08333333333334, 83.33333333333334, 77.08333333333334, 75.0, 77.08333333333334, 79.16666666666666],
#          [77.96610169491525, 71.1864406779661, 65.2542372881356, 71.1864406779661, 68.64406779661016, 63.559322033898304, 65.2542372881356, 55.08474576271186, 68.64406779661016, 57.6271186440678],
#          [24.354243542435427, 24.723247232472318, 30.62730627306273, 18.08118081180811, 28.78228782287823, 25.461254612546128, 13.284132841328415, 19.926199261992622, 26.56826568265683, 28.04428044280442],
#          [42.803030303030305, 35.3896103896104, 38.63636363636363, 42.803030303030305, 40.85497835497836, 36.201298701298704, 42.04545454545455, 42.316017316017316, 34.686147186147195, 37.44588744588744],
#          [58.43373493975904, 58.43373493975904, 58.43373493975904, 58.43373493975904, 58.43373493975904, 58.43373493975904, 58.43373493975904, 58.43373493975904, 58.43373493975904, 58.43373493975904],
#          [45.21354933726068, 45.06627393225332, 43.888070692194404, 42.857142857142854, 37.84977908689249, 37.26067746686304, 45.36082474226804, 41.82621502209131, 49.63181148748159, 43.446244477172314],
#          [96.85714285714285, 96.85714285714285, 88.85714285714286, 96.85714285714285, 92.85714285714286, 96.85714285714285, 96.85714285714285, 96.85714285714285, 96.85714285714285, 96.85714285714285],
#          [34.09090909090909, 34.09090909090909, 31.565656565656568, 34.09090909090909, 34.09090909090909, 31.565656565656568, 34.09090909090909, 34.09090909090909, 34.09090909090909, 34.09090909090909]
#          ]
# max_list = []

# for lst in lists:
#     max_list.append(max(lst))

# print(max_list)

listA = [76.66666666666667, 25.555555555555557, 53.6144578313253, 66.73306772908366, 75.0, 93.73881932021467, 84.35374149659864, 84.11552346570397, 71.625, 45.614035087719294, 83.40425531914893, 77.77777777777779, 88.70967741935483, 93.64238410596026, 83.74558303886926, 78.66354044548652, 83.33333333333334, 77.96610169491525, 30.62730627306273, 42.803030303030305, 58.43373493975904, 49.63181148748159, 96.85714285714285, 34.09090909090909]
listB = [76.16, 25.22, 32.68, 45.76, 70.27, 65.52, 63.04, 55.50, 70.00, 35.51, 63.94, 55.55, 79.35, 92.28, 64.10, 52.22, 64.91, 54.72, 24.77, 14.28, 53.36, 25.65, 61.37, 9.09]
listC = [94.32, 68.58, 65.36, 73.12, 73.42, 95.27, 82.88, 83.47, 71.00, 68.22, 80.61, 84.81, 76.12, 99.33, 89.45, 97.21, 75.43, 77.02, 39.82, 87.66, 69.23, 65.72, 94.94, 58.48]
listD = [99.66, 75.6637, 71.7073, 85.28, 68.8811, 96.5665, 80.9783, 85.2174, 75.2, 65.4206, 84.6939, 83.7037, 82.5806, 95.3606, 88.8889, 97.5594, 92.9825, 81.0811, 43.3628, 92.2944, 72.5962, 79.4326, 96.092, 65.1515]
listE = [98.5523, 82.7434, 85.3659, 81.44, 70.979, 96.4235, 84.7826, 85.5072, 74.7, 71.4953, 80.6122, 80, 83.2258, 92.0228, 99.4994, 91.2281, 81.7568, 45.4277, 97.9221, 85.5769, 74.4681, 95.4023, 95.8586, 1]

max_values = []
max_lists = []

print(len(listA), len(listB), len(listC), len(listD), len(listE))

for i in range(len(listA)):
    max_val = max(listA[i], listB[i], listC[i], listD[i], listE[i])
    max_values.append(max_val)
    if max_val == listA[i]:
        max_lists.append('AutoDes')
    elif max_val == listB[i]:
        max_lists.append('ZeroR')
    elif max_val == listC[i]:
        max_lists.append('Decision Table')
    elif max_val == listD[i]:
        max_lists.append('MultiClassClassifier')
    elif max_val == listE[i]:
        max_lists.append('Decorate')

max_lists_dic = {'AutoDes': 0, 'ZeroR': 0, 'Decision Table': 0, 'MultiClassClassifier': 0, 'Decorate': 0}
for i in max_lists:
    max_lists_dic[i] += 1
print(max_lists_dic)

# reverse order of x
# x = reversed(x)
# boxplot(x)