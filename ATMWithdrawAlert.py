from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from datetime import datetime
from numpy import array

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf = conf)

# Some functions that convert our CSV input data into numerical
# features for each job candidate
def binary(YN):
    if (YN == 'Y'):
        return 1
    else:
        return 0

def mapLocation(location):
    if (location == 'Office'):
        return 1
    elif (location =='Home'):
        return 2
    else:
        return 0
def getHour(timestamp):
    d=datetime.utcfromtimestamp(float(timestamp))
    if(d.hour < 10):
        return 1
    else:
        return 0
# Convert a list of raw fields from our CSV file to a
# LabeledPoint that MLLib can use. All data must be numerical...
def createLabeledPoints(fields):
    withdrawLocation = mapLocation(fields[0])
    timestamp = getHour(fields[1])
    alert = binary(fields[2])
    
    return LabeledPoint(alert, array([withdrawLocation, timestamp]))

#Load up our CSV file, and filter out the header line with the column names
rawData = sc.textFile("user/ATMDATA.csv")
header = rawData.first()
rawData = rawData.filter(lambda x:x != header)

# Split each line into a list based on the comma delimiters
csvData = rawData.map(lambda x: x.split(","))

# Convert these lists to LabeledPoints
trainingData = csvData.map(createLabeledPoints)

# Create a test candidate, with 10 years of experience, currently employed,
# 3 previous employers, a BS degree, but from a non-top-tier school where
# he or she did not do an internship. You could of course load up a whole
# huge RDD of test candidates from disk, too.
testCandidates = [ array([2,1])]
testData = sc.parallelize(testCandidates)

# Train our DecisionTree classifier using our data set
model = DecisionTree.trainClassifier(trainingData, numClasses=2,
                                     categoricalFeaturesInfo={0:3,1:2},
                                     impurity='gini', maxDepth=5, maxBins=32)

# Now get predictions for our unknown candidates. (Note, you could separate
# the source data into a training set and a test set while tuning
# parameters and measure accuracy as you go!)
predictions = model.predict(testData)
print('Alert prediction:')
results = predictions.collect()
for result in results:
    print(result)

# We can also print out the decision tree itself:
print('Learned classification tree model:')
print(model.toDebugString())
