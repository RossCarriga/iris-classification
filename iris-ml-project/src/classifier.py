from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark import SparkContext, SQLContext
from pyspark.sql import Row


sc = SparkContext()
sqlcontext = SQLContext(sc)

def encode_flower_type(flower):
    if flower == 'Iris-setosa':
        return 1.0
    elif flower == 'Iris-versicolor':
        return 2.0
    elif flower == 'Iris-virginica':
        return 3.0
    else:
        return 0.0


def parse_raw_data(line):
    nums = line.split(',')
    flower_type = encode_flower_type(nums[4])
    del nums[4]
    return Row(label=flower_type, features=Vectors.dense(nums))


def parse_test_data(line):
    nums = line.split(',')
    nums_vec = Vectors.dense(nums)
    return Row(label=0.0, features=nums_vec)


rdd = sc.textFile("resources/iris-training.txt").map(lambda x: parse_raw_data(x))
df = rdd.toDF()
df.show()

featureIndexer = VectorIndexer(inputCol='features', outputCol='indexedFeatures').fit(df)

forest = RandomForestClassifier(labelCol='label', featuresCol='indexedFeatures', numTrees=12)
pipeline = Pipeline(stages=[featureIndexer, forest])
model = pipeline.fit(df)

testingData = sc.textFile('resources/iris-testing.txt').map(lambda x: parse_test_data(x))
testingDF = testingData.toDF()

predictions = model.transform(testingDF)
predictions.show()
