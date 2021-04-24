####################### Getting to know PySpark

### Using Spark in Python

# Creating the connection is as simple as creating an instance of the 
# SparkContext class. The class constructor takes a few optional arguments that 
# allow you to specify the attributes of the cluster you're connecting to.

# An object holding all these attributes can be created with the SparkConf() 
# constructor. Take a look at the documentation for all the details!

# For the rest of this course you'll have a SparkContext called sc already 
# available in your workspace.

# How do you connect to a Spark cluster from PySpark? 
# Create an instance of the SparkContext class


## Examining The SparkContext

# Verify SparkContext
print(sc)

# Print Spark version
print(sc.version)


### Using DataFrames

# Spark's core data structure is the Resilient Distributed Dataset (RDD). 
# This is a low level object that lets Spark work its magic by splitting data 
# across multiple nodes in the cluster. However, RDDs are hard to work with 
# directly, so in this course you'll be using the Spark DataFrame abstraction 
# built on top of RDDs.

# To start working with Spark DataFrames, you first have to create a 
# SparkSession object from your SparkContext. You can think of the SparkContext 
# as your connection to the cluster and the SparkSession as your interface with 
# that connection.

# Remember, for the rest of this course you'll have a SparkSession called spark 
# available in your workspace!

# Which of the following is an advantage of Spark DataFrames over RDDs?
# Operations using DataFrames are automatically optimized.


## Creating a SparkSession

# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession 

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)


## Viewing tables

# Print the tables in the catalog
print(spark.catalog.listTables())


## Are you query-ious?

# Don't change this query
query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights
flights10 = spark.sql(query)

# Show the results
flights10.show()


## Pandafy a Spark DataFrame

# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())


## Put some Spark in your data

# maybe you want to go the other direction, and put a pandas DataFrame into a 
# Spark cluster! The SparkSession class has a method for this as well.

# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())
# [Table(name='temp', database=None, description=None, tableType='TEMPORARY', isTemporary=True)]

 
## Dropping the middle man (spark.read.csv)

# Now you know how to put data into Spark via pandas, but you're probably 
# wondering why deal with pandas at all? Wouldn't it be easier to just read a 
# text file straight into Spark?
# your SparkSession has a .read attribute which has several methods for 
# reading different data sources into Spark DataFrames. Using these you can 
# create a DataFrame from a .csv file just like with regular pandas DataFrames!

# The variable file_path is a string with the path to the file airports.csv. 
# This file contains information about different airports all over the 
# world.

# A SparkSession named spark is available in your workspace.

# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = spark.read.csv(file_path, header = True)

# Show the data
airports.show()




############################# Manipulating data


## Creating columns

# Create the DataFrame flights
flights = spark.table("flights")

# Show the head
flights.show()

# Update flights to include a new column called duration_hrs, that contains 
# the duration of each flight in hours.
flights = flights.withColumn("duration_hrs", flights.air_time / 60)


## Filtering Data

# Let's take a look at the .filter() method. As you might suspect, this is 
# the Spark counterpart of SQL's WHERE clause. The .filter() method takes 
# either an expression that would follow the WHERE clause of a SQL expression 
# as a string, or a Spark Column of boolean (True/False) values.
flights.filter("air_time > 120").show() # or
flights.filter(flights.air_time > 120).show()


# Filter flights by passing a SQL string
long_flights1 = flights.filter("distance > 1000")

# Filter flights by passing a column of boolean values
long_flights2 = flights.filter(flights.distance > 1000)

# Print the data to check they're equal
long_flights1.show()
long_flights2.show()


## Selecting

# Select the first set of columns
selected1 = flights.select("tailnum", "origin", "dest")

# Select the second set of columns
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Define first filter
filterA = flights.origin == "SEA"

# Define second filter
filterB = flights.dest == "PDX"

# Filter the data, first by filterA then by filterB
selected2 = temp.filter(filterA).filter(filterB)


## Selecting II

# similar to SQL, you can also use the .select() method to perform column-wise operations.

# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")



## Aggregating

# All of the common aggregation methods, like .min(), .max(), and .count() 
# are GroupedData methods.

# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()

# Find the longest flight from SEA in terms of air time
flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()


## Aggregating II

# Average duration of Delta flights  that left SEA. 
flights.filter(flights.origin == "SEA").filter(flights.carrier == "DL").groupBy().avg("air_time").show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()


## Grouping and Aggregating I

# Group by tailnum
by_plane = flights.groupBy("tailnum")

# Number of flights each plane made
by_plane.count().show()

# Group by origin
by_origin = flights.groupBy("origin")

# Average duration of flights from PDX and SEA
by_origin.avg("air_time").show()


## Grouping and Aggregating II

# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy("month", "dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation of departure delay
by_month_dest.agg(F.stddev("dep_delay")).show()


### Joining

# Examine the data
print(airports.show())

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, on="dest", how="leftouter")

# Examine the new DataFrame
print(flights_with_airports.show())




################ Getting started with machine learning pipelines

# PySpark has built-in, cutting-edge machine learning routines, along with 
# utilities to create full machine learning pipelines. 

### Machine Learning Pipelines

# At the core of the pyspark.ml module are the Transformer and Estimator 
# classes. Almost every other class in the module behaves similarly to 
# these two basic classes.

# Transformer classes have a .transform() method that takes a DataFrame 
# and returns a new DataFrame; usually the original one with a new column 
# appended.

# Estimator classes all implement a .fit() method. These methods also take 
# a DataFrame, but instead of returning another DataFrame they return a model 
# object.


## Join the DataFrames

# Rename year column
planes = planes.withColumnRenamed("year", "plane_year")

# Join the DataFrames
model_data = flights.join(planes, on="tailnum", how="leftouter")


## Data types

# Before you get started modeling, it's important to know that Spark only
# handles numeric data. That means all of the columns in your DataFrame must 
# be either integers or decimals (called 'doubles' in Spark).


## String to integer

# t's important to know that Spark only handles numeric data. That means 
# all of the columns in your DataFrame must be either integers or decimals 
# (called 'doubles' in Spark).

#  you can use the .cast() method in combination with the .withColumn() 
# method. It's important to note that .cast() works on columns, 
# while .withColumn() works on DataFrames.

# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))


## Create a new column

# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)


## Making a Boolean

# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")



### Strings and factors

# encode a categorical feature as a one-hot vector?


## Carrier variable

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol="carrier_fact")


## Destination variable

# Create a StringIndexer
dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")


## Assemble a vector

# The last step in the Pipeline is to combine all of the columns containing 
# our features into a single column

# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")


## Create the pipeline

# Pipeline is a class in the pyspark.ml module that combines all the 
# Estimators and Transformers that you've already created. This lets you 
# reuse the same modeling process over and over again by wrapping it up in 
# one simple object. 

# Import Pipeline
from pyspark.ml import Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])




### Test vs Train

# In Spark it's important to make sure you split the data after all the 
# transformations. This is because operations like StringIndexer don't 
# always produce the same index even when given the same list of strings.


## Transform the data

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)


## Split the data

# Split the data into training and test sets
# training with 60% of the data, and test with 40%
training, test = piped_data.randomSplit([.6, .4])




###################### Model tuning and selection


## Create the modeler

# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()


### Cross validation

# You'll be using cross validation to choose the hyperparameters by 
# creating a grid of the possible pairs of values for the two 
# hyperparameters, elasticNetParam and regParam, and using the cross 
# validation error to compare all the different models so you can choose 
# the best one!


## Create the evaluator

# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")


## Make a grid

# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()



## Make the validator

# Create the CrossValidator
# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )



## Fit the model(s)

# Fit cross validation models
models = cv.fit(training)

# Extract the best model
best_lr = models.bestModel

# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)


## Evaluate the model

# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))

