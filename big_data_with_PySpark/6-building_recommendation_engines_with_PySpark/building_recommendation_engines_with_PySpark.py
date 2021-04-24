############################## Recommendations Are Everywhere

### Why learn how to build recommendation engines?

## See the power of a recommendation engine

# View TJ_ratings
TJ_ratings.show()

# Generate recommendations for users
get_ALS_recs(["Taylor","Jane"]) 


### Recommendation engine types and data types

## Ratings data types

# Group the data by "Genre"
markus_ratings.groupBy("Genre").sum().show()



########################### How does ALS work?

### Overview of matrix multiplication

## Matrix multiplication

# Use the .head() method to view the contents of matrices a and b
print("Matrix A: ")
print (a.head())

print("Matrix B: ")
print (b.head())

# Complete the matrix with the product of matrices a and b
product = np.array([[10,12], [15,18]])

# Run this validation to see how your estimate performs
product == np.dot(a,b)


## Matrix multiplication part II

# Print the dimensions of C
print(C.shape)

# Print the dimensions of D
print(D.shape)

# Can C and D be multiplied together?
C_times_D = None


### Overview of matrix factorization

## Matrix factorization

# Take a look at Matrix G using the following print function
print("Matrix G:")
print(G)

# Take a look at the matrices H, I, and J and determine which pair of those matrices will produce G when multiplied together. 
print("Matrix H:")
print(H)
print("Matrix I:")
print(I)
print("Matrix J:")
print(J)

# Multiply the two matrices that are factors of the matrix G
prod = np.matmul(H, J)
print(G == prod)


## Non-negative matrix factorization

# View the L, U, W, and H matrices.
print("Matrices L and U:") 
print(L)
print(U)

print("Matrices W and H:")
print(W)
print(H)

# Calculate RMSE between LU and M
print("RMSE of LU: ", getRMSE(LU, M))

# Calculate RMSE between WH and M
print("RMSE of WH: ", getRMSE(WH, M))


### How ALS alternates to generate predictions

## Estimating recommendations

# Multiply factor matrices
UP = np.matmul(U,P)

# Convert to pandas DataFrame
print(pd.DataFrame(UP, columns = P.columns, index = U.index))


## RMSE as ALS alternates

# Use getRMSE(preds, actuals) to calculate the RMSE of matrices T and F1.
getRMSE(F1, T)
# F1:  2.4791263858912522

# Create list of F2, F3, F4, F5, and F6
Fs = [F2, F3, F4, F5, F6]

# Calculate RMSE for F2, F3, F4, F5, and F6.
getRMSEs(Fs, T)
# F2: 0.4389326310548279
# F3: 0.17555006757053257
# F4: 0.15154042416388636
# F5: 0.13191130368008455
# F6: 0.04533823201006271



### Data preparation for Spark ALS

##  Correct format and distinct users

# Import monotonically_increasing_id and show R
from pyspark.sql.functions import monotonically_increasing_id
R.show()

# Use the to_long() function to convert the dataframe to the "long" format.
ratings = to_long(R)
ratings.show()

# Get unique users and repartition to 1 partition
users = ratings.select("User").distinct().coalesce(1)

# Create a new column of unique integers called "userId" in the users dataframe.
users = users.withColumn("userId", monotonically_increasing_id()).persist()
users.show()


## Assigning integer id's to movies

# Extract the distinct movie id's
movies = ratings.select("Movie").distinct() 

# Repartition the data to have only one partition.
movies = movies.coalesce(1) 

# Create a new column of movieId integers. 
movies = movies.withColumn("movieId", monotonically_increasing_id()).persist() 

# Join the ratings, users and movies dataframes
movie_ratings = ratings.join(users, "User", "left").join(movies, "Movie", "left")
movie_ratings.show()


### ALS parameters and hyperparameters

## Build out an ALS model

# Split the ratings dataframe into training and test data
(training_data, test_data) = ratings.randomSplit([0.8, 0.2], seed=42)

# Set the ALS hyperparameters
from pyspark.ml.recommendation import ALS
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank =10, maxIter =15, regParam =0.1,
          coldStartStrategy="drop", nonnegative =True, implicitPrefs = False)

# Fit the mdoel to the training_data
model = als.fit(training_data)

# Generate predictions on the test_data
test_predictions = model.transform(test_data)
test_predictions.show()


## Build RMSE evaluator

# Import RegressionEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

# Complete the evaluator code
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Extract the 3 parameters
print(evaluator.getMetricName())
print(evaluator.getLabelCol())
print(evaluator.getPredictionCol())


## Get RMSE

# Evaluate the "test_predictions" dataframe
RMSE = evaluator.evaluate(test_predictions)

# Print the RMSE
print (RMSE)



###################### Recommending Movies 

### Introduction to the MovieLens dataset

## Viewing the MovieLens Data

# File location and type
file_location = "/FileStore/tables/df-5.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

ratings = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(ratings)

# Look at the column names
print(ratings.columns)

# Look at the first few rows of data
print(ratings.show())


## Calculate sparsity

# Count the total number of ratings in the dataset
numerator = ratings.select("rating").count()

# Count the number of distinct userIds and distinct movieIds
num_users = ratings.select("userId").distinct().count()
num_movies = ratings.select("movieId").distinct().count()

# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_movies

# Divide the numerator by the denominator
sparsity = (1.0 - (numerator *1.0)/denominator)*100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")


## The GroupBy and Filter methods

# Two common methods that will be helpful to you as you aggregate summary statistics in Spark 
# are the .filter() and the .groupBy() methods.

# Import the requisite packages
from pyspark.sql.functions import col

# View the ratings dataset
ratings.show()

# Filter to show only userIds less than 100
ratings.filter(col("userId") < 100).show()

# Group data by userId, count ratings
ratings.groupBy("userId").count().show()


## MovieLens Summary Statistics

# Min num ratings for movies
print("Movie with the fewest ratings: ")
ratings.groupBy("movieId").count().select(min("count")).show()

# Avg num ratings per movie
print("Avg num ratings per movie: ")
ratings.groupBy("movieId").count().select(avg("count")).show()

# Min num ratings for user
print("User with the fewest ratings: ")
ratings.groupBy("userId").count().select(min("count")).show()

# Avg num ratings per users
print("Avg num ratings per user: ")
ratings.groupBy("userId").count().select(avg("count")).show()


## View Schema

# Use .printSchema() to see the datatypes of the ratings dataset
ratings.printSchema()

# Tell Spark to convert the columns to the proper data types
ratings = ratings.select(ratings.userId.cast("integer"), ratings.movieId.cast("integer"), ratings.rating.cast("double"))

# Call .printSchema() again to confirm the columns are now in the correct format
ratings.printSchema()


### ALS model buildout on MovieLens Data

## Create test/train splits and build your ALS model

# Import the required functions
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create test and train set
(train, test) = ratings.randomSplit([0.8, 0.2], seed = 1234)

# Create ALS model
als = ALS(userCol="userId", itemCol="userId", ratingCol="rating", nonnegative = True, implicitPrefs = False)

# Confirm that a model called "als" was created
type(als)

## Tell Spark how to tune your ALS model

# Import the requisite items
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Add hyperparameters and their respective values to param_grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 50, 100, 150]) \
            .addGrid(als.maxIter, [5, 50, 100, 200]) \
            .addGrid(als.regParam, [.01, .05, .1, .15]) \
            .build()

           
# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction") 
print ("Num models to be tested: ", len(param_grid))


## Build your cross validation pipeline

# Build cross validation using CrossValidator
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

# Confirm cv was built
print(cv)


## Best Model and Best Model Parameters

#Fit cross validator to the 'train' dataset
model = cv.fit(train)

#Extract best model from the cv model above
best_model = model.bestModel


## Best Model and Best Model Parameters

# Print best_model
print(type(best_model))

# Complete the code below to extract the ALS model parameters
print("**Best Model**")

# Print "Rank"
print("  Rank:", best_model.getRank())

# Print "MaxIter"
print("  MaxIter:", best_model.getMaxIter())

# Print "RegParam"
print("  RegParam:", best_model.getRegParam())



### Model Performance Evaluation

## Generate predictions and calculate RMSE

test_predictions = best_model.transform(test)

# View the predictions 
test_predictions.show()

# Calculate and print the RMSE of test_predictions
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)


## Do recommendations make sense

# Look at user 60's ratings
print("User 60's Ratings:")
original_ratings.filter(col("userId") == 60).sort("rating", ascending = False).show()

# Look at the movies recommended to user 60
print("User 60s Recommendations:")
recommendations.filter(col("userId") == 60).show()

# Look at user 63's ratings
print("User 63's Ratings:")
original_ratings.filter(col("userId") == 63).sort("rating", ascending = False).show()

# Look at the movies recommended to user 63
print("User 63's Recommendations:")
recommendations.filter(col("userId") == 63).show()




############################# What if you don't have customer ratings?

### Introduction to the Million Songs Dataset

## MSD summary statistics

# Look at the data
msd.show()

# Count the number of distinct userIds
user_count = msd.select("userId").distinct().count()
print("Number of users: ", user_count)

# Count the number of distinct songIds
song_count = msd.select("songId").distinct().count()
print("Number of songs: ", song_count)


## Grouped summary statistics

# Min num implicit ratings for a song
print("Minimum implicit ratings for a song: ")
msd.filter(col("num_plays") > 0).groupBy("songId").count().select(min("count")).show()

# Avg num implicit ratings per songs
print("Average implicit ratings per song: ")
msd.filter(col("num_plays") > 0).groupBy("songId").count().select(avg("count")).show()

# Min num implicit ratings from a user
print("Minimum implicit ratings from a user: ")
msd.filter(col("num_plays") > 0).groupBy("userId").count().select(min("count")).show()

# Avg num implicit ratings for users
print("Average implicit ratings per user: ")
msd.filter(col("num_plays") > 0).groupBy("userId").count().select(avg("count")).show()


## Add zeros

# View the data
Z.show()

# Extract distinct userIds and productIds
users = Z.select("userId").distinct()
products = Z.select("productId").distinct()

# Cross join users and products
cj = users.crossJoin(products)

# Join cj and Z
Z_expanded = cj.join(Z, ["userId", "productId"], "left").fillna(0)

# View Z_expanded
Z_expanded.show()



### Evaluating implicit ratings models

## Specify ALS hyperparameters

# Complete the lists below
ranks = [10, 20, 30, 40]
maxIters = [10, 20, 30, 40]
regParams = [.05, .1, .15]
alphas = [20, 40, 60, 80]


## Build implicit models

# For loop will automatically create and store ALS models
for r in ranks:
    for mi in maxIters:
        for rp in regParams:
            for a in alphas:
                model_list.append(ALS(userCol= "userId", itemCol= "songId", ratingCol= "num_plays", rank = r, maxIter = mi, regParam = rp, alpha = a, coldStartStrategy="drop", nonnegative = True, implicitPrefs = True))

# Print the model list, and the length of model_list
print (model_list, "Length of model_list: ", len(model_list))

# Validate
len(model_list) == (len(ranks)*len(maxIters)*len(regParams)*len(alphas))


## Running a cross-validated implicit ALS model

# Split the data into training and test sets
#(training, test) = msd.randomSplit([0.8, 0.2])

#Building 5 folds within the training set.
#train1, train2, train3, train4, train5 = training.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed = 1)
#fold1 = train2.union(train3).union(train4).union(train5)
#fold2 = train3.union(train4).union(train5).union(train1)
#fold3 = train4.union(train5).union(train1).union(train2)
#fold4 = train5.union(train1).union(train2).union(train3)
#fold5 = train1.union(train2).union(train3).union(train4)

#foldlist = [(fold1, train1), (fold2, train2), (fold3, train3), (fold4, train4), (fold5, train5)]

# Empty list to fill with ROEMs from each model
#ROEMS = []

# Loops through all models and all folds
#for model in model_list:
#    for ft_pair in foldlist:

        # Fits model to fold within training data
#        fitted_model = model.fit(ft_pair[0])

        # Generates predictions using fitted_model on respective CV test data
#        predictions = fitted_model.transform(ft_pair[1])

        # Generates and prints a ROEM metric CV test data
#        r = ROEM(predictions)
#        print ("ROEM: ", r)

    # Fits model to all of training data and generates preds for test data
#    v_fitted_model = model.fit(training)
#    v_predictions = v_fitted_model.transform(test)
#    v_ROEM = ROEM(v_predictions)

    # Adds validation ROEM to ROEM list
#    ROEMS.append(v_ROEM)
#    print ("Validation ROEM: ", v_ROEM)

# Import numpy
import numpy

# Find the index of the smallest ROEM
i = numpy.argmin(ROEMS)
print("Index of smallest ROEM:", i)

# Find ith element of ROEMS
print("Smallest ROEM: ", ROEMS[i])


## Extracting parameters

# Extract the best_model
best_model = model_list[38]

# Extract the Rank
print ("Rank: ", best_model.getRank())

# Extract the MaxIter value
print ("MaxIter: ", best_model.getMaxIter())

# Extract the RegParam value
print ("RegParam: ", best_model.getRegParam())

# Extract the Alpha value
print ("Alpha: ", best_model.getAlpha())
#     Rank:  10
#    MaxIter:  40
#    RegParam:  0.05
#    Alpha:  60.0


### Overview of binary, implicit ratings

## Binary model performance

# Import the col function
from pyspark.sql.functions import col

# Look at the test predictions
binary_test_predictions.show()

# Evaluate ROEM on test predictions
ROEM(binary_test_predictions)

# Look at user 42's test predictions
binary_test_predictions.filter(col("userId") == 42).show()


## Recommendations from binary data

# View user 26's original ratings
print ("User 26 Original Ratings:")
original_ratings.filter(col("userId") == 26).show()

# View user 26's recommendations
print ("User 26 Recommendations:")
binary_recs.filter(col("userId") == 26).show()

# View user 99's original ratings
print ("User 99 Original Ratings:")
original_ratings.filter(col("userId") == 99).show()

# View user 99's recommendations
print ("User 99 Recommendations:")
binary_recs.filter(col("userId") == 99).show()


