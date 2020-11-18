#----------------------------------------------------------------------------
#Decision Tree Example
#----------------------------------------------------------------------------

# STEP (1): Import libraries including sklearn decision tree classifier
#----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

# STEP (2): Import the dataset (here: flag dataset)
#----------------------------------------------------------------------------
dataset = pd.read_csv('flag_semicolon.csv',
                      sep=';',
                      names=['name',
                             'landmass', #1=N.America, 2=S.America, 3=Europe, 4=Africa, 5=Asia, 6=Oceania
                             'zone',
                             'area',
                             'population',
                             'language',
                             'religion',
                             'bars',
                             'stripes',
                             'colours',
                             'red',
                             'green',
                             'blue',
                             'gold',
                             'white',
                             'black',
                             'orange',
                             'mainhue',
                             'circles',
                             'crosses',
                             'saltires',
                             'quarters',
                             'sunstars',
                             'crescent',
                             'triangle',
                             'icon',
                             'animate',
                             'text',
                             'topleft',
                             'botright'])


# STEP (3): Modify the dataset
#----------------------------------------------------------------------------
#replace continents numbers by strings
dataset['landmass'] = dataset['landmass'].map({1: 'North America',
                                               2: 'South America',
                                               3: 'Europe',
                                               4: 'Africa',
                                               5: 'Asia',
                                               6: 'Oceania'})


#Dont split the data based on name of flag (doesnt make sense)
dataset=dataset.drop('name',axis=1)
#I also remove mainhue, topleft and botright because DecisionTreeClassifier cannot handle strings and I dont have enough time to encode this stuff now
dataset=dataset.drop('mainhue',axis=1)
dataset=dataset.drop('topleft',axis=1)
dataset=dataset.drop('botright',axis=1)
#also remove values that are categorial; sklearn DecisionTreeClassifier is NOT made for features as enums etc since it will treat every feature as numeric!
#So only useful for int, float, boolean.
dataset=dataset.drop('zone',axis=1)
dataset=dataset.drop('language',axis=1)
dataset=dataset.drop('religion',axis=1)
#also remove area and population (only look at the features of the flag itself)
dataset=dataset.drop('area',axis=1)
dataset=dataset.drop('population',axis=1)

# STEP (4): Split data into training and testing data set (lets say 2/3 training data and 1/3 testing data)
#----------------------------------------------------------------------------
train_and_test_features = dataset.loc[:, dataset.columns != 'landmass']  #set everything but continent (landmass)
# python starts counting with 0, so name is 0th row, landmass is 1st row) as feature
train_and_test_targets =  dataset.loc[:, dataset.columns == 'landmass']  #or: dataset.iloc[:, 1]  #set continent as target

print("\n Features:")
print(train_and_test_features)
#print(train_and_test_features.to_string(index=False))
print("\n Targets:")
print(train_and_test_targets)
#print(train_and_test_targets.to_string(index=False))

#with shuffle every time the data tree is random/different
train_features, test_features, train_targets, test_targets \
    = sklearn.model_selection.train_test_split(train_and_test_features, train_and_test_targets, test_size=0.333, shuffle=True)

# STEP (5): Train the model
#----------------------------------------------------------------------------
my_tree = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(train_features, train_targets) #use entropy
#There's an element of randomness to Decision Tree algorithms --> different results for every compilation
#fix random_state for repeatable result for testing: same "seed" value each time by using DecisionTreeClassifier(random_state=0)

# STEP (5.2): Predict classes of the test data
#----------------------------------------------------------------------------
#prediction = my_tree.predict(test_features)
#print(prediction)

# STEP (6): Calculate the accuracy
#----------------------------------------------------------------------------
print("Prediction accuracy ≈", round(my_tree.score(test_features, test_targets)*100, 2), "%")

# STEP (7): Plot the decision tree
#----------------------------------------------------------------------------
cn = my_tree.classes_ #class names (1,2,3... for North America etc.)
fn = [ #feature names (zone, area etc.)
        'bars',
        'stripes',
        'colours',
        'red',
        'green',
        'blue',
        'gold',
        'white',
        'black',
        'orange',
        'circles',
        'crosses',
        'saltires',
        'quarters',
        'sunstars',
        'crescent',
        'triangle',
        'icon',
        'animate',
        'text']

#Possibility 1: Text tree
##################################
output_text = export_text(my_tree,
                          feature_names = fn)
print(output_text)

#Possibility 2: Matplotlib
##################################
#sklearn.tree.plot_tree
# Setting dpi higher to make image clearer than default
fig, axes = plt.subplots(nrows = 1,
                         ncols = 1,
                         figsize = (10,10),
                         dpi=1500
                         )
tree.plot_tree(my_tree,
               feature_names = fn,
               filled = True)

fig.savefig('matplotlib_output.png')

#Possibility 3: Graphviz
##################################
#sklearn.tree.export_graphviz(tree)


