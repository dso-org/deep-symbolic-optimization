
IMPORTANT: sometimes the characte '"' may cause a problem because of the enconding. 
I tried " encoding='utf-8' " but that did not work. So, I had to manually substitute problematic characters

pandas is being used to read with a package called chardet that can identify the enconding used.

The code implemented in this package reads the data sets from a csv, parses it (using sympy) and loads them in a dictionary.
The following classes were implemented:
  ParseData: parses the data
  CheckData: checks all data sets stored in the csv 

IMPORTANT: sympy may return a parse tree that is not binary. So, in ParseData, there is a method that converts from an n-ary tree to a binary tree

Example:

from parsedata import ParseData,CheckData

#-- stores parsing results on file 'parseresults.csv'
ck = CheckData('datasets.csv','parseresults.csv')

#-- parsing results will be shown on screen
ck = CheckData('datasets.csv')

#-- retrives and parses the data set "Burks(*)"
pd = ParseData('datasets.csv',"Burks(*)")

#-- parses data:
ecode = pd.parse()
print("Results of the pasrsing: "+str(ecode))

#-- actual retrieval of the data, if parsing is ok, i.e., no errors
if not ecode:
   data = pd() 

Return codes of the parse method:
#--- Return types:
#--- 0: no error
#--- 1: data set not found
#--- 2: parsing error
#--- 3: error in training set
#--- 4: error in testing set


