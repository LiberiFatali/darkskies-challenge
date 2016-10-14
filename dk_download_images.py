import os, urllib
import traceback
from urlparse import urlparse


def downloadTestFile(line):
    try:
        folderpath = 'test_release'        
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
            
        filepath = os.path.join(folderpath, os.path.basename(line))
        if not os.path.exists(filepath):            
	    urllib.urlretrieve(line, filepath)	    
    except Exception, e:        
        print line, traceback.format_exc()
        return "error", line

def downloadTrainFile(line):
    try:
        tokens = line.split(',')        
        url = tokens[0].strip()
        label = tokens[1].strip()
        folderpath = os.path.join("train", label)        
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)        
            
        filepath = os.path.join(folderpath, os.path.basename(url))
        if not os.path.exists(filepath):            	    
	    urllib.urlretrieve(url, filepath)	    
    except Exception, e:        
        print url, traceback.format_exc()
        return "error", url

      
with open('train.csv') as csvTrain:
    lines = csvTrain.readlines()
print len(set(lines[1:]))
print 'Downloading...'
for line in set(lines[1:]):
    print line
    downloadTrainFile(line.strip())   