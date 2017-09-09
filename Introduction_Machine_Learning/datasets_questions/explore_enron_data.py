#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "total persons : ", len(enron_data.keys())

#  get the first key --> find the values of the keys --> get the length of keys in the enron dataaset
print "feature per person : ", len(enron_data[enron_data.keys()[0]])

# get total POI - person of interest
count = 0
for p in enron_data.keys(): 
    if enron_data[p]["poi"]==1 : count = count + 1
print "person of interest : ", count

print "stock for james prentice:" , enron_data["PRENTICE JAMES"]
print "email to poi : ", enron_data["COLWELL WESLEY"]
print "stock options excersized : ", enron_data["SKILLING JEFFREY K"]

print "total money :", enron_data["SKILLING JEFFREY K"]['total_payments']
print "total money :", enron_data["LAY KENNETH L"]['total_payments']
print "total money :", enron_data["FASTOW ANDREW S"]['total_payments']

s_count = 0
e_count = 0
t_count = 0
pt_count = 0
poi_count = 0
for p in enron_data.keys():
    if enron_data[p]['salary']!='NaN' : s_count = s_count + 1
    if enron_data[p]['email_address']!='NaN' : e_count = e_count + 1
    if enron_data[p]['total_payments']=='NaN' : t_count = t_count + 1
    if enron_data[p]['poi']==1 and enron_data[p]['total_payments']=='NaN' : pt_count = pt_count + 1
    if enron_data[p]['poi']==1 : poi_count = poi_count + 1

print "valid salary :", s_count
print "valid email address : ", e_count
print "total keys :", len(enron_data.keys())
print "invalid finance data :", (t_count/len(enron_data.keys())) * 100
print "poi count :", poi_count
print "pt count :", pt_count
