import pandas as pd
mydata = {
 "PatientID": [
     "P001","P002","P003","P004","P005",
     "P006","P007","P008","P009","P010",
     "P011","P012","P013","P014","P015"
 ],
 "Patient_Name" :[
     "AYUSH","RAJ","SURAJ","RAVI","KUMAR",
     "RAY", "PAGLU", "RAJESH", "RAMESH", "SURESH",
     "RAJIV", "RANJEET", "RAHUL", "RAVI", "RAJESH"
 ],
 "Age":[
     25,24,21,22,24,
     23,22,21,24,25,
     23,22,21,24,25
 ],
 "Gender":[
        "M","M","M","M","M",
        "M","M","M","M","M",
        "M","M","M","M","M"
 ],
 "Disease":[
        "Fever","Cold","Cough","Fever","Cold",
        "Cough","Fever","Cold","Cough","Fever",
        "Cold","Cough","Fever","Cold","Cough"
 ],
 "Date":[
        "2021-01-01","2021-01-02","2021-01-03","2021-01-04","2021-01-05",
        "2021-01-06","2021-01-07","2021-01-08","2021-01-09","2021-01-10",
        "2021-01-11","2021-01-12","2021-01-13","2021-01-14","2021-01-15"
 ],
 "Status":[
        "Recovered","Discharged","Recovered","Recovered","Critical",
        "Critical","Recovered","Recovered","Discharged","Recovered",
        "Recovered","Discharged","Recovered","Recovered","Critical"
 ],
 "Treatment-Cost":[
        1000,2000,1500,1200,2500,
        2200,1800,1600,2100,1900,
        1700,2300,1400,2400,2600
 ],
}
df = pd.DataFrame(mydata)
df.to_csv("Hospital_Patient_Data.csv", index=False)
print("Hospital patient data has been created successfully.")