"""import numpy as np
import matplotlib.pyplot as plt
time = np.linspace(0, 10, 100)
amplitude = np.exp(-time/3) * np.sin(2 * np.pi * time)
#create the canvas
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

#plot the data along the axis 
ax.plot(time, amplitude, color='crimson', linewidth=2, label="Sensor A Displacement")
ax.set_title('Damped Harmonic Response of Structure', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Displacement (mm)', fontsize=12)
# Enable engineering gridlines
ax.grid(True, linestyle='--', alpha=0.6)
# Configure the Legend
ax.legend(loc='upper right', frameon=True, shadow=True)
plt.show()
import matplotlib.pyplot as plt
from networkx import display
import numpy as np
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
x = np.linspace(0, 4*np.pi, 200)
line, = ax.plot(x, np.sin(x), color='crimson', linewidth=2)
line.set_ydata(np.cos(x))
plt.show()
import pandas as pd
mydataset = {'name': ["Ayus", "Kashu", "Rashi"],
'age': [20, 21, 19]}  
mypaglu = pd.DataFrame(mydataset)
print(mypaglu)"""
from pydoc import describe
import re
import pandas as pd
import numpy as np
data = {'Visitor_ID': [f'V{1000 + i}' for i in range(20)],
    'Name': ['ABHISHEK SANDEEP      ZADE     ', 'ARNAV AJAY DESHPANDE.. ', '.. .. ASHWINI LALCHAND MUNDAWARE/ ', 'GAYATRI SURESH GAIKWAD', 'HARSHADA GANESH CHAUDHARI',
             'VAIBHAVI HARISHWAR PATIL', np.nan, 'VISHAKHA PUNDLIK JADHAV', 'YASH BHARAT SOLUNKE/', 'VIVEK SANTOSH KHANDWE',
             'VISHAKHA PUNDLIK JADHAV', 'Tanushree chhanwal', 'Shruti jaiswal', 'Shriyash Sulakhe', 'YASH BHARAT SOLUNKE',
             np.nan, 'ARNAV AJAY DESHPANDE', 'RUTUJA SANTOSH THOTE', 'ROHIT DILIP BILWAL', 'RITESH SHIVAJI BAIRAGI'],
    'Age': [25, 23, 22, np.nan, 21, 25, 24, 24, 28, np.nan,
            22, 23, 25, 27, 25, 30, 31, 26, 19, 19],
    'Ticket_Price': [500, 750, 500, 1000, np.nan, 500, 700, 650, 750, 1000,
                      500, 800, np.nan, 750, 500, 700, 900, 850, 750, np.nan],
    'Check_In_Time': ['10:00 AM', '10:30 AM', '11:00 AM', '11:30 AM', np.nan,
                       '10:00 AM', '12:00 PM', '12:30 PM', '01:00 PM', '01:30 PM',
                       '11:00 AM', '02:00 PM', np.nan, '02:30 PM', '10:00 AM',
                       '12:00 PM', '03:00 PM', '03:30 PM', '04:00 PM', np.nan],
    'City':['Delhi','Aurangabad', 'Mumbai','Bombay','New Delhi','NDL','Chennai','Chenai','Chennaai','Bangalore',
            'Delhi','Pune', 'New Delhi','Bombay','New Delhi','NDL','Indore','Bangalore','Ujjain','Bangalore'],
    'State': ['Delhi','Maharastra', 'Maharastra','Maharastra','Delhi','Delhi','Tamilnadu','Tamilnadu','Tamilnadu','Karnataka',
            'Delhi','Maharastra', 'Delhi','Maharastra','Delhi','Delhi','Madhya Pradesh','Karnataka','Madhya Pradesh','Karnataka']}

df = pd.DataFrame(data)

# Save DataFrame as CSV
df.to_csv('data.csv', index=False)

print("CSV file created successfully!")

# Read CSV back
df = pd.read_csv('data.csv')

print(df.to_string())
df.info()
print(df.describe())
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.isnull().sum())
