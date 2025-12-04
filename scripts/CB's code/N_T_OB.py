"""
v230209
FUNCTIONALITY
    Transforms the dataset from its oroginal form (O) to a dataset with buffers (buf)
EXPLANATION
    Transforms the dataset from its oroginal form (O) to a dataset with buffers (buf)
INPUT
    The dataset
OUTPUT
    A dataset with buffers buf
    csv-file "N_C_Dataset_g_buffer.csv"
INPUT PARAMETERS: buffer distance buf
"""

# Read parameters from environment variables
#con = int(os.environ.get('CON', 8))   # number of configurations
#tst = int(os.environ.get('TST', 2))   # number of time stamps
#poi = int(os.environ.get('POI', 2))   # number of points
#dim = int(os.environ.get('DIM', 2))   # number of descriptors/dimensions

#import the necessary modules
import av
import csv
import time

#start time
t_start = time.time()

#open the csv
with open('N_C_Dataset.csv', 'r') as csv_file:
  #read the csv
  csv_reader = csv.reader(csv_file)

  #add a new line
  lines = []
  for line in csv_reader:  # One new buffer point (at distance 3) for each DD (if descriptor and dimension is the same)
    lines.append([line[0],line[1],round((float(line[2])*5+0), 2),round((float(line[3])-av.buffer), 2),line[4]])
    lines.append([line[0],line[1],round((float(line[2])*5+1), 2),round((float(line[3])+av.buffer), 2),line[4]])
    lines.append([line[0],line[1],round((float(line[2])*5+2), 2),line[3],line[4]])
    lines.append([line[0],line[1],round((float(line[2])*5+3), 2),line[3],round((float(line[4])-av.buffer), 2)])
    lines.append([line[0],line[1],round((float(line[2])*5+4), 2),line[3],round((float(line[4])+av.buffer), 2)])

#save the new csv-file
with open('N_C_PDPg_buffer_Dataset.csv', 'w', newline='') as new_csv_file:
  csv_writer = csv.writer(new_csv_file)

  #write the new lines
  for line in lines:
    csv_writer.writerow(line)

# End and print time
print('Time elapsed for running module "N_T_OB": {:.3f} sec.'.format(time.time() - t_start))