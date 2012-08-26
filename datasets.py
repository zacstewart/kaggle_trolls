import csv
import numpy as np

def readFile(filename, delimiter=','):
  '''Loads a file into a matrix'''
  f = open(filename, 'rb')
  reader = csv.reader(f)
  rows = [row for row in reader]
  header = rows[0]
  matrix = np.array(rows[1:])
  return (header, matrix)
