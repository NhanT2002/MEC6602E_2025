from scipy import special
from math import *
from scipy.interpolate import interp1d

def readAeroData(filename):

    aoa = []
    cl = []
    cd = []
    test = []
    try:
        with open(filename, 'r') as f:
            # 1. Skip the first line (the header)
            next(f)

            # 2. Iterate over the remaining lines
            for line in f:
                # Strip leading/trailing whitespace and split the line by space or tab
                parts = line.strip().split()

                # Ensure the line is not empty and has exactly 3 columns of data
                if len(parts) == 4:
                    try:
                        # Convert parts to float and append to the respective lists
                        aoa_val = float(parts[0])
                        cl_val = float(parts[1])
                        cd_val = float(parts[2])
                        test_val = float(parts[3])
                        aoa.append(aoa_val)
                        cl.append(cl_val)
                        cd.append(cd_val)
                        test.append(test_val)
                    except ValueError:
                        print(f"Skipping line due to non-numeric data: {line.strip()}")
                elif line.strip(): # Check if line is not just empty whitespace
                     print(f"Skipping malformed line (expected 3 columns): {line.strip()}")


    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return [], [], [], []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], [], [], []

    return aoa, cl, cd, test

def ClPolar(alpha):

    #print(alpha)
    aoa, cl, cd, test = readAeroData("Polar.dat")

    #print(cl)

    ClSpline = interp1d(aoa, cl, fill_value = "extrapolate", kind = 'quadratic')

    #print(alpha)
    #print(ClSpline(alpha))
    return ClSpline(alpha)

def CdPolar(alpha):

    #print(alpha)
    aoa, cl, cd, buffet = readAeroData("Polar.dat")

    CdSpline = interp1d(aoa, cd, fill_value = "extrapolate", kind = 'quadratic')


    return CdSpline(alpha)

# Use this to pass the information about an invalid condition (stall or buffet)
def getTestCriterion(alpha):
    aoa, cl, cd, test = readAeroData("Polar.dat")

    testSpline = interp1d(aoa, test, fill_value = "extrapolate", kind = 'linear')

    return testSpline(alpha)>0.0
