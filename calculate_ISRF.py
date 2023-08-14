# Importing modules
from astropy.io import fits
from astropy.table import Table
import concurrent.futures
import numpy as np
import pandas as pd
import pprint
from scipy import integrate
import time



def main():

    fits_file_name = 'FB15N1024_gal5_z0_i0_total.fits'
    Jdat_file_name = 'FB15N1024_gal5_z0_grid_radiationField_J.dat'
    # dummy_Jdat_file_name = 'FB15N1024_gal5_z0_properties_spatialCell_cellprops.dat'

    # Reading J. file
    header, content = read_J_dat_file(file_name = Jdat_file_name, loud = 1)

    # Reading fits file to get wavelength, frequency and unit. I am assuming fits file and _J.dat file have the same wavelength bins
    wavelengths, frequencies, unit = read_fits_file(file_name = fits_file_name)

    # Finding the indices corresponding to the wavelengths where the flux will be integrated.
    # For UV the wavelengths is as follows
    start_wavelength = eV_2_micron(13.6)  # As energy increases the wavelength reduces
    end_wavelength = eV_2_micron(6)       # As energy decreases the wavelength increases
    indices = np.where((wavelengths > start_wavelength) & (wavelengths < end_wavelength))[0]
    print(f"The indices are found for wavelengths between {round(start_wavelength,4)} micron - {round(end_wavelength,4)} micron")

    ####### This is where the parallelization starts

    # Defining the name of the output file
    csv_output_file = 'integrated_filtered_data_between_' + 'λ'+ str(min(wavelengths[indices])) + '_λ' + str(max(wavelengths[indices])) + '.csv'


    # Dividing the content into different chunks
    number_of_lines_per_chunk = int(1e5)
    #     number_of_lines_per_chunk = int(1e3) # TODO: Delete 
    content_chunks = create_content_chunks(content = content, number_of_lines_per_chunk = number_of_lines_per_chunk)

    # Initialize end line and i
    end_line = 0      # I have initialize the end_line because I am using the while loop below.
    i = 0             # There is a multiplication with i is going on the while loop below. With each look it is incremented with one to determine the next end line
    cell_number = []  # This is the cell identification number array
    integrated_value = [] # This is the integrated values for the wavelengths of interest

    # Define how many parallel processes you want to do in the same time
    number_of_processes_in_each_cycle = 30    


    # Initialize the end_chunk_index for control if below
    end_chunk_index = 0

    while (end_line < len(content)):    # Continue the loop as long as there are there are lines unprocessed.
    # while (i < 1): # TODO: Delete. This is done for debugging purposses.

        print(f"\n\n New Cycle for lines between {i * number_of_processes_in_each_cycle * number_of_lines_per_chunk} ---\
        {min((i+1) * number_of_processes_in_each_cycle * number_of_lines_per_chunk, len(content))}")

        # Updating the end line to see if the while loop needs to be stopped.
        end_line = (i+1) * number_of_processes_in_each_cycle * number_of_lines_per_chunk

        # In every loop check if the number_of_processes_in_each_cycle will exceed the size of the content_chunks. 
        # If the number of number_of_processes_in_each_cycle exceed the remaining content_chunks, then I can get an error. 
        # This limitation is required to prevent the case where number_of_processes_in_each_cycle exceeds the remaning content_chunks. 
        j_range_limiter = number_of_processes_in_each_cycle + 1 # Just to be sure it will be more than the number_of_processes_in_each_cycle initially. See inside the concurrent_future


        # The loop will enter the case below, if it is the last loop.
        if (end_chunk_index + number_of_processes_in_each_cycle >= len(content_chunks)):

            # The limitation below required to prevent the case where number_of_processes_in_each_cycle exceeds the remaning content_chunks. 
            j_range_limiter = len(content_chunks) - end_chunk_index   # By this way at then end content_chunks[start_index:len(content_chunks)] will be submited as a final job submission.
            print('Last loop!') # Inform the user that this is the last loop

        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for j in range(min(number_of_processes_in_each_cycle, j_range_limiter)):

                start_chunk_index = i * number_of_processes_in_each_cycle + j

                end_chunk_index = min(start_chunk_index+1, len(content_chunks))

                # Creating a job submission array. Job will be submitted below, but the results are not available directly. See below to see how I got the results. 
                future = executor.submit(integrate_flux, content_chunks[start_chunk_index:end_chunk_index], indices, wavelengths)
                # future is the array that stores the information about the jobs that are going to be submitted just below. 
                futures.append(future)


            # Get results
            for future in futures:
                cell_number.append(future.result()[0])
                integrated_value.append(future.result()[1])

        print(f'loop count: {i}')

        # Increment i by one to start the next loop.
        i += 1


    print("All of the integrals are calculated!!")

    data = []
    for i in range (len(cell_number)):
        for j in range (len(cell_number[i])):
            data.append({'index': int(cell_number[i][j]),
                         'integrated_value': integrated_value[i][j]})


    print(f"last row of the data: {data[-1]}")

    return 0

###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################
## Functions are defined in here

# _J.dat file reader
def read_J_dat_file(file_name:str, loud:bool = 0) -> (list, list):
    
    
    '''This function is written to read the J_dat_file.
    
    Parameters: 
    
    file_name: str 
    File name and it's path
    
    loud: int
    If loud == 1, then the function will print out the information
    about the processes in the code
    
    ---------------------------------------------------------------
    
    Returns: 
    
    all_lines: list
    All lines belongs to the _J.dat file.
    
    header: list 
    Header file belonging to file
    
    content: list 
    Lines that do not start with '#'
    
    '''
    
    print("I am in the function read_J_dat_file")
    if loud:
        print("Reading this file will take around a minute")
    
    # Defining the lists
    header = []
        
    # Reading the file
    with open(file_name, 'r') as file:

        if loud:
            print('Starting reading file')
        
        all_lines = file.readlines() # TODO: Delete
        
        for line in all_lines:
                                        
            # Read header file
            # Header lines start with '#'
            if line[0]=='#':
                header.append(line)
            # It is not a header file but the content of the file if it is not starting with '#'                
            else:
                break

    if loud:
        print('File read!')

    
    content = all_lines.copy()
    del content[:len(header)]
        
    return header, content

##############################################################################################################################

# Fits file reader
def read_fits_file(file_name:str):
    
    '''This function is written to read the file.
    
    Inputs: 
    
    file_name: str
    This is the path and fits file name
    
    Return: 
    wavelengths: np.ndarray
    wavelength array
    
    frequencies: np.ndarray
    frequency array
    
    unit: str
    unit of the outputted fluxes
    '''
    print("I am in the function read_fits_file")    
    
    with fits.open(file_name) as hdul:
#         grid = hdul[0].data
        wavelengths_fits_data = hdul[1].data

        wavelengths = np.zeros(len(wavelengths_fits_data))
        wavelengths = [wavelengths_fits_data[i][0] for i in range (0, len(wavelengths_fits_data))]
        wavelengths = np.array(wavelengths)
        frequencies = 3e8/(wavelengths*1e-6)

        header = hdul[0].header
        unit = hdul[0].header['BUNIT']
    
    return wavelengths, frequencies, unit

##############################################################################################################################

def eV_2_micron(energy):
    
    '''Conversion from electron volts to the wavelength
    
    #######
    
    Parameters: 
    
    energy: in eV
    
    Returns: 
    
    wavelength: in micron
    
    '''
    
    h = 4.1357e-15 # eV s
    c = 3e8 # m/s
    
    wavelength = h * c / energy * 1e6 # in microns
    
    return wavelength 

###################################################################################################################################################################################################
###################################################################################################################################################################################################
###################################################################################################################################################################################################

# Parallelized filter data function
def filter_data(line:str, indices:list, loud:int = 0) -> list:
    
    '''This function filters out the data such that only the first column and the flux columns that corresponds to the 
    specific wavelengths specified by the indices list is taken from the original content list.
    
    ## 
    Parameters: 
    
    row_number: int 
        This is the number of the line that parsing needs to be happened. It is being read and parsed and converted to
        float. 
    
    indices: list 
        This is the column numbers of the indices in the wavelenghts file. Note that the column numbers are not found by 
        reading the same file that is being used to create content list. This is a different file and one needs to come
        up with a solution that reads the headers of the J_dat.file and extracts the indices from there. But for the purposes
        of this code, J_dat file and fits file have the same wavelength bins. 
        The only difference needs to be done is the fact that the first column (O'th column) of the content list belongs to the 
        identification number of the cells, therefore while filtering the data one needs to take index+1 columns where index is 
        in indices.
    '''
    if loud:
        print("I am in the function filter_data\n")
    
    data = []

    # Convert list to an array such that every element will be float
    flux_data = np.array(line.split(" "), dtype='float')
        
    # Now filter the impoted data. First column is the cell identification number.
    data.append(flux_data[0])
        
    # Append the flux values corresponding to the wavelengths.
    for i in indices:
        data.append(flux_data[i+1])
                
    return data

#############################################################################################################################
# After filtering the data integrate the data 
def integrate_filtered_data(data:list, wavelengths:list, indices:list, loud:int = 0) -> float:
    
    '''This function is created to integrate the filtered data
    
    #####
    
    Parameters:
    
    data: list 
        This is the filtered J. file
        
    wavelengths: list
        This is the wavelengths that I am going to use to integrate the data 
    
    #####
    
    Return:
    
    integrated_value: float
    
    
    '''
    if loud:
        print("I am in the function integrate_data\n")
    
    # Only get the flux values corresponding to that wavelengths 
    # Oth column belongs to the cell identification number
    flux_values = data[1:len(wavelengths)+1]
    
    # Only use the wavelengths considered
    wavelengths_filtered = wavelengths[indices]
    
    # Integration: https://en.wikipedia.org/wiki/Simpson%27s_rule
    integrated_value = integrate.simpson(y=flux_values, x=wavelengths_filtered)
    
    return integrated_value
    
#############################################################################################################################

def integrate_flux(parsed_content:list, indices:list, wavelengths:list) -> (int, float):
      
    '''This is the function that integrates the data inputted as the parsed_content. First every line is grabbed from the 
    parsed content. Then this line is inputted to the filter data function where data is fitered according to the 
    wavelength of the flux. If the wavelength of the photons within the wavelengths I am interested then these Lv values are 
    integrated using the simpson integration method. This is done until every line inside the parsed_content is finished.
    
    Arguments: 
    
    parsed_content: list 
        This is the inputted set of lines that integration is going to happen. 
        
    indices: list
        These are the indices of the flux values that needs to be integrated. This is needed to filter the data.
        
    wavlengths: micrometer
        These are the wavelengths. It is used to integrate the data. 
        
    number_of_lines_per_chunk: TODO
        Probably I don't need that. 
    
    '''
    # loud == 0 => No print o.w. print name of the functions
    loud = 0 

    integrated_value_list = []
    data_col_0_list = []
    
    for i in range (len(parsed_content)):        
        for j in range (len(parsed_content[0])):
        
            line = parsed_content[i][j]

            # data is the filtered version of the J. file by only taking the wavelenghts of interest
            data = filter_data(line, indices = indices)

            # Data is integrated by scipy.integrate method using simpson integral
            integrated_value = integrate_filtered_data(data = data, wavelengths = wavelengths, indices = indices)

            data_col_0_list.append(data[0])
            integrated_value_list.append(integrated_value)

    return data_col_0_list, integrated_value_list

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

# This function is tested
def create_content_chunks(content:int, number_of_lines_per_chunk:int) -> list: 
    
    '''Creating content chunks to feed them to parallel processor unit
    
    ######
    
    Parameters:
    
    content: list
    This is the cell indentification number and the flux content of the runned skirt simulatation. J. file
    
    number_of_lines_per_chunk: int 
    This is the number specified by the user to determine the number of lines per chunk
    
    
    Returns:
    
    content_chunks: list
    Parsed content. Every element of this list contains ~number_of_lines_per_chunk lines. 
    
    '''
    
    # Initialize empty list to store the contents
    content_chunks = []

    # Initialize end line 
    end_line = 0
    k = 0
    while (end_line < len(content)):
        start_line = int(number_of_lines_per_chunk * k)
        end_line = min(int(number_of_lines_per_chunk * (k+1)), len(content)) # Select the minimum so that it will not exceed the size of the array

        content_chunks.append(content[start_line:end_line])        

        # Increment k by one 
        k += 1


    return content_chunks


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


    def check_data(content, wavelengths, indices):
        # To check if the value is correct ----> Checked. It is correct.  
        trial_line = content[0]
        flux_data = np.array(trial_line.split(" "), dtype='float')

        flux_data = flux_data[indices+1]

        print(len(flux_data))
        print(len(wavelengths[indices]))
        resultt = integrate.simpson(y=flux_data, x=wavelengths[indices])
        print(indices+1)
        print(resultt)

if __name__ == "__main__":

    main()
