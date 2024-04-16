
import matlab.engine
import numpy as np
import os
import re
from scipy.io import loadmat

# Set Project, need to be the same name as project containing TempEstInput.m in actuatorControl\units\acTempEst
matlab_project = '1_682_25_FXD_Winter_20180219'

mseloss_path = 'C:/Code/CC_ECC/target/components/internal/actuatorControl/units/acTempEst/' + matlab_project + '/Measurements'


estinput_path = 'C:/Code/CC_ECC/target/components/internal/actuatorControl/units/acTempEst/' + matlab_project + '/TempEstInput.m'
log_path = 'C:/Code/CC_ECC/target/components/internal/actuatorControl/units/acTempEst/test/log.txt'
root = 'C:/Code/CC_ECC/target/components/internal/actuatorControl/units/acTempEst/test'

# TempEstModel can be Gen6, Gen6FXD, DogClutch, TVDC or TCase
matlab_tempEstModel = 'Gen6FXD'
os.chdir(root)

class TempModel:

    def __init__(self):
        # Smallest loss
        self.highscore = 19000
        self.measurementData = self.find_filtered_mat_files(mseloss_path)
        self.eng = self.startmodel()
        # init display
        self.reset()
        self.oldloss = self.read_currentloss()

    def startmodel(self):
        """Start MATLAB engine"""
        eng = matlab.engine.start_matlab()
        # nargout = 0, no output
        if matlab_tempEstModel == 'DogClutch':
            eng.TempEstTestDogClutchInit_python(nargout=0)
        elif matlab_tempEstModel == 'TVDC':
            eng.TempEstTestTVDCInit_python(nargout=0)
        elif matlab_tempEstModel == 'TCase':
            eng.TempEstTestTCaseInit_python(nargout=0)
        else:
            # genvi
            eng.TempEstTestInit_python(nargout=0)

        return eng

    def quitmodel(self):
        """Quit MATLAB engine"""
        self.eng.quit()
    
    def find_filtered_mat_files(self, folder_path):
        # List all .mat files in the folder
        all_mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

        # Initialize a list to store filtered filenames
        filtered_files = []

        # Loop through each file and check if it meets the criteria
        for file_name in all_mat_files:
            if not file_name.startswith(('data_', 'read_', 'mseloss_', 'SimOutput_')):
                # Extract filename without extension
                filtered_files.append(os.path.splitext(file_name)[0])

        return filtered_files

    def read_currentloss(self):
        """Return average loss of all training mat files"""
        total_sum = 0
        total_elements = 0
        # Load mse loss data
        for element in self.measurementData:
            filepath = mseloss_path + '/' + 'mseloss_' + element + '.mat'
            data = loadmat(filepath)
            total_sum += sum(data['loss'][0,0])
            total_elements += len(data['loss'][0,0])

        return float(total_sum / total_elements)

    def readparameters(self):
        """Read TempEstInput.m"""
        with open(estinput_path, 'r') as file:
            estinput = file.read()
        return estinput

    def run_simulation(self, resetfile=False):
        """Write current parameters to TempEstInput.m then run the simulation, calculate loss and store to mat file
        """ 
        if resetfile:
            log = "\nReset Parameters: \n"
        else:
            log = "\nParameters: \n"
        estinput = self.readparameters()
        
        for variable, value in self.paramsdict.items():
            # Define the regular expression pattern to match the parameter value
            pattern = rf'{variable}\s*=\s*(-?\d+\.*\d*)'

            # Find and update the parameter value
            match = re.search(pattern, estinput)
            if match:
                #current_value = float(match.group(1))
                log += str(variable) + "= " + str(value) + ", "
                updated_content = re.sub(pattern, f'{variable} = {value}', estinput)
                estinput = updated_content
            else:
                print(f"Parameter {variable} not found in the script.")

        # Write the updated content back to the file
        with open(estinput_path, 'w') as file:
            file.write(estinput)

        self.eng.runsimulation(matlab_tempEstModel, matlab_project, nargout=0)

        # Load mse loss data
        self.avgmseloss = self.read_currentloss()
        # Set target loss
        self.targetloss = self.avgmseloss - 1

        if self.avgmseloss < self.highscore:
            self.highscore = self.avgmseloss

        if resetfile:
            log += "\nStart new training cycle"
        else:
            log += "\nAverage Loss: " + str(self.avgmseloss) + ", High score: " + str(self.highscore)
 
        # Update log file
        with open(log_path, "a+") as file:
            file.write(log)

    def reset(self):
        self.paramsdict = {'AlfaCplgOilToPmpHd': 0, 'AlfaCplgOilToLmlLo': 0, 'AlfaCplgOilToLmlHi': 0,
                   'AlfaCplgOilToCplg': 0, 'AlfaCplgOilToFnGear': 0, 'AlfaCplgToPmpHd': 0,
                   'AlfaCplgToLml': 0, 'AlfaCplgToFinGear': 0, 'AirToPumpHead': 30,
                   'AirToCoupling': 20, 'AirToFinalGear': 20}

        # Init parameters in TempEstInput.m
        self.run_simulation(resetfile=True)
        self.iteration = 0

    def play_step(self, action):

        # Todo: Run simulation to calculate loss
        self.iteration += 1
        
        # 1. update the params, run the simulation and calculate the loss
        self._move(action)
        print(f"Iteration: {self.iteration}, MSE Loss: {self.avgmseloss}")
        # 2. check if simulation over
        reward = 0
        game_over = False
        if self.is_termination() or self.iteration > 1000:
            game_over = True
            reward = -10
            return reward, game_over, self.avgmseloss

        # 3. update oldloss or just move
        if self.avgmseloss < self.oldloss:
            reward += 10
        if self.avgmseloss > self.oldloss:
            reward -= 10
        # Update oldloss
        self.oldloss = self.avgmseloss

        # 4. if the loss equals highscore 
        if self.avgmseloss == self.highscore:
            reward += 10

        # 6. return simulation over and average loss
        return reward, game_over, self.avgmseloss


    def is_termination(self):
        for variable, value in self.paramsdict.items():
            # hits boundary
            if value > 50:
                return True
        return False

    def _move(self, action):
        #     {'AlfaCplgOilToPmpHd': 0, 'AlfaCplgOilToLmlLo': 0, 'AlfaCplgOilToLmlHi': 0,
        #     'AlfaCplgOilToCplg': 0, 'AlfaCplgOilToFnGear': 0, 'AlfaCplgToPmpHd': 0,
        #     'AlfaCplgToLml': 0, 'AlfaCplgToFinGear': 0, 'AirToPumpHead': 0,
        #     'AirToCoupling': 0, 'AirToFinalGear': 0}

        if np.array_equal(action, [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
            self.paramsdict['AlfaCplgOilToPmpHd'] += 1
        elif np.array_equal(action, [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
            self.paramsdict['AlfaCplgOilToLmlLo'] += 1
        elif np.array_equal(action, [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
            self.paramsdict['AlfaCplgOilToLmlHi'] += 1
        elif np.array_equal(action, [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]):
            self.paramsdict['AlfaCplgOilToCplg'] += 1
        elif np.array_equal(action, [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]):
            self.paramsdict['AlfaCplgOilToFnGear'] += 1
        elif np.array_equal(action, [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]):
            self.paramsdict['AlfaCplgToPmpHd'] += 1
        elif np.array_equal(action, [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]):
            self.paramsdict['AlfaCplgToLml'] += 1
        elif np.array_equal(action, [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]):
            self.paramsdict['AlfaCplgToFinGear'] += 1
        elif np.array_equal(action, [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]):
            self.paramsdict['AirToPumpHead'] += 1
        elif np.array_equal(action, [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]):
            self.paramsdict['AirToCoupling'] += 1
        elif np.array_equal(action, [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]):
            self.paramsdict['AirToFinalGear'] += 1

        elif np.array_equal(action, [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]):
            self.paramsdict['AlfaCplgOilToLmlLo'] += 5
        elif np.array_equal(action, [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]):
            self.paramsdict['AlfaCplgOilToLmlHi'] += 5
        elif np.array_equal(action, [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]):
            self.paramsdict['AlfaCplgOilToCplg'] += 5
        elif np.array_equal(action, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]):
            self.paramsdict['AirToPumpHead'] += 5
        elif np.array_equal(action, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]):
            self.paramsdict['AirToCoupling'] += 5
        elif np.array_equal(action, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]):
            self.paramsdict['AirToFinalGear'] += 5

        # Update parameters in TempEstInput.m and run simulation
        self.run_simulation()

def find_filtered_mat_files(folder_path):
    # List all .mat files in the folder
    all_mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    # Initialize a list to store filtered filenames
    filtered_files = []

    # Loop through each file and check if it meets the criteria
    for file_name in all_mat_files:
        if not file_name.startswith(('data_', 'read_', 'mseloss_', 'SimOutput_')):
            filtered_files.append(os.path.splitext(file_name)[0])  # Extract filename without extension

    return filtered_files
def read_currentloss():
    """Return average loss of all training mat files"""
    total_sum = 0
    total_elements = 0
    measurementData = find_filtered_mat_files(mseloss_path)
    # Load mse loss data
    for element in measurementData:
        filepath = mseloss_path + '/' + 'mseloss_' + element + '.mat'
        data = loadmat(filepath)
        total_sum += sum(data['loss'][0,0])
        total_elements += len(data['loss'][0,0])

    return float(total_sum / total_elements)
if __name__ == '__main__':
    # tempmodel = TempModel()
    # tempmodel.reset()
    loss = read_currentloss()
    print(loss)