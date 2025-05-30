import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

public class Main {

	//Hyperparameters
	int training_size = 60000;
	int test_size = 10000;
	int feature_size = 784;//Same as # of inputs
	int output_size = 10;
	int epoch_limit = 51;//Training stops automatically once this number is hit
	int hidden_units = 100;//experiments will be with 20,50,100
	double learning_rate = 0.1;
	double momentum = 0.9;
	double bias_input = 1.0;//Although the bias weights are random, the input is always 1
	
	//Layer of neuron weights (input and hidden)
	double[][] input_weights = new double[hidden_units][feature_size];
	double[][] hidden_weights = new double[output_size][hidden_units];
	double[] bias_hidden_weights = new double[hidden_units];
	double[] bias_output_weights = new double[output_size];
	
	//Training Data
	double[][] training_data = new double[training_size][feature_size];
	double[] training_labels = new double[training_size];
	int[][] training_target_label = new int[training_size][output_size];
	
	//Test Data 
	double[][] test_data = new double[test_size][feature_size];
	double[] test_labels = new double[test_size];
	int[][] test_target_label = new int[test_size][output_size];
	
	//Momentum arrays
	double[][] hidden_weight_delta = new double[output_size][hidden_units];
	double[][] input_weight_delta = new double[hidden_units][feature_size];
	double[] bias_hidden_weight_delta = new double[hidden_units];
	double[] bias_output_weight_delta = new double[output_size];
	
	
	Random rand = new Random();
	
	public static void main(String[] args) throws Exception{
		Main m = new Main();
	}
	
	public Main() throws Exception{
		//Reads training data
		readData();
		
		//ONLY USE THE data_reducer FUNCTION FOR EXPERIMENT 3
		//Reduction of data for experiment 3
		//data_reducer(30000);//Either 15000 for quarter or 30000 for half
		
		
		settingInitialWeights();
		//Training till it reaches the epoch limit
		for(int i = 0;i < epoch_limit;i++) {
			accuracy(training_data,training_labels,"Training",i);
			accuracy(test_data,test_labels,"Test",i);
			train();	
		}
	}
	
	private void train() {
		//Shuffle the data
		shuffleTrainingData();
		//Use squared loss 
		for(int i = 0;i < training_size;i++) {
			double sum = 0.0;
			//Forward Pass of hidden layer
				//Getting the sum of the weights * inputs and then the sigmoid
				//Storing the value of each hidden neuron in hidden_output
			double[] hidden_output = new double[hidden_units];
			for(int a = 0;a < hidden_units;a++) {
				sum = 0.0;
				for(int z = 0;z < feature_size;z++) {
					sum += training_data[i][z] * input_weights[a][z];
				}
				//Adding Bias
				sum += bias_input * bias_hidden_weights[a];
				//Getting the sigmoid of the sum and then adding to 
				hidden_output[a] = sigmoid(sum);
			}
			
			//Forward Pass of output layer
				//Same as the forward pass of hidden layer but with the output layer
			double[] final_output = new double[output_size];
			for(int a = 0;a < output_size;a++) {
				sum = 0.0;
				for(int z = 0; z < hidden_units;z++) {
					sum += hidden_output[z] * hidden_weights[a][z];
				}
				sum += bias_input * bias_output_weights[a];
				final_output[a] = sigmoid(sum);
			}
			
			//Backward Pass
				//Compute Error at output layer using formula from slides
			double[] output_error = new double[output_size];
			double error;
			for(int a = 0;a < output_size;a++) {
				error = training_target_label[i][a] - final_output[a];
				output_error[a] = error * final_output[a] * (1 - final_output[a]); 
			}
				//Compute Error at hidden layer using formula from slides
			double[] hidden_error = new double[hidden_units];
			for(int a = 0;a < hidden_units;a++) {
				error = 0;
				for(int z = 0;z < output_size;z++) {
					error += output_error[z] * hidden_weights[z][a];
				}
				hidden_error[a] = error * hidden_output[a] * (1 - hidden_output[a]);
			}
				//Update the output layer weights
			double delta,bias_delta;
			for(int a = 0;a < output_size;a++) {
				for(int z = 0;z < hidden_units;z++) {
					delta = learning_rate * output_error[a] * hidden_output[z] + momentum * hidden_weight_delta[a][z];
					//updating the weight with the delta
					hidden_weights[a][z] += delta;
					//storing the magnitutde of delta in the momentum array
					hidden_weight_delta[a][z] = delta;
				}
				//Same as above but for the bias weight
				bias_delta = learning_rate * output_error[a] * bias_input + momentum * bias_output_weight_delta[a];
				bias_output_weights[a] += bias_delta;
				bias_output_weight_delta[a] = bias_delta;
			}
				//Update the hidden layer weights
				//Same formula as the output layer but with proper arrays
			for(int a = 0;a < hidden_units;a++) {
				for(int z = 0;z < feature_size;z++) {
					delta = learning_rate * hidden_error[a] * training_data[i][z] + momentum * input_weight_delta[a][z];
					//updating the weight with the delta
					input_weights[a][z] += delta;
					//storing the magnitutde of delta in the momentum array
					input_weight_delta[a][z] = delta;
				}
				//Same as above but for the bias weight
				bias_delta = learning_rate * hidden_error[a] * bias_input + momentum * bias_hidden_weight_delta[a];
				bias_hidden_weights[a] += bias_delta;
				bias_hidden_weight_delta[a] = bias_delta;
			}
		}
	
	}

	
	/*
	 * Setting intial weights for input neurons and hidden layer neurons
	 * to between -.05 and .05
	 * Also setting the bias weights here
	 */
	private void settingInitialWeights() {
		//input neurons being set to random weights
		for(int i = 0;i < hidden_units;i++) {
			bias_hidden_weights[i] = rand.nextDouble() * (.1) -.05;
			for(int a = 0;a < feature_size;a++) {
				input_weights[i][a] = rand.nextDouble() * (.1) -.05;
			}
		}
		//Hidden neurons being set to random weights
		for(int i = 0;i < output_size;i++) {
			bias_output_weights[i] = rand.nextDouble() * (.1) -.05;
			for(int a = 0;a < hidden_units;a++) {
				hidden_weights[i][a] = rand.nextDouble() * (.1) -.05;
			}
		}
		
	}
	
	//Sigmoid function
	private double sigmoid(double num) {
		return((1) / (1 + Math.exp(-num)));
	}
	
	/*
	 * Data reducer function, ensures that the data is balanced
	 * Takes in the value of how many training data examples should be in the 
	 * new matrices
	 */
	private void data_reducer(int target_training_size) {
		//Have to first find how many of each label there are before I can balance
		//Since I don't know how many of each label there are I will have to use an arraylist (dynamic)
		ArrayList<ArrayList<Integer>> label_count = new ArrayList<>();
		for(int i = 0;i < output_size;i++) {
			label_count.add(new ArrayList<>());
		
		}
		for(int i = 0;i < training_size;i++) {
			//Adding every index of each label to the corresponding arraylist
			label_count.get((int)training_labels[i]).add(i);
		}
		
		double[][] new_training_data = new double[target_training_size][feature_size];
		double[] new_training_labels = new double[target_training_size];
		int[][] new_training_target_labels = new int[target_training_size][output_size];
		int samples_per_class = target_training_size/10;
		//Going through the original training set, 
		//shuffling before adding back to the new matrices
		int index = 0;
		for(int i = 0;i < output_size;i++) {
			ArrayList<Integer> temp_arr = label_count.get(i);
			java.util.Collections.shuffle(temp_arr,rand);
			for(int a = 0;a < samples_per_class;a++) {
				int temp_index = temp_arr.get(a);
				new_training_labels[index] = training_labels[temp_index];
				for(int z = 0;z < feature_size;z++) {
					new_training_data[index][z] = training_data[temp_index][z];
				}
				for(int z = 0;z < output_size;z++) {
					new_training_target_labels[index][z] = training_target_label[temp_index][z];
				}
				index++;
			}
		}
		//Now setting the training data to the new matrices
		training_data = new_training_data;
		training_labels = new_training_labels;
		training_target_label = new_training_target_labels;
		//Also change hyperparameter of training size to new number
		training_size = target_training_size;
		
	}
	
	/*
	 * Calculating Accuracy function
	 * The double[][] matrix is either test set or training set
	 * A lot of this is the same as the train() function but it needs to 
	 * be separate since I want to calculate on training or test data 
	 */
	private void accuracy(double[][] matrix, double[] labels,String str,int e) {
		int correct = 0;
		int[][] confusion_matrix = new int[output_size][output_size];
		for(int i = 0;i < matrix.length;i++) {
			double sum;
			//Forward pass through the hidden layer 
			double[] hidden_output = new double[hidden_units];
			for(int a = 0;a < hidden_units;a++) {
				sum = 0.0;
				for(int z = 0;z < feature_size;z++) {
					sum += matrix[i][z] * input_weights[a][z];
				}
				//Adding Bias
				sum += bias_input * bias_hidden_weights[a];
				//Getting the sigmoid of the sum and then adding to 
				hidden_output[a] = sigmoid(sum);
			}
			//Foward pass through the inner layer
			double[] final_output = new double[output_size];
			for(int a = 0;a < output_size;a++) {
				sum = 0.0;
				for(int z = 0; z < hidden_units;z++) {
					sum += hidden_output[z] * hidden_weights[a][z];
				}
				sum += bias_input * bias_output_weights[a];
				final_output[a] = sigmoid(sum);
			}
			
			//Predictions
			//The index with the highest output value is chosen as the prediction
			//Once the predicted index is found it is compared to the correct label
			int pred = 0;
			for(int a = 0;a < output_size;a++) {
				if(final_output[a] > final_output[pred]) {
					pred = a;
				}
			}
			if(pred == ((int)labels[i])) {
				correct++;
			}
			
			//Confusion Matrix code
			confusion_matrix[(int)labels[i]][pred] += 1;
			
		}
		//Print out the accuracy
		double accuracy = ((double)100 * (double)correct) / (double)matrix.length;
		System.out.println("Epoch " + e + " , " + str + " Accuracy: " + accuracy);
		//Printout the confusion matrix but only for the test set and if the last epoch
		if((e == (epoch_limit-1)) && str.equals("Test")) {
			for(int i = 0;i < output_size;i++) {
				for(int a = 0;a < output_size;a++) {
					System.out.print(confusion_matrix[i][a] + " ");
				}
				System.out.println("");
			}
		}
	}
	
	//Function that shuffles the training data 
	private void shuffleTrainingData() {
		for (int i = training_data.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            double[] temp = training_data[i];
            training_data[i] = training_data[j];
            training_data[j] = temp;

            int temp_label = (int) training_labels[i];
            training_labels[i] = training_labels[j];
            training_labels[j] = temp_label;
            
            int[] temp_one_hot_encode = training_target_label[i];
            training_target_label[i] = training_target_label[j];
            training_target_label[j] = temp_one_hot_encode;
        }
	}
	
	/*
	 * This function reads the training and test csv files
	 *  and puts it into the corresponding matrices
	 */
	private void readData() throws Exception{
		//Reading Training Data 
		BufferedReader br = new BufferedReader(new FileReader("C:\\Users\\sharp\\OneDrive\\Desktop\\MNIST_DATA\\MNIST_TRAIN.csv"));
        String line;
        int index = 0;
        br.readLine();
        while ((line = br.readLine()) != null) {
            String[] str_arr = line.split(",");
            for (int i = 0; i < feature_size; i++) {
            	/*
            	 * Scaling data to between 0 and 1
            	 */
                double scaled = Integer.parseInt(str_arr[i + 1]) / 255.0;
                training_data[index][i] = scaled;
            }
            //One hot encoding
            training_labels[index] = Integer.parseInt(str_arr[0]);
            for(int i = 0;i < output_size;i++) {
            	if(i == Integer.parseInt(str_arr[0])) {
            		training_target_label[index][i] = 1;
            	}
            	else {
            			training_target_label[index][i] = 0;
            	}
            }
            
            
            index++;
        }
        br.close();
        //Reading Testing Data
        br = new BufferedReader(new FileReader("C:\\Users\\sharp\\OneDrive\\Desktop\\MNIST_DATA\\MNIST_TEST.csv"));
        index = 0;
        br.readLine();
        while ((line = br.readLine()) != null) {
            String[] str_arr = line.split(",");

            for (int i = 0; i < feature_size; i++) {
            	/*
            	 * Scaling data to between 0 and 1
            	 */
                double scaled = Integer.parseInt(str_arr[i + 1]) / 255.0;
                test_data[index][i] = scaled;
            }
            //One hot encoding
            test_labels[index] = Integer.parseInt(str_arr[0]);
            for(int i = 0;i < output_size;i++) {
            	if(i == Integer.parseInt(str_arr[0])) {
            		test_target_label[index][i] = 1;
            	}
            	else {
            			test_target_label[index][i] = 0;
            	}
            }
            
            index++;
        }
        br.close();
 
        
        
	}
	
	
	
}
