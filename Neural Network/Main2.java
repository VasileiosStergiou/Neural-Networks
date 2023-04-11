import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter; 
import java.io.IOException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;

public class Main2{
	public static Layer Input_Layer; //The input layer of the MLP
	public static Layer H1; //The first hidden layer of the MLP
	public static Layer H2; //The second hidden layer of the MLP
	public static Layer H3; //The third hidden layer of the MLP
	public static Layer Output_Layer; //The output layer of the MLP

	public static void main(String[] args) {
		ArrayList<float[]> training_set = parse_Data("training_set.txt"); // load the data of the traing set
		ArrayList<Float> errors = new ArrayList<Float>(); // the error per epoch
		//Relu ,b=1, thr=0.001; l_r=0.0005 ,l=30
		int d = 2; //the number of inputs on MLP
		int h1 = 15; // the number of neurons on first hidden layer
		int h2 = 15; // the number of neurons on second hidden layer
		int h3 = 7; // the number of neurons on third hidden layer
		int k = 4; // the number of outputs on the output layer
		int batches_number = 40; //the number of samples that will be propagated through the network
		int epoch = 700; //the minimum number of epoches that the algorithm has to run
		float threshold = 0.01f; //the minimum difference of the errors between two epoches
		float l_r = 0.005f; //the learning rate 
		String activation_function = "Tanh"; //the activation function of the hidden layers

		float error = 0.0f; //the error per sample
		float total_error = 0.0f; // the total error of each epoch
		float previous_error = 0.0f; // the error of the previous epoch

		float[] sample = new float[d]; // the x,y values of the current sample
		float[] category = new float[k]; // a vector of size k that indicates the category that the sample belongs


		//Initialiazation of layers
		Input_Layer = new Layer(d,d,activation_function,l_r); 
		H1 = new Layer(d,h1,activation_function,l_r);
		H2 = new Layer(h1,h2,activation_function,l_r);
		H3 = new Layer(h2,h3,activation_function,l_r);
		Output_Layer = new Layer(h3,k,"Sigmoid",l_r);

		//a counter that indicates the epoch we are currently on
		int epoch_counter = 0;

		while (true){
			//Set the dE and delta on each neuron of the layers to 0
			reset_grad_weigths_on_network();

			//Depending on the number of batches (b)
			//Apply forward pass and back-propagation b times
			for (int ex = 0; ex < training_set.size();){
				for (int b = 0; b < batches_number;b++, ex++){
					sample[0] = training_set.get(ex)[0]; // the x value of the current sample
					sample[1] = training_set.get(ex)[1]; // the y value of the current sample
					category = cat2vec(k,(int) training_set.get(ex)[2]); // tranform the category of the sample to a k-size vector

					//In the input layer, its outputs are the inputs that are given
					//So set the outputs of the neurons (y) to have the x,y values of the sample
					Input_Layer.init_Input_Layer_y(sample); 

					//Apply forward pass on each layer, exluding the first one
					forward_passess_on_network();

					//Apply back-propagation o each layer
					//The category of the sample is passed to the output layer
					//to begin the back-propagation
					back_prop_on_network(category);

					//sum the error of the sample
					error+=calculate_error(category);
				}
				//Update the weights on network
				update_weigths_on_network();

				//Set the dE and delta on each neuron of the layers to 0
				reset_grad_weigths_on_network();
			}
			// Calculate the total error of the epoch
			total_error = 0.5f * error;

			// Store the error of the epoches
			// to plot them later
			errors.add(total_error);

			System.out.println("Epoch: "+epoch_counter+" Error: "+total_error);
			
			// Check if the algorithm ends if the termination condition is true
			if (epoch_counter >= epoch && previous_error - total_error <= threshold){
				break;
			}

			// If the algorithm loops again, keep the error to check the condition
			// On the next loop
			previous_error = total_error;

			// And reset the error so that it can be calculated fot the next epoch
			error = 0.0f;
			epoch_counter++;
		}
		// At the end of the algorithm, calculate and print
		// the generalization rate
		generalization("test_set.txt",d,k);

		// Write the estimated outputs and the errors to a file
		points2file("test_set.txt",d,k);
		errors2file(errors);
	}
	// Forward pass on each layer, excluding the first one
	static void forward_passess_on_network(){
		H1.forward_pass(Input_Layer);
		H2.forward_pass(H1);
		H3.forward_pass(H2);
		Output_Layer.forward_pass(H3);
	}

	// back-propagation from the output layer to the input
	static void back_prop_on_network(float [] category){

		// calculate delta on each layer
		Output_Layer.back_prop_final(category);
		H3.back_prop(Output_Layer);
		H2.back_prop(H3);
		H1.back_prop(H2);

		// calculate partial derivative (dE) on MLP
		H1.calculate_de();
		H2.calculate_de();
		H3.calculate_de();
		Output_Layer.calculate_de();
	}

	// Update weigths on each layer, exluding the first one
	static void update_weigths_on_network(){
		H1.update_weigths_on_layer();
		H2.update_weigths_on_layer();
		H3.update_weigths_on_layer();
		Output_Layer.update_weigths_on_layer();
	}

	//Set the dE and delta on each neuron of the layers to 0
	static void reset_grad_weigths_on_network(){
		H1.reset_grad_weights();
		H2.reset_grad_weights();
		H3.reset_grad_weights();
		Output_Layer.reset_grad_weights();
	}
	//Compute the error of a sample using its category and the output layer
	static float calculate_error(float [] category){
		float error = 0.0f;
		for (int i = 0; i < Output_Layer.neurons.size();i++){
			error += (float) Math.pow(Output_Layer.neurons.get(i).y - category[i],2);
		}
		return error;
	}

	// On each sample of the test set, apply forward pass
	// and check the output of the neetwork with its category
	// and compute the generalisation rate
	static void generalization(String test_filename, int d, int k){
		ArrayList<float[]> test_set = parse_Data(test_filename);
		float[] test_data = new float[d];
		int correct_guesses = 0;

		for (float[] test_array : test_set) {  
			float[] output = new float[k];  
			test_data[0] = test_array[0];
			test_data[1] = test_array[1];
			Input_Layer.init_Input_Layer_y(test_data);
			H1.forward_pass(Input_Layer);
			H2.forward_pass(H1);
			H3.forward_pass(H2);
			Output_Layer.forward_pass(H3);
			for(int j = 0;j < Output_Layer.neurons.size();j++){
				output[j] = Output_Layer.neurons.get(j).y;
			}

			if (probs2cat(output) == test_array[2]){
				correct_guesses++;
			}
		}
		System.out.println("Correct guesses rate : " + correct_guesses + " out of " +test_set.size() + " examples");
		System.out.println("Correct guesses rate : " + correct_guesses/ (float) test_set.size());
	}

	// load the data from a file
	static ArrayList<float[]> parse_Data(String filename){
		ArrayList<float[]> all_data = new ArrayList<float[]>();
		float[] point;
		try {
			File myObj = new File(filename);
			Scanner myReader = new Scanner(myObj);
			while (myReader.hasNextLine()) {
				point= new float[3];
				String[] data = myReader.nextLine().split(",");
				for(int i = 0;i < 3;i++){
					point[i]=Float.parseFloat(data[i]);
				}
				all_data.add(point);
			}
			myReader.close();
		} catch (FileNotFoundException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}  
		return all_data;
	}

	// write the outputs of the network to a file
	static void points2file(String test_filename, int d, int k) {
		ArrayList<float[]> test_set = parse_Data(test_filename);
		float[] test_data = new float[d];

		try {
			FileWriter myWriter = new FileWriter("outputs.txt");
			for (int i = 0;i < test_set.size();i++) {  
				float[] output = new float[k];  
				test_data[0] = test_set.get(i)[0];
				test_data[1] = test_set.get(i)[1];
				Input_Layer.init_Input_Layer_y(test_data);
				H1.forward_pass(Input_Layer);
				H2.forward_pass(H1);
				H3.forward_pass(H2);
				Output_Layer.forward_pass(H3);
				for(int j = 0;j < Output_Layer.neurons.size();j++){
					output[j] = Output_Layer.neurons.get(j).y;
				}

				myWriter.write(test_set.get(i)[0]+","+test_set.get(i)[1]+","+probs2cat(output)+"\n");   
			}
			myWriter.close();
		} catch (IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
	}

	// write the errors of the network to a file

	static void errors2file(ArrayList<Float> errors) {
		try {
			FileWriter myWriter = new FileWriter("errors.txt");
			for (float er : errors) {
				myWriter.write(er+"\n");   
			}
			myWriter.close();
		} catch (IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
	}

	// find the output with the highest probabilty
	// its position on the array indicates and the category too
	static float probs2cat(float[] outputs){
		float max_val = outputs[0];
		int pos = 0;
		for(int i = 1;i < outputs.length;i++){
			if(outputs[i]>max_val){
				max_val=outputs[i];
				pos=i;
			}
		}
		return pos;
	}

	// tranform the category to a vector
	static float[] cat2vec(int k,int c){
		float[] catvec = new float[k];
		catvec[c] = 1.0f;
		return catvec;
	}
}