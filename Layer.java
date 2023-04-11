import java.util.ArrayList;

public class Layer {
	public ArrayList<Neuron> neurons = new ArrayList<Neuron>();	//List of neurons
	private int H; 												//Number of neurons-outputs
	private int D;												//Number of inputs
	private String func_name;									//Activation function name
	private float l_r;											//Learning rate

	public Layer(int d,int h, String func_name, float l_r){
		this.func_name = func_name;
		this.H = h;
		this.D = d;
		this.l_r = l_r;
		init_Neurons();
	}

	// This function is used on the input layer
	// As it has no outputs to be calculated,
	// set the outputs to the values of the inputs 
	public void init_Input_Layer_y(float[] x_inputs){
		for (int i=0; i< neurons.size();i++){
			neurons.get(i).y = x_inputs[i];
		}
	}

	// Create neurons equal to the number H
	// that indicates the neurons on the layer
	public void init_Neurons(){
		for (int h = 0;h < H;h++) {
			neurons.add(new Neuron(D,func_name,l_r));
		}
	}

	// Apply forward pass using the outputs of the neurons
	// On the previous layer and the weights on the current one
	public void forward_pass(Layer prev_Layer){
		for (Neuron neuron : neurons){
			neuron.x[0] = 1.0f;
			for (int i = 0;i < D;i++){
				neuron.x[i+1] = prev_Layer.neurons.get(i).y;
			}
			neuron.calculate_u();
		}		
	}

	// Apply back-propagation from the last hidden layer
	// to the input layer using the dela
	public void back_prop(Layer next_layer){
		for (int h = 0;h < H;h++) {
			for(Neuron n:next_layer.neurons){
				neurons.get(h).delta+=n.weights[h+1]*n.delta;
			}
			neurons.get(h).delta*=neurons.get(h).compute_delta();
		}		
	}

	// Apply back-propagation for the output layer
	public void back_prop_final(float[] category){
		for (int h = 0;h < H;h++) {
			neurons.get(h).delta=neurons.get(h).compute_delta_final(category[h]);
		}	
	}

	// Calculate dE on each neuron of the layer
	public void calculate_de(){
		for (Neuron n : neurons){
			n.calculate_grad();
		}
	}

	// Update weigths on the current layer
	public void update_weigths_on_layer(){
		for (Neuron n : neurons){
			n.update_weigths();
		}
	}

	// Reset dE and delta on each neuron of the layer
	public void reset_grad_weights(){
		for (Neuron n : neurons){
			n.clear_grads();
		}
	}
}