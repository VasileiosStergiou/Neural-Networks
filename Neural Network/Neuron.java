import java.util.Random;
import java.lang.Math;

public class Neuron {
	public float[] weights;			//Weights
	public float[] x; 				//Inputs
	public float delta = 0.0f;		//Delta
	public float y = 0.0f; 			//Output after activation function
	private float[] grad_weights; 	//Gradient weights
	private int D; 					//Number of weights including bias
	private float u = 0.0f; 		//Output before activation function
	private String func_name; 		//Activation function name
	private float l_r; 				//Learning rate

	// Class constructor
	public Neuron(int d, String func_name, float l_r){
		this.func_name = func_name;
		this.D = d+1;
		this.weights = new float[this.D];
		this.x = new float[this.D];
		this.l_r = l_r;
		init_weights();	//start weights initialization
	}

	// Initialize weights with random values between -1 and 1
	private void init_weights(){
		Random random_weigth = new Random();

		this.grad_weights = new float [D];

		// w[0] is always 1
		this.weights[0] = 1.0f;

		for (int i =1;i< D;i++){
	 		float w = random_weigth.nextFloat() * 2 - 1 ; //2=(max - min) + min
	  		this.weights[i] = w;
		}
	}

	// Calculate derivative (dE) for every weight
	public void calculate_grad(){
		this.grad_weights[0] += delta;
		for (int i =1; i<grad_weights.length;i++){
			this.grad_weights[i] += delta * x[i];		
		}
	}

	// Calculate the output of neuron (u) and pass it thought an activation function (y)
	public void calculate_u(){
		this.u=weights[0]*x[0];
		for (int i=1;i<D;i++) {
			this.u+=weights[i]*x[i];
		}
		this.y = activation_func(func_name,u);
	}

	// Select activation function by name
	private float activation_func(String func_name,float x){
		switch (func_name){
			case "Sigmoid":
				return sigmoid_f(1.0f,x);
			case "Tanh":
				return tanh_f(1.0f,x);
			case "Relu":
				return relu_f(x);
			default:
				return -1.0f;
		}
	}

	// Select derivative of activation function by name
	private float der_activation_func(String func_name,float x){
		switch (func_name){
			case "Sigmoid":
				return sigmoid_df(1.0f,x);
			case "Tanh":
				return tanh_df(1.0f,x);
			case "Relu":
				return relu_df(x);
			default:
				return -1.0f;
		}
	}

	// Sigmoid activation function
	private float sigmoid_f(float a,float x){
		return (float) (1/(1+Math.exp(-a*x)));
	}

	// Tanh activation function
	private float tanh_f(float a,float x){
		return (float) ((Math.exp(a*x)-Math.exp(-a*x))/(Math.exp(a*x)+Math.exp(-a*x)));
		//return (float) Math.tanh(x);
	}

	// Relu activation function
	private float relu_f(float x){
		return (float) (Math.max(x,0));
	}

	// Sigmoid derivative activation function
	private float sigmoid_df(float a,float x){
		return sigmoid_f(a,x)*(a-sigmoid_f(a,x));
	}

	// Tanh derivative activation function
	private float tanh_df(float a,float x){
		return a-tanh_f(a,x)*tanh_f(a,x);
	}

	// Relu derivative activation function
	private float relu_df(float x){
		if(x>0){
			return 1.0f;
		}
		return 0.0f;
	}	

	// Update the weights given derivative of weights (dE)
	public void update_weigths(){
		for (int i =0; i< weights.length;i++){
			this.weights[i] -= l_r * grad_weights[i];
		}
	}

	// Get the first part for delta calculation (derivative activation function) for inner layers 
	public float compute_delta(){
		return der_activation_func(func_name,u);
	}

	// Get the first part for delta calculation for output layer
	public float compute_delta_final(float category){
		return der_activation_func(func_name,u)*(activation_func(func_name,u)-category);
	}

	// Reset delta and derivative of weights (dE) 
	public void clear_grads(){
		this.grad_weights = new float[D];
		this.delta=0.0f;
	}
}