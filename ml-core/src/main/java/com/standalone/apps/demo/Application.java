package com.standalone.apps.demo;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 
 * @author indranilm
 *
 */
public class Application {
	public static void main(String args[]) {
		linear();
		logistic();
	}

	/**
	 * Linear Regression boils down to four operations: 
	 * 1. Calculate the hypothesis h = X * theta 
	 * 2. Calculate the loss = h - y and maybe the squared cost (loss^2)/2m 
	 * 3. Calculate the gradient = X' * loss / m 4. Update the parameters theta = theta - alpha * gradient
	 */
	public static void linear() {
		// create the dataset
		List<Double[]> dataset = new ArrayList<Double[]>();
									//	x,		x^2
		dataset.add(new Double[] { 1.0, 90.0, 8100.0 }); // feature vector of house#1
		dataset.add(new Double[] { 1.0, 101.0, 10201.0 }); // feature vector of house#2
		dataset.add(new Double[] { 1.0, 103.0, 10609.0 }); // feature vector of house#3
		dataset.add(new Double[] { 1.0, 203.0, 20609.0 }); // feature vector of house#4
		dataset.add(new Double[] { 1.0, 406.0, 40609.0 }); // feature vector of house#5
		dataset.add(new Double[] { 1.0, 306.0, 30609.0 }); // feature vector of house#6
		dataset.add(new Double[] { 1.0, 154.0, 15509.0 }); // feature vector of house#7

		// create the labels
		List<Double> labels = new ArrayList<Double>();
		labels.add(249.0); // price label of house#1
		labels.add(338.0); // price label of house#2
		labels.add(304.0); // price label of house#3
		labels.add(649.0); // price label of house#4
		labels.add(1238.0); // price label of house#5
		labels.add(954.0); // price label of house#6
		labels.add(452.0); // price label of house#7

		dataset.forEach(action -> {
			System.out.print("Normal dataset : ");
			for (Double d : action) {
				System.out.print(d);
				System.out.print(" , ");
			}
			System.out.println();
		});
		
		int [] posA = {1,2};
		// scale the extended feature list
		Function<Double[], Double[]> scalingFunc = FeaturesScaling.createLinearNormalizationFunction(dataset, posA);
		List<Double[]> scaledDataset = dataset.stream().map(scalingFunc).collect(Collectors.toList());
		scaledDataset.forEach(action -> {
			System.out.print("scaledDataset : ");
			for (Double d : action) {
				System.out.print(d);
				System.out.print(" , ");
			}
			System.out.println();
		});
		// create hypothesis function with initial thetas and train it with learning
		// rate 0.1
		RegressionFunction<Double, Double> targetFunction = new LinearRegressionFunction(new Double[] { 1.0, 1.0, 1.0 });
		Double cost = Learner.linearCost(targetFunction, scaledDataset, labels);
		targetFunction = train(labels, scaledDataset, targetFunction, ALGO_TYPE.LINEAR, 0.1, 0.0000001, cost);
		/*for (int i = 0; i < 10000; i++) {
			// gradient descent
			targetFunction =  new LinearRegressionFunction(Learner.gradientDescent(targetFunction, scaledDataset, labels, 0.1)); // learning rate
			System.out.print("newThetaVector : ");
			for (double d : targetFunction.getThetas()) {
				System.out.print(d);
				System.out.print(" , ");
			}
			System.out.println();
			double cost = Learner.linearCost(targetFunction, scaledDataset, labels);
			System.out.println("Cost after training " + i + " training iterations is : " + cost);
		}*/

		// make a prediction of a house with size if 600 m2
		Double[] scaledFeatureVector = scalingFunc.apply(new Double[] { 1.0, 250.0, 25000.0 });
		System.out.print("scaledFeatureVector to predict : ");
		for (Double d : scaledFeatureVector) {
			System.out.print(d);
			System.out.print(" , ");
		}
		System.out.println();
		double predictedPrice = targetFunction.apply(scaledFeatureVector);
		System.out.println("final predictedPrice: " + predictedPrice);
	}

	public static void logistic() {
		// create the dataset
		List<Double[]> dataset = new ArrayList<Double[]>();
									//  hrs	of study
		dataset.add(new Double[] { 1.0, 0.05}); 
		dataset.add(new Double[] { 1.0, 0.12});
		dataset.add(new Double[] { 1.0, 0.26}); 
		dataset.add(new Double[] { 1.0, 0.31});
		dataset.add(new Double[] { 1.0, 0.44}); 
		dataset.add(new Double[] { 1.0, 0.5});
		dataset.add(new Double[] { 1.0, 0.65}); 
		dataset.add(new Double[] { 1.0, 0.67});
		dataset.add(new Double[] { 1.0, 0.8}); 
		dataset.add(new Double[] { 1.0, 0.82});
		dataset.add(new Double[] { 1.0, 1.02}); 
		dataset.add(new Double[] { 1.0, 1.15});
		dataset.add(new Double[] { 1.0, 1.35});
		dataset.add(new Double[] { 1.0, 1.6}); 
		dataset.add(new Double[] { 1.0, 1.5});
		dataset.add(new Double[] { 1.0, 1.99}); 
		dataset.add(new Double[] { 1.0, 1.99}); 
		dataset.add(new Double[] { 1.0, 1.9});
		dataset.add(new Double[] { 1.0, 2.59}); 
		dataset.add(new Double[] { 1.0, 2.09});
		dataset.add(new Double[] { 1.0, 3.0}); 
		dataset.add(new Double[] { 1.0, 2.89});
		
		dataset.forEach(action -> {
			System.out.print("Normal dataset : ");
			for (Double d : action) {
				System.out.print(d);
				System.out.print(" , ");
			}
			System.out.println();
		});
		
		// create the labels fail = 0.0/pass = 1.0
		List<Double> labels = new ArrayList<Double>();
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(0.0);
		labels.add(1.0);
		labels.add(1.0);
		labels.add(1.0);
		labels.add(1.0);
		labels.add(1.0);
		labels.add(1.0);
		labels.add(1.0);
		labels.add(1.0);
		labels.add(1.0);
		int [] posA = {1};
		// scale the extended feature list
		Function<Double[], Double[]> scalingFunc = FeaturesScaling.createLogisticNormalizationFunction(dataset, posA);
		List<Double[]> scaledDataset = dataset.stream().map(scalingFunc).collect(Collectors.toList());
		scaledDataset.forEach(action -> {
			System.out.print("FeaturesScaled dataset : ");
			for (Double d : action) {
				System.out.print(d);
				System.out.print(" , ");
			}
			System.out.println();
		});
		// create hypothesis function with initial thetas and train it with learning rate 0.01
		RegressionFunction<Double, Double> targetFunction = new BinomialLogisticRegressionFunction(scalingFunc.apply(new Double[] { 1.0, 1.5}));
		Double cost = Learner.logisticCost(targetFunction, scaledDataset, labels);
		targetFunction = train(labels, scaledDataset, targetFunction, ALGO_TYPE.LOGISTIC, 0.01, 0.0000001, cost);

		Double[] scaledFeatureVector = scalingFunc.apply(new Double[] { 1.0, 1.87});
		System.out.print("scaledFeatureVector to predict : ");
		for (Double d : scaledFeatureVector) {
			System.out.print(d);
			System.out.print(" , ");
		}
		System.out.println();
		double predicted = targetFunction.apply(scaledFeatureVector);
		System.out.println("final predicted: " + (predicted > 0.5 ? "pass" : "fail"));
		
		scaledFeatureVector = scalingFunc.apply(new Double[] { 1.0, 0.87});
		System.out.print("scaledFeatureVector to predict : ");
		for (Double d : scaledFeatureVector) {
			System.out.print(d);
			System.out.print(" , ");
		}
		System.out.println();
		predicted = targetFunction.apply(scaledFeatureVector);
		System.out.println("final predicted: " + (predicted > 0.5 ? "pass" : "fail"));
	}

	private static RegressionFunction<Double, Double> train(List<Double> labels, List<Double[]> scaledDataset,
			RegressionFunction<Double, Double> targetFunction, ALGO_TYPE type, Double lr, Double convergeChange,Double cost) {
		System.out.println("Cost without training is : " + cost);
		Double changeCost = 1.0;
		int i = 1;
		//for (int i = 0; i < 1000; i++) {
		while(changeCost > convergeChange) {
			Double oldCost = cost;
			// gradient descent
			switch (type) {
				case LINEAR:
					targetFunction =  new LinearRegressionFunction(Learner.gradientDescent(targetFunction, scaledDataset, labels, lr)); // learning rate
					cost = Learner.linearCost(targetFunction, scaledDataset, labels);
					break;
				case LOGISTIC:
					targetFunction = new BinomialLogisticRegressionFunction(Learner.gradientDescent(targetFunction, scaledDataset, labels, lr)); // learning rate
					cost = Learner.logisticCost(targetFunction, scaledDataset, labels);
					break;
				default:
					break;
			}
			/*System.out.print("newThetaVector : ");
			for (double d : targetFunction.getThetas()) {
				System.out.print(d);
				System.out.print(" , ");
			}
			System.out.println();*/
			changeCost = oldCost - cost;
			i++;
		}
		System.out.println("Cost after training " + i + " training iterations is : " + cost);
		return targetFunction;
	}

	public static void decisionTree() {

	}

	public static void neural() {

	}
	
	enum ALGO_TYPE {LINEAR, LOGISTIC, DECISION_TREE, NUERAL}
}
