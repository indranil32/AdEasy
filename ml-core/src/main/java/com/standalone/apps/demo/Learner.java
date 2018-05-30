package com.standalone.apps.demo;

import java.util.List;
import java.util.function.Function;

public class Learner {
	public static Double[] gradientDescent(RegressionFunction<Double, Double> targetFunction, List<Double[]> dataset,
			List<Double> labels, Double learningRate) {
		int m = dataset.size();
		Double[] thetaVector = targetFunction.getThetas();
		Double[] newThetaVector = new Double[thetaVector.length];

		// compute the new theta of each element of the theta array
		for (int j = 0; j < thetaVector.length; j++) {
			// summarize the error gap * feature
			Double sumErrors = 0.0;
			for (int i = 0; i < m; i++) {
				Double[] featureVector = dataset.get(i);
				Double error = targetFunction.apply(featureVector) - labels.get(i);
				sumErrors += error * featureVector[j];
			}

			// compute the new theta value
			Double gradient = (1.0 / m) * sumErrors;
			newThetaVector[j] = thetaVector[j] - learningRate * gradient;
		}
		return newThetaVector;
	}
	
	public static Double linearCost(Function<Double[], Double> targetFunction, List<Double[]> dataset, List<Double> labels) {
		int m = dataset.size();
		Double sumSquaredErrors = 0.0;

		// calculate the squared error ("gap") for each training example and add it to
		// the total sum
		for (int i = 0; i < m; i++) {
			// get the feature vector of the current example
			Double[] featureVector = dataset.get(i);
			// predict the value and compute the error based on the real value (label)
			Double predicted = targetFunction.apply(featureVector);
			Double label = labels.get(i);
			Double gap = predicted - label;
			sumSquaredErrors += Math.pow(gap, 2);
		}
		// calculate and return the mean value of the errors (the smaller the better)
		return (1.0 / (2 * m)) * sumSquaredErrors;
	}
	
	public static Double logisticCost(Function<Double[], Double> targetFunction, List<Double[]> dataset, List<Double> labels) {
		Double finalVal = 0.0;
		int m = dataset.size();
		for (int i = 0; i < m; i++) {
			Double predicted = targetFunction.apply(dataset.get(i));
			Double label = labels.get(i);
			Double step1 = label * Math.log(predicted);
			Double step2 = (1- label) * Math.log(1 - predicted);
			finalVal += -step1 - step2;
		}
		return (1.0 / m) * finalVal;
	}
}
