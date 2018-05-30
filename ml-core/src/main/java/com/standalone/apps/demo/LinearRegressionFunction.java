package com.standalone.apps.demo;

/**
 * Linear regression : h(x) = theta0 * 1 + theta1 * x1 + theta2 * x2 + .... 
 * 
 * @author guser2
 *
 */
public class LinearRegressionFunction extends RegressionFunction<Double, Double>{
	
	LinearRegressionFunction(Double[] thetaVector) {
		super(thetaVector);
	}

	public Double apply(Double[] featureVector) {
		return calculateLinearExpression(featureVector);
	}

}
