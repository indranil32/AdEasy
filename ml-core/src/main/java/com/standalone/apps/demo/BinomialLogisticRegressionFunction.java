package com.standalone.apps.demo;

public class BinomialLogisticRegressionFunction extends RegressionFunction<Double, Double> {

	BinomialLogisticRegressionFunction(Double[] thetaVector) {
		super(thetaVector);
	}

	@Override
	public Double apply(Double[] featureVector) {
		return 1/(1+Math.exp(-calculateLinearExpression(featureVector)));
	}

}
