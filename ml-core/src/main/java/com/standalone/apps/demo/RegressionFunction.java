package com.standalone.apps.demo;

import java.util.Arrays;
import java.util.function.Function;

public abstract class RegressionFunction<T extends Number, R extends Number> implements Function<T[], R> {
	private final T[] thetaVector;

	public RegressionFunction(T[] thetaVector) {
		this.thetaVector = Arrays.copyOf(thetaVector, thetaVector.length);
	}

	public T[] getThetas() {
		return Arrays.copyOf(thetaVector, thetaVector.length);
	}

	protected Double calculateLinearExpression(T[] featureVector) {
		// for computational reasons the first element has to be 1.0
		assert featureVector[0].doubleValue() == 1.0;

		// simple, sequential implementation
		Double prediction = 0.0;
		for (int j = 0; j < thetaVector.length; j++) {
			if (Double.isNaN(prediction.doubleValue()) || Double.isNaN(thetaVector[j].doubleValue()) || Double.isNaN(featureVector[j].doubleValue())
					|| Double.isInfinite(prediction.doubleValue()) || Double.isInfinite(thetaVector[j].doubleValue())
					|| Double.isInfinite(featureVector[j].doubleValue())) {
				System.out.println("Something went wrong||");
				System.out.println("prediction: " + prediction);
				System.out.println("thetaVector: " + thetaVector[j]);
				System.out.println("featureVector: " + featureVector[j]);
				System.exit(0);
			}
			prediction += thetaVector[j].doubleValue() * featureVector[j].doubleValue();
		}
		return prediction;
	}

}
