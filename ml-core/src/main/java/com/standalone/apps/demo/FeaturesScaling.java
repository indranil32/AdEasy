package com.standalone.apps.demo;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * 
 * @author indranilm
 *
 */
public class FeaturesScaling {

	static Map<Integer, Double> max = new HashMap<>();
	static Map<Integer, Double> min = new HashMap<>();
	static Map<Integer, Double> avg = new HashMap<>();
	
	public static Function<Double[], Double[]> createLinearNormalizationFunction(List<Double[]> dataset, int[] posA) {
		deviation(dataset, posA);
		Function<Double[], Double[]> f = Normalization::normalizelinear;
		return f;
	}

	public static Function<Double[], Double[]> createLogisticNormalizationFunction(List<Double[]> dataset, int[] posA) {
		deviation(dataset, posA);
		Function<Double[], Double[]> f = Normalization::normalizeLogistic;
		return f;
	}

	private static double avg(List<Double[]> dataset, int pos) {
		return dataset.stream().mapToDouble(a -> a[pos]).average().getAsDouble();
	}

	private static double min(List<Double[]> dataset, int pos) {
		return dataset.stream().max(new VectorComparator<Double>(pos)).get()[pos];
	}

	private static double max(List<Double[]> dataset, int pos) {
		return dataset.stream().min(new VectorComparator<Double>(pos)).get()[pos];
	}

	private static void deviation(List<Double[]> dataset, int[] posA) {
		for (int i = 0 ; i < posA.length ; i++) {
			max.put(posA[i], max(dataset, posA[i]));
			min.put(posA[i], min(dataset, posA[i]));
			avg.put(posA[i], avg(dataset, posA[i]));
			System.out.println("max"+posA[i]+" :"+max.get(posA[i]));
			System.out.println("min"+posA[i]+" :"+min.get(posA[i]));
			System.out.println("avg"+posA[i]+" :"+avg.get(posA[i]));
		}
	}
	
	static class Normalization {

		/**
		 * norm_X' = (x-avg(x))/(max-min)
		 * 
		 * 
		 * @param data
		 * @return 
		 */
		private static Double[] normalizelinear(Double[] data) {
			for (int i = 0 ; i < data.length  ; i++) {
				if (max.get(i) != null)
					data[i] = (data[i] - avg.get(i))/(max.get(i) - min.get(i));  
			}
			return data;
		}
		
		
		/**
		 * norm_X' = (x-min(x))/(max-min)
		 * 
		 * 
		 * @param data
		 * @return 
		 */
		private static Double[] normalizeLogistic(Double[] data) {
			for (int i = 0 ; i < data.length  ; i++) {
				if (max.get(i) != null)
					data[i] = (data[i] - min.get(i))/(max.get(i) - min.get(i));  
			}
			return data;
		}
		
	}
	
	static class VectorComparator<T extends Comparable<T>> implements Comparator<T[]> {
		int pos;
		
		public VectorComparator(int pos) {
			this.pos = pos;
		}
		
		@Override
		public int compare(T[] o1, T[] o2) {
			return o2[pos].compareTo(o1[pos]);
		}
		
	}
}
