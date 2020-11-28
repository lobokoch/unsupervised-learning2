package br.furb.kmeans;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

public class KMeans {

	public static void main(String[] args) {
		
		// Carregar as amostras ou objetos.
		List<Sample> samples = loadData();
		
		//samples.forEach(System.out::println);
		
		int k = 3; // C1, C2, C3
		// Lista para os k centroids
		List<Sample> centroids = new ArrayList<>();
		
		centroids.add(samples.get(0).buildClone().setLabel(Integer.toString(centroids.size() + 1)));
		centroids.add(samples.get(2).buildClone().setLabel(Integer.toString(centroids.size() + 1)));
		
		// Calcula o k-means efetivamente
		kmeans(samples, k, centroids);
		// C1, C2, C3
		// Que cada amostra esteja associada a um label (classe) de um centroide.
		
		centroids.forEach(System.out::println);

	}

	private static void kmeans(List<Sample> samples, int k, List<Sample> centroids) {
		if (centroids.isEmpty()) {
			getRandomCentroids(samples, k, centroids);
		}
		
		kmeans(samples, centroids);
		
	}

	private static void kmeans(List<Sample> samples, List<Sample> centroids) {
		// Coloca cada amostra (objeto) ao seu centroide mais próximo
		// A amostra fica com a mesma classe/label do centroide.
		samples.forEach(sample -> computeNearestCentroid(sample, centroids));
		
		// Clone a lista de centroids antes de recalcular ela
		// Para poder comparar os centroids antes e depois do recálculo.
		// A ideia é ver se teve mudança dos centroides anteriores para os
		// centroides reclaculados.
		List<Sample> previusCentroids = centroids
				.stream()
				.map(Sample::buildClone)
				.collect(Collectors.toList());
				
				
		recalcCentroids(samples, centroids);
		if (hasChanges(previusCentroids, centroids)) {
			kmeans(samples, centroids);
		}
		
	}

	private static void computeNearestCentroid(Sample sample, List<Sample> centroids) {
		double minDist = Double.MAX_VALUE;
		for (int i = 0; i < centroids.size(); i++) {
			Sample centroid = centroids.get(i);
			double dist = sample.getDistance(centroid);
			if (dist < minDist) {
				minDist = dist;
				sample.setLabel(centroid.getLabel());
			}
		}
		
	}

	private static void getRandomCentroids(List<Sample> samples, 
			int k, List<Sample> centroids) {
		
		Set<Integer> usedIndexes = new HashSet<>(k);
		centroids.clear();
		Random rand = new Random();
		while (centroids.size() < k) {
			// 0.. samples.size()-1 // 0..6
			int index = rand.nextInt(samples.size());
			
			// Evitar que pegue indices duplicados.
			while (usedIndexes.contains(index)) {
				index = rand.nextInt(samples.size());
			}
			usedIndexes.add(index);
			
			Sample sample = samples.get(index);
			Sample centroid = sample.buildClone();
			centroids.add(centroid);
			centroid.setLabel(Integer.toString(centroids.size()));
		}
		
	}

	private static List<Sample> loadData() {
		List<Sample> result = new ArrayList<>();
		
		double[][] data = { {1,1}, {2,1}, {3,2}, {2,4.5}, {1,5}, {3,7}, {6,5} };
		
		for (int i = 0; i < data.length; i++) {
			double[] data_i = data[i];
			Sample sample = new Sample();
			sample.setData(data_i);
			sample.setLabel(Integer.toString(i + 1));
			result.add(sample);
		}
		
		return result;
	}

}
