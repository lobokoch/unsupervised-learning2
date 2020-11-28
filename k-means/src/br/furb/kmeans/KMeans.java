package br.furb.kmeans;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
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
		
		samples.forEach(System.out::println);
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
				
		// Recalcula as variáveis dos centroides com base nas amostras de label igual ao centroide.
		recalcCentroids(samples, centroids);
		
		// Em caso dos centroides estabilizarem, para o algoritmo, caso contrário, chama a rotina recursivamente até estabilizar.
		if (hasChanges(previusCentroids, centroids)) {
			kmeans(samples, centroids);
		}
		
	}

	private static boolean hasChanges(List<Sample> previusCentroids, List<Sample> centroids) {
		int index = 0;
		// Inicia pegando a distância do primeiro centroide anterior e do primeiro atual.
		double dist = previusCentroids.get(index).getDistance(centroids.get(index));
		index++; // preprara para acessar o próximo item da lista, caso tenha um.
		// Na primeira ocorreência de uma distancia maior do que zero, já aborta e retorna true (tem mudanças)
		while (dist == 0 && index < centroids.size()) {
			dist = previusCentroids.get(index).getDistance(centroids.get(index));
			index++;
		}
		return dist != 0; // Retorna true em caso de haver alguma distância entre centroides, maior que zero, e false caso contrário.
	}

	private static void recalcCentroids(List<Sample> samples, List<Sample> centroids) {
		// "1"=Sample[C1:x1=1, x2=5]
		// "2"=Sample[C1]
		
		// Cria-se um mapa para guardar os centroids de forma indexada.
		Map<String, Sample> centroidsIndexed = centroids
				.stream()
				.peek(Sample::restData) // Reseta os valores do centroide para o recálculo
				.collect(Collectors.toMap(Sample::getLabel, c -> c));
		
		// Para cada amostra, somamos suas variáveis ao centroide correspondente.
		samples.forEach(sample -> {
			Sample centroid = centroidsIndexed.get(sample.getLabel());
			centroid.sumData(sample.getData());
		});
		
		// Dividimos o somatório de cada variável do centróide
		// pela quantidade de amostra que geraram esse somatório
		// produzindo assim a média do centroide.
		centroids.forEach(Sample::applyDataMean);
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
			sample.setOriginalLabel(Integer.toString(i + 1));
			result.add(sample);
		}
		
		return result;
	}

}
