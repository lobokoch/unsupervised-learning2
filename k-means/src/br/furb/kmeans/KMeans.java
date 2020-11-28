package br.furb.kmeans;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

public class KMeans {

	public static void main(String[] args) throws Exception {
		
		// Carregar as amostras ou objetos.
		//List<Sample> samples = loadData();
		
//		String filename = "D:\\git\\furb\\aprendizado-nao-supervisionado\\dataset\\DataHotDogs.csv";
//		boolean hasHeader = true; 
//		int labelColumnIndex = 0;
//		String separator = ",";
		
		String filename = "D:\\git\\furb\\aprendizado-nao-supervisionado\\dataset\\iris.data";
		boolean hasHeader = false; 
		int labelColumnIndex = 4;
		String separator = ",";
		
		List<Sample> samples = loadData(filename, hasHeader, labelColumnIndex, separator);
		//samples.forEach(System.out::println);
		
		// Armazenará os valores de silhoeta de cada k, para poder obter o melhor k.
		// O mlehor k, será o que tiver o maior valor de silhoeta.
		Map<Integer, Double> coefficients = new HashMap<>();
		
		int k = 9; // C1, C2, C3
		// Lista para os k centroids
		List<Sample> centroids = new ArrayList<>();
		
		//centroids.add(samples.get(0).buildClone().setLabel(Integer.toString(centroids.size() + 1)));
		//centroids.add(samples.get(2).buildClone().setLabel(Integer.toString(centroids.size() + 1)));
		
		// Calcula o k-means efetivamente
		kmeans(samples, k, centroids);
		// C1, C2, C3
		// Que cada amostra esteja associada a um label (classe) de um centroide.
		
		samples.forEach(System.out::println);
		System.out.println("----------------");
		centroids.forEach(System.out::println);
		
		//1=Iris-setosa:10
		//1=Iris-versicolor:5
		//1=Iris-virginica:3
		
		//2=Iris-setosa:0
		//2=Iris-versicolor:10
		//2=Iris-virginica:0
		Map<String, Map<String, Integer>> statistics = new HashMap<>();
		samples.forEach(sample -> {
			Map<String, Integer> mapClass = statistics.get(sample.getLabel());
			if (mapClass == null) {
				mapClass = new HashMap<>();
				statistics.put(sample.getLabel(), mapClass);
			}
			
			Integer count = mapClass.get(sample.getOriginalLabel());
			if (count == null) {
				count = 0;
			}
			count++;
			mapClass.put(sample.getOriginalLabel(), count);
			
		});
		
		statistics.forEach((clusterLabel, itens) -> {
			System.out.println("---------------");
			itens.forEach((name, count) -> {
				System.out.println(clusterLabel + "=" + name + ":" + count);
			});
		});
		
		calcSilhouette(samples, centroids, coefficients);
		System.out.format("%nSilhueta para k=%d%n", k);
		showSilhouette(samples, centroids);

	}

	private static void showSilhouette(List<Sample> samples, List<Sample> centroids) {
		// -1   0   +1
		//      |22222 (0.5)
		//      |22222222 0.8
		//      |22222222 0.8
		//   222| -0.3
		
		centroids.forEach(centroid -> {
			
			List<Sample> objects = samples.stream()
					.filter(it -> it.getLabel().equals(centroid.getLabel()))
					.collect(Collectors.toList());
			
			objects.forEach(sample -> {
				int factor = 100;
				int si = (int) Math.round((sample.getSilhouette() * factor));
				
				if (sample.getSilhouette() >= 0) { // si positivo
					for (int i = 0; i < factor; i++) {
						System.out.print(i == factor - 1 ? "|" : " ");
					}
				} else { // si negativo
					si = si * -1;
					int max = factor - si;
					for (int i = 0; i < max - 1; i++) {
						System.out.print(" ");
					}
				}
				
				// A cor da barra é o label do centroid
				for (int i = 0; i < si; i++) {
					System.out.print(centroid.getLabel());
				}
				
				if (sample.getSilhouette() < 0) {
					System.out.print("|");
				}
				
				System.out.printf(" (%.2f)", sample.getSilhouette());
				System.out.println(" "); // Imprimi uma quebra de linha.
				
			});
			
			System.out.println(" "); // Imprimi um espaço entre cada cluster.
			
		});
		
	}

	private static void calcSilhouette(List<Sample> samples, 
			List<Sample> centroids,
			Map<Integer, Double> coefficients) {
		
		// 1) Calcular ai dentro do Ci.
		// 2) Calcular bi do vizimho mais próximo.
		
		Map<String, Sample> indexedCentroids = centroids
				.stream()
				.collect(Collectors.toMap(Sample::getLabel, c -> c));
				
		samples.forEach(sample -> {
			
			Sample myCluster = indexedCentroids.get(sample.getLabel());
			
			// Passo 1.
			double ai = calcMedia(sample, samples, myCluster);
			
			List<Sample> neigborCentroids = new ArrayList<>(centroids);
			neigborCentroids.remove(myCluster);
			
			// Passo 2.
			// Obtém a menor média entre "sample" e as "sample" dos demais clusters.
			double[] bi_min = new double[1];
			bi_min[0] = Double.MAX_VALUE;
			neigborCentroids.forEach(Ck -> {
				double bi = calcMedia(sample, samples, Ck);
				if (bi < bi_min[0]) {
					bi_min[0] = bi;
				}
			});
			
			// Conta a quantidade de "sample" em "samples" do cluster Ci, o cluster em questão.
			long CiSize = samples.stream()
					.filter(it -> it.getLabel().equals(myCluster.getLabel()))
					.count();
			
			// Calcula efetivamente si.
			double si = CiSize > 1 
					? (bi_min[0] - ai) / Math.max(ai, bi_min[0])
					: 0;
					
			sample.setSilhouette(si);
			
			// Somando todos os valores de silhueta por cluster (k) para depois calcular o melhor k.
			int k = centroids.size();
			coefficients.put(k, coefficients.getOrDefault(k, 0.0) + si);
			
		});
		
	}

	private static double calcMedia(Sample sample, 
			List<Sample> samples, 
			Sample cluster) {
		
		// Obtém todas as amostras do universo X (samples), cujo label for igual
		// ao label do cluster passado.
		List<Sample> myFriends = samples.stream()
				.filter(it -> it.getLabel().equals(cluster.getLabel()) && it != sample)
				.collect(Collectors.toList());
		
		// Somando as distâncias da amostra passada, o "ai" no caso, com as demais amostras
		// do cluster passado como parâmetro.
		double[] distanceSum = new double[1];
		myFriends.forEach(friend -> {
			distanceSum[0] += sample.getDistance(friend);
		});
		
		// Calcula a média efetivamente, dividindo o somatório, pela quantidade de amostras
		// do cluster passado.
		return distanceSum[0] / myFriends.size();
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

	private static List<Sample> loadData(String filename, boolean hasHeader, 
			int labelColumnIndex, String separator) throws Exception {
		
		//label,x1,x2,xn
		//1,x,xx
		//2,x2,xx2
		//3,x30,xx25
		
		//x,xx,1
		//x2,xx,2
		//x30,xx25,3
		//
		//
		
		List<Sample> result = new ArrayList<>();
		Scanner scan = new Scanner(new File(filename));
		
		if (hasHeader) {
			scan.nextLine(); // Descarta a linha do cabeçalho do arquivo.
		}
		
		// Itera por todas as linhas do arquivo.
		while (scan.hasNextLine()) {
			String line = scan.nextLine();
			if (!line.isBlank()) { // Descarta linhas em branco
				String[] dataStr = line.split(separator); // Separa em um array cada valor da linha, ou seja, cada coluna em uma posição do array.
				Sample sample = new Sample(); // Cria um objeto de amostra novo.
				double[] data = new double[dataStr.length - 1]; // Prepara o array para receber as variáveis x1, x2, xn.
				int j = 0; // Controle próprio para o array de vars x1,x2,xn.
				for (int i = 0; i < dataStr.length; i++) { // Itera pelas colunas da linha, separando o que é label do que é variável x1, x2. xn.
					if (i == labelColumnIndex) { // Indica que está "olhando" para a coluna do label.
						sample.setOriginalLabel(dataStr[i]);
					} else { // Está olhando uma coluna x1,x2,xn.
						data[j] = Double.parseDouble(dataStr[i]); // Como é um arquivo texto, mesmo que é um número no arquivo, deve ser convertido de string para double.
						j++;
					}
				}
				sample.setData(data);
				result.add(sample);
			} //if
		} // while
		
		return result;
		
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
