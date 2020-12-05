package br.furb;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class PCA {

	public static void main(String[] args) {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		/*Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
		System.out.println(mat.dump());*/
		
		// Carrega as imagens de dataset e já divide o conjunto em treino (70%) e teste (30%).
		String path = "D:\\PCA\\dataset\\ORL\\";
		List<Person> train = new ArrayList<>();
		List<Person> test = new ArrayList<>();
		int p = 7; // Holout de 70/30, treino/teste
		loadDataset(path, train, test, p);
		
		/*System.out.println("Treino:" + train.size());
		System.out.println("Teste:" + test.size());*/
		
		// Treino
		
		// Teste
		
		PCAEigenFace model = new PCAEigenFace(10);
		model.train(train);
		
		int corrects = 0;
		for (Person personTest: test) {
			Mat testData = personTest.getData();
			int[] label = new int[1];
			double[] confidence = new double[1];
			double[] reconstructionError = new double[1];
			
			model.predict(testData, label, confidence, reconstructionError);
			
			if (label[0] == personTest.getLabel()) {
				corrects++;
			}
			
		}
		
		double x = corrects / (double) test.size() * 100;
		System.out.println("Taxa de acerto:" + x);

	}

	private static void loadDataset(String path, List<Person> train, List<Person> test, int p) {
		File folder = new File(path);
		File[] filesArray = folder.listFiles((dir, fileName) -> fileName.toLowerCase().endsWith(".jpg"));
		
		List<Person> people = Arrays.asList(filesArray)
				.stream()
				.map(file -> toPerson(file.getPath()))
				.collect(Collectors.toList());
		
		//people.forEach(System.out::println);
		
		people.sort(Comparator.comparing(Person::getId));
		
		Random ran = new Random();
		final int numSamplesPerPerson = 10;
		List<Person> samples = new ArrayList<>(numSamplesPerPerson);
		people.forEach(person -> {
			samples.add(person);
			if (samples.size() == numSamplesPerPerson) {
				while (samples.size() > p) { // 1 - p% = 
					int index = ran.nextInt(samples.size());
					test.add(samples.remove(index));
				}
				
				if (p == numSamplesPerPerson) {
					test.addAll(samples);
				}
				
				train.addAll(samples);
				samples.clear();
			}
		});
		
	}

	private static Person toPerson(String filename) {
		Person person = new Person();
		
		// D:\PCA\dataset\ORL\1_1.jpg
		String dataPart = filename.substring(filename.lastIndexOf("\\") + 1, filename.lastIndexOf(".jpg"));
		String[] data = dataPart.split("_");
		
		person.setId(Integer.parseInt(data[0]));
		person.setLabel(Integer.parseInt(data[1]));
		person.setData(getImageData(filename));
		
		return person;
	}

	private static Mat getImageData(String filename) {
		
		Mat img = Imgcodecs.imread(filename, Imgcodecs.IMREAD_GRAYSCALE);
		
		Mat dst = new Mat();
		Imgproc.resize(img, dst, new Size(80, 80));
		
		// Converter a imagem de uma matriz de pixels em um vetor coluna
		// no formato de reconnhecimento de padrões.
		// 1 2
		// 3 4		
		
		// 1
		// 2
		// 3
		// 4
		
		// 1 3
		// 2 4
		// 1
		// 3
		// 2
		// 4
		
		// De imagem com 8 bits, sem sinal, 1 canal
		dst = dst.t().reshape(1, dst.cols() * dst.rows());
		
		Mat data = new Mat();
		// Para imagem com 64 bits, com sinal e ponto flutuante, 1 canal
		dst.convertTo(data, CvType.CV_64FC1);
		
		return data;
	}

}
