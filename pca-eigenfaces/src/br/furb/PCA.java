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
import org.opencv.face.EigenFaceRecognizer;
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
		
		test.add(toPerson("D:\\dataset\\orl\\1002_182.jpg"));
		test.add(toPerson("D:\\dataset\\orl\\1003_182.jpg"));
		test.add(toPerson("D:\\dataset\\orl\\7000_777.jpg"));
		test.add(toPerson("D:\\dataset\\orl\\9000_999.jpg"));
		test.add(toPerson("D:\\dataset\\orl\\9001_999.jpg"));
		test.add(toPerson("D:\\dataset\\orl\\8000_888.jpg"));
		
		int start = 2;
		int end = 30;
		
		final int MAX_REC = 3500;
		final int MAX_DIS = 1700;
		
		for (int k = start; k <= end; k++) {
			PCAEigenFace model = new PCAEigenFace(k);
			model.train(train);
			
			
			/// Begin usando OpenCV
			EigenFaceRecognizer model2 = EigenFaceRecognizer.create(k);
			List<Mat> src = new ArrayList<>(train.size());
			Mat labels = new Mat(train.size(), 1, CvType.CV_32SC1);
			for (int i = 0; i < train.size(); i++) {
				Person person = train.get(i);
				src.add(person.getData());
				labels.put(i, 0, person.getLabel());
			}
			model2.train(src, labels);
			/// End usando OpenCV
			
			
			/////////////////////////////
			double minDis = Double.MAX_VALUE;
			double maxDis = Double.MIN_VALUE;
			
			double minRec = Double.MAX_VALUE;
			double maxRec = Double.MIN_VALUE;
			
			/////////////////////////////
			int trueNegativesCount = 0;
			int truePositivesCount = 0;
			
			int corrects = 0;
			int corrects2 = 0;
			for (Person personTest: test) {
				Mat testData = personTest.getData();
				int[] label = new int[1];
				double[] confidence = new double[1];
				double[] reconstructionError = new double[1];
				model.predict(testData, label, confidence, reconstructionError);
				
				/// Begin usando OpenCV
				int[] label2 = new int[1];
				double[] confidence2 = new double[1];
				model2.predict(testData, label2, confidence2);
				boolean labelOk2 = label2[0] == personTest.getLabel();
				if (labelOk2) {
					corrects2++;
				}
				/// End usando OpenCV
				
				boolean labelOk = label[0] == personTest.getLabel();
				if (labelOk) {
					corrects++;
				}
				
				if (reconstructionError[0] > MAX_REC) {
					System.out.println("NOT A PERSON - Predicted label" + label[0] +
							", confidence:" + confidence[0] + 
							", reconstructionError:" + reconstructionError[0] + 
							", original label:" + personTest.getLabel()
							);
					
					if (!labelOk) {
						trueNegativesCount++;
					}
				} else if (confidence[0] > MAX_DIS) {
					System.out.println("UNKNOWN PEOPLE (by distance) - Predicted label" + label[0] +
							", confidence:" + confidence[0] + 
							", reconstructionError:" + reconstructionError[0] + 
							", original label:" + personTest.getLabel()
							);
					
					if (!labelOk) {
						trueNegativesCount++;
					}
				} else if (reconstructionError[0] > 2400 && confidence[0] > 1500) {
					System.out.println("UNKNOWN PEOPLE (by two factors) - Predicted label" + label[0] +
							", confidence:" + confidence[0] + 
							", reconstructionError:" + reconstructionError[0] + 
							", original label:" + personTest.getLabel()
							);
					
					if (!labelOk) {
						trueNegativesCount++;
					}
				} else if (labelOk) {
					truePositivesCount++;
				} else {
					System.out.println("UNKNOWN - Predicted label" + label[0] +
							", confidence:" + confidence[0] + 
							", reconstructionError:" + reconstructionError[0] + 
							", original label:" + personTest.getLabel()
							);
				}
				
				////
				if (labelOk && personTest.getId() <= 400) {
					if (confidence[0] < minDis) {
						minDis = confidence[0];
					}
					if (confidence[0] > maxDis) {
						maxDis = confidence[0];
					}
					//////////////////
					if (reconstructionError[0] < minRec) {
						minRec = reconstructionError[0];
					}
					if (reconstructionError[0] > maxRec) {
						maxRec = reconstructionError[0];
					}
				}
				////
				
				if (personTest.getLabel() == 182) {
					System.out.format("Label:%d, confidence:%.2f, reconstructionError:%.2f%n", 
							label[0], confidence[0], reconstructionError[0]);
				}
				
				
				
			}
			
			System.out.format("%ncorrect:%d%ncorrectOpenCV:%d%n%n", corrects, corrects2);
			
			int trues = truePositivesCount + trueNegativesCount;
			double accuracy = (double) trues / test.size() * 100;
			System.out.format("K=%d, taxa de acerto=%.2f%n", k, accuracy);
			
			//double x = corrects / (double) test.size() * 100;
			System.out.format("minDis=%.2f, maxDis=%.2f, minRec=%.2f, maxRec=%.2f %n", 
					minDis, maxDis, minRec, maxRec);
			
			System.out.println("*********************************************************************");
			
		}

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
