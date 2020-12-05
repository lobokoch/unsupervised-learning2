package br.furb;

import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class PCAEigenFace {
	
	private int numComponents;
	private Mat mean; // Vai produ��o
	private Mat diffs;
	private Mat covariance;
	private Mat eigenvalues;
	private Mat eigenvectors;
	private Mat eigenfaces; // Vai produ��o
	private Mat projections; // Vai produ��o
	private int[] labels; // Vai produ��o
	
	public PCAEigenFace(int numComponents) {
		this.numComponents = numComponents;
	}
	
	public void train(List<Person> train) {
		calcMean(train);
		calcDiff(train);
		calcCovariance();
		calcEigen();
		calcEigenFaces();
		calcProjections(train);
	}
	
	// Proje a cada imagem de treino em um novo espa�os com k dimens�es, onde k << person.size
	private void calcProjections(List<Person> train) {
		// k = 4
		// 1 9 7
		// 4 5 8
		// 7 8 0
		// 6 1 7
		
		labels = new int[train.size()]; // 1 label para cada imagem de treino.
		projections = new Mat(numComponents, train.size(), CvType.CV_64FC1);
		for (int j = 0; j < diffs.cols(); j++) {
			Mat diff = diffs.col(j);
			Mat w/*k x 1*/ = mul(eigenfaces.t(), diff); // U=(6400 x k)t, Ut=k x 6400 * diff= 6400 x 1 = w=k x 1
			w.copyTo(projections.col(j));
			labels[j] = train.get(j).getLabel();
		}
		
	}

	private void calcEigenFaces() {
		Mat evt = eigenvectors.t();
		int k = numComponents > 0 ? numComponents : evt.cols();
		numComponents = k;
		Mat ev_k = evt.colRange(0, k);
		for (int j = 0; j < ev_k.cols(); j++) {
			evt.col(j).copyTo(ev_k.col(j));
		}
		
		eigenfaces = mul(diffs, ev_k);
		// k =3
		// 1 9 7
		// 4 5 8
		// 7 8 0
		// 6 1 7
		for (int j = 0; j < eigenfaces.cols(); j++) {
			Mat ef = eigenfaces.col(j);
			// Normaliza��o L2 = Yi = Xi / sqrt(sum(Xi)^2)), onde i = 0, ... rows-1
			Core.normalize(ef, ef);
		}
		
		printEigenFaces();
	}

	private void printEigenFaces() {
		for (int j = 0; j < eigenfaces.cols(); j++) {
			Mat y = new Mat(eigenfaces.rows(), 1, eigenfaces.type());
			eigenfaces.col(j).copyTo(y.col(0));
			//saveImagem(y, "D:\\PCA\\eigenfaces\\e_" + (j + 1) + ".jpg");
		}
	}

	private void calcEigen() {
		eigenvalues = new Mat();
		eigenvectors = new Mat();
		Core.eigen(covariance, eigenvalues, eigenvectors);
		
		printEigenValues();		
	}
	
	private void printEigenValues() {
		// Soma os eigenvalues
		double sum = 0;
		for (int i = 0; i < eigenvalues.rows(); i++) {
			sum += eigenvalues.get(i, 0)[0];
		}
		
		// Calcula o percentual de contribui��o de cada eigenvalue na explica��o dos dados.
		double acumulado = 0;
		for (int i = 0; i < eigenvalues.rows(); i++) {
			double v = eigenvalues.get(i, 0)[0];
			double percentual = v / sum * 100;
			acumulado += percentual;
			System.out.format("CP%d, percentual: %.2f (%.2f)%n", (i + 1), percentual, acumulado);
		}
	}

	// Calcula a matriz de covariancia.
	private void calcCovariance() {
		covariance = mul(diffs.t(), diffs);
	}
	
	private Mat mul(Mat a, Mat b) {
		// a=400 x 6400 * b=6400 x 400, c= 400 x 400.
		Mat c = new Mat(a.rows(), b.cols(), CvType.CV_64FC1);
		/*
		for (int i = 0; i < c.rows(); i++) {
			double v = 0;
			for (int j = 0; j < c.cols(); j++) {
				for (int k = 0; k < a.cols(); k++) {
					double av = a.get(i,  k)[0];
					double bv = b.get(k, j)[0];
					v += av * bv;
				}
				c.put(i, j, v);
			}
		}*/
		
		// Begin OpenCV
		Core.gemm(a, b, 1, new Mat(), 1, c);
		// End OpenCV
		
		return c;
	}

	// Calcula a diferen�a entre cada imagem e a m�dia e armazena na matriz diffs.
	private void calcDiff(List<Person> train) {
		Mat sample = train.get(0).getData();
		// 1 9 7
		// 4 5 8
		// 7 8 0
		// 6 1 7
		diffs = new Mat(sample.rows(), train.size(), sample.type());
		for (int i = 0; i < diffs.rows(); i++) {
			for (int j = 0; j < diffs.cols(); j++) {
				double mv = mean.get(i, 0)[0];
				Mat data = train.get(j).getData();
				double pv = data.get(i, 0)[0];
				double v = pv - mv;
				diffs.put(i, j, v);
			}			
		}
		
		// Begin OpenCV
//		for (int i = 0; i < train.size(); i++) {
//			Core.subtract(train.get(i).getData(), mean, diffs.col(i));
//		}
		// End OpenCV
		
	}

	// Calcula a imagem m�dia.
	private void calcMean(List<Person> train) {
		Mat sample = train.get(0).getData();
		mean = Mat.zeros(/*6400*/sample.rows(), /*1*/sample.cols(), /*CvType.CV_64FC1*/sample.type());
		
		/// Begin Calculado na m�o
		train.forEach(person -> {
			Mat data = person.getData();
			for (int i = 0; i < mean.rows(); i++) {
				double mv = mean.get(i, 0)[0]; // Obt�m o valor da c�lula no primeiro canal.
				double pv = data.get(i, 0)[0]; // Obt�m o valor da c�lula no primeiro canal.
				mv += pv;
				mean.put(i, 0, mv);
			}
		});
		
		int M = train.size();
		for (int i = 0; i < mean.rows(); i++) {
			double mv = mean.get(i, 0)[0]; // Obt�m o valor da c�lula no primeiro canal.
			mv /= M;
			mean.put(i, 0, mv);
		}
		/// End Calculado na m�o
		
		// Begin OpenCV
		// 1 9 7
		// 4 5 8
		// 7 8 0
		// 6 1 7
//		Mat src = new Mat(sample.rows(), train.size(), sample.type());
//		for (int i = 0; i < train.size(); i++) {
//			train.get(i).getData().col(0).copyTo(src.col(i));
//		}
//		
//		Mat mean2 = Mat.zeros(sample.rows(), sample.cols(), sample.type());
//		Core.reduce(src, mean2, /*0=linha, 1=coluna*/1, Core.REDUCE_AVG, mean.type());
		// End OpenCV
		
		/*saveImagem(mean, "D:\\PCA\\mean1.jpg");
		saveImagem(mean2, "D:\\PCA\\mean2.jpg");*/
	}
	
	private void saveImagem(Mat image, String filename) {
		// [1,2,3,4,5]t
		// 1 2
		// 2 3
		Mat dst = new Mat();
		Core.normalize(image, dst, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
		
		// 6400 x 1
		// 80 x 80
		
		dst = dst.reshape(1, 80);
		dst = dst.t();
		
		Imgcodecs.imwrite(filename, dst);
	}

	public void predict(Mat testData, int[] label, double[] confidence, double[] reconstructionError) {
		Mat diff = new Mat();
		// Subtrai a imagem desconhecida, da imagem m�dia.
		Core.subtract(testData, mean, diff);
		
		// Projeta a imagem desconhecida, no mesmo espa�o das images de treino.
		Mat w = mul(eigenfaces.t(), diff);
		
		// Calcula a imagem de treino mais pr�xima da imagem desconhecida que foi projetada.
		int minJ = 0;
		double minDistance = calcDistance(w, projections.col(minJ));
		for (int j = 1; j < projections.cols(); j++) {
			double distance = calcDistance(w, projections.col(j));
			if (distance < minDistance) {
				minDistance = distance;
				minJ = j;
			}
		}
		
		// Obt�m o label e a dist�ncia da imagem de treino mais pr�xima da imagem de teste
		// e as retorna como resposta
		label[0] = labels[minJ];
		confidence[0] = minDistance;
		
		// Calcular o erro de reconstru��o.
		Mat reconstruction = calcReconstruction(w);
		reconstructionError[0] = Core.norm(testData, reconstruction, Core.NORM_L2);
	}

	private Mat calcReconstruction(Mat w) {
		Mat result = mul(eigenfaces, w); //[eigenfaces=6400 x k] * [w=k x 1] = result=6400 x 1.
		// result += mean
		Core.add(result, mean, result);
		return result;
	}

	private double calcDistance(Mat p, Mat q) {
		// Calcula a dist�ncia euclidiana
		// d = sqrt(sum(pi - q1)^2))
		// 1
		// 2
		// 3
		double distance = 0;
		for (int i = 0; i < p.rows(); i++) {
			double pi = p.get(i, 0)[0];
			double qi = q.get(i, 0)[0];
			double d = pi - qi;
			distance += d * d;
		}
		
		//double d2 = Core.norm(p, q, Core.NORM_L2);
		distance = Math.sqrt(distance);
		return distance;
	}

}
