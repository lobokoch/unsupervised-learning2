package br.furb;

import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class PCAEigenFace {
	
	private int numComponents;
	private Mat mean;
	
	public PCAEigenFace(int numComponents) {
		this.numComponents = numComponents;
	}
	
	public void train(List<Person> train) {
		calcMean(train);
		/*calcDiff();
		calcCovariance();
		calcEigen();
		calcEigenFaces();
		calcProjections();*/
	}

	private void calcMean(List<Person> train) {
		Mat sample = train.get(0).getData();
		mean = Mat.zeros(/*6400*/sample.rows(), /*1*/sample.cols(), /*CvType.CV_64FC1*/sample.type());
		
		/// Begin Calculado na mão
		train.forEach(person -> {
			Mat data = person.getData();
			for (int i = 0; i < mean.rows(); i++) {
				double mv = mean.get(i, 0)[0]; // Obtém o valor da célula no primeiro canal.
				double pv = data.get(i, 0)[0]; // Obtém o valor da célula no primeiro canal.
				mv += pv;
				mean.put(i, 0, mv);
			}
		});
		
		int M = train.size();
		for (int i = 0; i < mean.rows(); i++) {
			double mv = mean.get(i, 0)[0]; // Obtém o valor da célula no primeiro canal.
			mv /= M;
			mean.put(i, 0, mv);
		}
		/// End Calculado na mão
		
		// Begin OpenCV
		// 1 9 7
		// 4 5 8
		// 7 8 0
		// 6 1 7
		Mat src = new Mat(sample.rows(), train.size(), sample.type());
		for (int i = 0; i < train.size(); i++) {
			train.get(i).getData().col(0).copyTo(src.col(i));
		}
		
		Mat mean2 = Mat.zeros(sample.rows(), sample.cols(), sample.type());
		Core.reduce(src, mean2, /*0=linha, 1=coluna*/1, Core.REDUCE_AVG, mean.type());
		// End OpenCV
		
		saveImagem(mean, "D:\\PCA\\mean1.jpg");
		saveImagem(mean2, "D:\\PCA\\mean2.jpg");
		
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

}
