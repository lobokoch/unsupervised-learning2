package br.furb;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class FaceDetection {

	public static void main(String[] args) {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		/*Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
		System.out.println(mat.dump());*/
		
		final String cascadeFile = "D:\\mk\\OpenCV440\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
		CascadeClassifier classifier = new CascadeClassifier(cascadeFile);
		
		String filename = "D:\\git\\unsupervised-learning2\\opencv\\dataset\\avengers.jpg";
		Mat detected = detectFaces(filename, classifier);
		Imgcodecs.imwrite("D:\\git\\unsupervised-learning2\\opencv\\dataset\\avengers2.jpg", detected);
		
		System.out.println("DONE!");

	}

	private static Mat detectFaces(String filename, CascadeClassifier classifier) {
		Mat img = Imgcodecs.imread(filename);
		
		Mat grayImg = new Mat();
		Imgproc.cvtColor(img, grayImg, Imgproc.COLOR_BGR2GRAY);
		
		Imgproc.equalizeHist(grayImg, grayImg);
		//Mat grayImg = img;
		MatOfRect objects = new MatOfRect();
		classifier.detectMultiScale(grayImg, objects, 1.1, 15);
		
		objects.toList().forEach(rect -> {
			Scalar red = new Scalar(0, 0, 255);
			Imgproc.rectangle(img, rect, red, 3);
		});
		
		return img;
	}

}
