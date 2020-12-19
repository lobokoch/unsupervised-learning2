package br.furb;

import java.awt.Image;
import java.util.Timer;
import java.util.TimerTask;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class FaceDetectionWithCamera {
	
	private CascadeClassifier faceClassifier;
	private CascadeClassifier eyeClassifier;
	private CascadeClassifier smileClassifier;
	
	private VideoCapture camera;
	private Timer timer;
	private JFrame frame;
	private JPanel panel;
	private Image image;
	
	public FaceDetectionWithCamera() {
		init();
	}

	private void init() {
		final String cascadePath = "D:\\mk\\OpenCV440\\install\\etc\\haarcascades\\";
		faceClassifier = new CascadeClassifier(cascadePath + "haarcascade_frontalface_alt.xml");
		eyeClassifier = new CascadeClassifier(cascadePath + "haarcascade_eye.xml");
		smileClassifier = new CascadeClassifier(cascadePath + "haarcascade_smile.xml");
		
		camera = new VideoCapture();
		camera.open(0);
		
		buildFrame();
		
		timer = new Timer();
	}

	private void buildFrame() {
		frame = new JFrame();
		frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		frame.setSize(800, 600);
		frame.setTitle("Face detection with OpenCV.");
		frame.setLocationRelativeTo(null);
		
		panel = new JPanel() {
			
			private static final long serialVersionUID = 1L;

			@Override
			public void paintComponent(java.awt.Graphics g) {
				super.paintComponent(g);
				
				if (image != null) {
					g.drawImage(image, 0, 0, 800, 600, null);
				}
				
			};
		};
		
		frame.add(panel);
		frame.setVisible(true);
	}
	
	public void start() {
		long delay = 0;
		long period = 33;
		
		timer.schedule(new TimerTask() {
			
			@Override
			public void run() {
				Mat img = new Mat();
				camera.read(img);
				detectAndShow(img);
			}
			
		}, delay, period);
	}
	
	private void detectAndShow(Mat img) {
		
		MatOfRect faceObjects = new MatOfRect();
		MatOfRect eyeObjects = new MatOfRect();
		MatOfRect smileObjects = new MatOfRect();
		
		Scalar white = new Scalar(255, 255, 255);
		Scalar yellow = new Scalar(0, 255, 0);
		
		faceClassifier.detectMultiScale(img, faceObjects);
		
		faceObjects.toList().forEach(faceRect -> {
			Scalar red = new Scalar(0, 0, 255);
			Imgproc.rectangle(img, faceRect, red, 3);
			
			//Mat faceROI = img.submat(faceRect);
			
			Point p1 = new Point(faceRect.x, faceRect.y);
			Point p2 = new Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height / 2);
			Rect eyeRectROI = new Rect(p1, p2);
			Mat eyeROI = img.submat(eyeRectROI);
			Imgproc.rectangle(img, eyeRectROI, white, 2);
			
			eyeClassifier.detectMultiScale(eyeROI, eyeObjects);
			eyeObjects.toList().forEach(eyeRect -> {
				
				int x = faceRect.x + eyeRect.x;
				int y = faceRect.y + eyeRect.y;
				eyeRect.x = x;
				eyeRect.y = y;
				
				Imgproc.rectangle(img, eyeRect, white, 2);
			});
			
			// Smiles
			Point p11 = new Point(faceRect.x, faceRect.y + faceRect.height/2);
			Point p22 = new Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height);
			
			Rect smileRectROI = new Rect(p11, p22);
			Imgproc.rectangle(img, smileRectROI, yellow, 2);
			
			Mat smileROI = img.submat(smileRectROI);
			
			smileClassifier.detectMultiScale(smileROI, smileObjects);
			smileObjects.toList().forEach(smileRect -> {
				int x = faceRect.x + smileRect.x;
				int y = (int)p11.y + smileRect.y;
				
				x += smileRect.width / 2;
				y += smileRect.height / 2;
				
				RotatedRect box1 = new RotatedRect(new Point(x, y), new Size(smileRect.width, smileRect.height), 0); 
				Imgproc.ellipse(img, box1, yellow, 2);			
				//Imgproc.rectangle(img, smileRect, yellow, 2);			
			});
			
			
		});
		
		
		
		
		image = HighGui.toBufferedImage(img);
		panel.repaint();
	}

	public static void main(String[] args) {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		FaceDetectionWithCamera faceDetection = new FaceDetectionWithCamera();
		faceDetection.start();

	}

}
