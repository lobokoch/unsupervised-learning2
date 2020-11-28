package br.furb.kmeans;

import java.util.Arrays;

public class Sample {

	private double[] data;
	private String label;
	private String originalLabel;
	private int sumDataTimes;
	private double silhouette;
	
	public Sample buildClone() {
		Sample clone = new Sample();
		clone.label = label;
		clone.originalLabel = originalLabel;
		
		clone.data = new double[data.length];
		System.arraycopy(data, 0, clone.data, 0, data.length);
		
		return clone;
	}
	
	public void restData() {
		data = new double[data.length];
		sumDataTimes = 0;
	}
	
	public double[] getData() {
		return data;
	}
	
	public void setData(double[] data) {
		this.data = data;
	}
	
	public String getLabel() {
		return label;
	}
	
	public Sample setLabel(String label) {
		this.label = label;
		return this;
	}

	public double getDistance(Sample other) {
		// Cálculo da distância Euclidiana
		// d = sqrt(sum((pi-qi)^2))
		double result = 0;
		for (int i = 0; i < data.length; i++) {
			double pi = data[i];
			double qi = other.data[i];
			double d = pi - qi;
			result += d * d;
		}
		
		return Math.sqrt(result);
	}

	public void sumData(double[] moreData) {
		for (int i = 0; i < moreData.length; i++) {
			data[i] += moreData[i];
		}
		
		sumDataTimes++;
	}
	
	public void applyDataMean() {
		for (int  i = 0; i < data.length; i++) {
			data[i] /= sumDataTimes;
		}
	}

	public String getOriginalLabel() {
		return originalLabel;
	}

	public void setOriginalLabel(String originalLabel) {
		this.originalLabel = originalLabel;
	}


	public double getSilhouette() {
		return silhouette;
	}

	public void setSilhouette(double silhouette) {
		this.silhouette = silhouette;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("Sample [data=");
		builder.append(Arrays.toString(data));
		builder.append(", label=");
		builder.append(label);
		builder.append(", originalLabel=");
		builder.append(originalLabel);
		builder.append(", sumDataTimes=");
		builder.append(sumDataTimes);
		builder.append(", silhouette=");
		builder.append(silhouette);
		builder.append("]");
		return builder.toString();
	}
	
	
	
}
