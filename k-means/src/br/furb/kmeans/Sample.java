package br.furb.kmeans;

import java.util.Arrays;

public class Sample {

	private double[] data;
	private String label;
	
	public Sample buildClone() {
		Sample clone = new Sample();
		clone.label = label;
		
		clone.data = new double[data.length];
		System.arraycopy(data, 0, clone.data, 0, data.length);
		
		return clone;
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
	
	
	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("Sample [data=");
		builder.append(Arrays.toString(data));
		builder.append(", label=");
		builder.append(label);
		builder.append("]");
		return builder.toString();
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
	
	
}
