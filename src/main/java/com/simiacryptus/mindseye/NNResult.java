package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class NNResult {

  public final NDArray data;

  public NNResult(NDArray data) {
    super();
    this.data = data;
  }
  
  public boolean isAlive() {
    return false;
  }
  
  public void feedback(NDArray data, FeedbackContext ctx) {
  }

  public final void feedback(NDArray data) {
    feedback(data, new FeedbackContext());
  }

  public final void learn(double d, NDArray out) {
    feedback(delta(d, out));
  }

  public NDArray delta(double d, NDArray out) {
    NDArray delta = new NDArray(data.getDims());
    Arrays.parallelSetAll(delta.data, i->(out.data[i] - NNResult.this.data.data[i]) * d);
    return delta;
  }

  public final void learn(double d, int k) {
    feedback(delta(d, k));
  }

  public NDArray delta(double d, int k) {
    NDArray delta = new NDArray(data.getDims());
    Arrays.parallelSetAll(delta.data, i->((i==k?1.:0.)-(NNResult.this.data.data[i])) * d);
    return delta;
  }

  public double errMisclassification(int k) {
    int prediction = IntStream.range(0, data.dim()).mapToObj(i->i).sorted(Comparator.comparing(i->data.data[(int) i])).findFirst().get();
    return k==prediction?0:1;
  }

  public double errRms(NDArray out) {
    double[] mapToDouble = IntStream.range(0, data.dim()).mapToDouble(i->Math.pow(NNResult.this.data.data[i] - out.data[i], 2.)).toArray();
    double sum = DoubleStream.of(mapToDouble).average().getAsDouble();
    return Math.sqrt(sum);
  }
  
}
