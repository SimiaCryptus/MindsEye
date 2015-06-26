package com.simiacryptus.mindseye;

import java.util.ArrayList;
import java.util.List;

public class PipelineNetwork extends NNLayer {
  private List<NNLayer> layers = new ArrayList<NNLayer>();
  {
  }
  
  @Override
  public NNResult eval(NNResult array) {
    NNResult r = array;
    for (NNLayer l : layers)
      r = l.eval(r);
    return r;
  }
  
  public PipelineNetwork add(NNLayer layer) {
    layers.add(layer);
    return this;
  }
  
  public void test(NDArray[][] samples, int maxIterations, double convergence) {
    PipelineNetwork net = this;
    double rms = 0;
    for (int i = 0; i < samples.length; i++) {
      NDArray input = samples[i][0];
      NDArray output = samples[i][1];
      rms += net.eval(input).errRms(output);
    }
    TestNetworkUnit.log.info("RMS Error: {}", rms);
    for (int i = 0; i < maxIterations; i++)
    {
      rms = 0;
      for (int j = 0; j < samples.length; j++) {
        NDArray input = samples[j][0];
        NDArray output = samples[j][1];
        double rate = getRate(i);
        net.eval(input).learn(rate, output);
        rms += net.eval(input).errRms(output);
      }
      TestNetworkUnit.log.info("RMS Error: {}", rms);
      if(rms<convergence) return;
    }
    throw new AssertionError();
  }

  private double rate = 0.01;
  public double getRate(int iteration) {
    return rate;
  }

  public double getRate() {
    return rate;
  }

  public PipelineNetwork setRate(double rate) {
    this.rate = rate;
    return this;
  }
  
}