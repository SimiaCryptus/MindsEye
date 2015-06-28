package com.simiacryptus.mindseye;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.layers.DenseSynapseLayer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.test.TestNetworkUnit;

public class PipelineNetwork extends NNLayer {
  static final Logger log = LoggerFactory.getLogger(TestNetworkUnit.class);

  private List<NNLayer> layers = new ArrayList<NNLayer>();
  private double lastRms = Double.MAX_VALUE;
  private boolean verbose = false;
  private int timesSinceImprovement = 0;
  int improvementStaleThreshold = 20;
  
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
  
  public void test(NDArray[][] samples, int maxIterations, double convergence, int trials) {
    for (int epoch = 0; epoch < trials; epoch++)
    {
      // BUG: The previous network's state ensures future trials succeed immediately. 
      double rms = train(samples, maxIterations, convergence);
      log.info("RMS Error: {}", rms);
      if (rms >= convergence) {
        throw new RuntimeException("Failed in trial " + epoch);
      }
    }
  }

  protected double train(NDArray[][] samples, int maxIterations, double convergence) {
    PipelineNetwork net = this;
    double rms = 0;
    for (int i = 0; i < samples.length; i++) {
      NDArray input = samples[i][0];
      NDArray output = samples[i][1];
      rms += net.eval(input).errRms(output);
    }
    rms /= samples.length;
    log.info("RMS Error: {}", rms);
    for (int i = 0; i < maxIterations; i++)
    {
      if(shouldMutate(i,rms)){
        mutate();
      }
      rms = 0;
      for (int j = 0; j < samples.length; j++) {
        NDArray input = samples[j][0];
        NDArray output = samples[j][1];
        double rate = getRate(i);
        NNResult eval = net.eval(input);
        rms += eval.errRms(output);
        NDArray delta = eval.delta(rate, output);
        eval.feedback(delta);
      }
      rms /= samples.length;
      if (rms < convergence) break;
      if(isVerbose()) log.info("RMS Error: {}", rms);
    }
    return rms;
  }
  
  protected boolean shouldMutate(int i, double rms) {
    boolean improved = (lastRms * 1.) < rms;
    lastRms = rms;
    if(improved) {
      return timesSinceImprovement++>improvementStaleThreshold;
    }
    else
    {
      timesSinceImprovement = 0;
      return false;
    }
  }

  protected void mutate() {
    layers.stream()
    .filter(l->(l instanceof DenseSynapseLayer))
    .forEach(l->mutate((DenseSynapseLayer)l));
  }

  protected DenseSynapseLayer mutate(DenseSynapseLayer l) {
    Random random = new Random();
    l.addWeights(() -> 0.1 * random.nextGaussian() * Math.exp(Math.random() * 4) / 2);
    return l;
  }
  
  private double rate = 0.00001;
  
  public double getRate(int iteration) {
    double exp = Math.exp(Math.random()*Math.random()*4)/2;
    return rate * exp;
  }
  
  public double getRate() {
    return rate;
  }
  
  public PipelineNetwork setRate(double rate) {
    this.rate = rate;
    return this;
  }

  public boolean isVerbose() {
    return verbose;
  }

  public PipelineNetwork setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
}